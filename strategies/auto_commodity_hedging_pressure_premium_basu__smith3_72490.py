# Commodity hedging-pressure premium (Basu-Miffre, 2013) — REAL COT commercial positioning.
# Cross-sectional L/S on the owned futures shelf. NO side effects: pure returns+trades producer.
#
# Mechanism (Keynes "normal backwardation"): speculators are PAID to absorb the net hedging
# demand of commercial producers/consumers. We measure that demand DIRECTLY (CFTC commercial
# net / open interest) rather than proxying it with past returns (which is what the FAILED
# hedging_pressure_footprint_ls_v2 did — that null falsified the PROXY, not the premium; this
# is the anti-pattern-#3 axis-change onto fundamentally new, now-owned data).
#
# DATA SHELF (proposal-specified, both OWNED -> $0 incremental; these are the tested adapters
# for this data per the proposal's gate0, COT is free, Databento curve purchased 2026-06-12):
#   cot_positioning(roots, field) -> weekly wide panel indexed by CFTC *RELEASE* date
#       (look-ahead pre-closed in the adapter); fields 'comm_net' (commercial long-short) & 'oi'.
#   fut_curve(roots, field='ret')  -> daily wide panel of FRONT-contract returns computed
#       WITHIN each contract (roll-aware: held to ~5d before expiry then rolled, never across a
#       roll). The roll engine lives in the adapter on purpose — re-implementing it by hand is a
#       fresh chance for a roll/look-ahead bug the rails can't see (MANDATORY-KIT philosophy).
#   NB: these adapters take roots (+ field) only — NOT a `start` kwarg. We clip the date range
#       to START by index slicing AFTER load (the prior version crashed passing start=...).

from sdk.harness import StrategySpec
from sdk.adapters import cot_positioning, fut_curve
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

SPEC_ID = "comm_hedging_pressure_cot_ls_v1"
START   = "2013-01-01"   # gate0(4): grains/livestock curve coverage starts 2012-13 -> run 2013+ for >=10 live roots

# --- sector map (commodity "sectors" the trade ledger needs for the cross-regime/spread gates) ---
SECTOR_MAP = {
    "CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "precious", "SI": "precious", "PL": "precious", "HG": "base_metals",
    "ZC": "grains", "ZW": "grains", "ZS": "oilseeds", "ZL": "oilseeds", "ZM": "oilseeds",
    "LE": "livestock", "HE": "livestock", "GF": "livestock",
}

# SEARCH universe: energy + grains + oilseeds (9 roots, 3 distinct sectors -> a real cross-section,
# ~3 names/leg). The full-16 book is the *deployment/research narrative*; the SEARCH set is held
# DISJOINT from the generalization sets so the broad-scope stage-2 battery is honest.
SEARCH_ROOTS = ["CL", "NG", "HO", "RB", "ZC", "ZW", "ZS", "ZL", "ZM"]

# GENERALIZATION universes — DISJOINT from SEARCH (share NO tickers): entirely untouched metals
# & livestock sectors. The premium is a UNIVERSAL insurance-provision mechanism -> it must appear
# in markets we never searched, or it's a sector-bound outlier (2026-06-09 BAB lesson).
GEN_UNIVERSES = {
    "metals_livestock": ["GC", "SI", "HG", "PL", "LE", "HE", "GF"],  # 7-root cross-section
    "metals":           ["GC", "SI", "HG", "PL"],                    # precious + base
    "livestock":        ["LE", "HE", "GF"],                          # cattle/hogs/feeders
}

DEFAULTS = dict(
    hp_lookback_weeks=52,   # per-root rolling window: the signal is the DEVIATION in hedging demand,
    hp_min_weeks=26,        #   not the level (handles structurally-net-short commercials cleanly)
    quantile=1.0 / 3.0,     # terciles
    min_roots=3,            # need >=3 live roots to form a cross-section
    vol_lb=60,              # inverse-vol lookback (days)
    target_vol=0.10,        # 10% annualized portfolio vol
    gross_cap=2.0,          # gross <= 2x
    cost_bps=8.0,           # ~8bps on turnover (conservative for liquid futures: real tick≈1-3bps)
)

# pre-declared grid for the DSR effective-N (honest search burden); "default"={} is primary.
GRID = {
    "default":      {},
    "lookback_26w": {"hp_lookback_weeks": 26},
    "quartile":     {"quantile": 0.25},
    "vol_90d":      {"vol_lb": 90},
}


# ----------------------------- data assembly -----------------------------
def _build_panel(roots):
    """One MultiIndex-column panel: level0 in {'ret','comm_net','oi'}, level1 = root. Daily index.

    Adapters take (roots, field) only — no `start` kwarg — so we clip to START by index slicing."""
    roots = list(roots)
    start_ts = pd.Timestamp(START)
    ret = fut_curve(roots, field="ret").sort_index()
    cn  = cot_positioning(roots, field="comm_net").sort_index()
    oi  = cot_positioning(roots, field="oi").sort_index()
    ret = ret[ret.index >= start_ts]                         # date clip post-load (no start kwarg)
    keep = [r for r in roots if (r in ret.columns) and (r in cn.columns) and (r in oi.columns)]
    ret = ret.reindex(columns=keep)
    idx = ret.index
    # COT is RELEASE-date indexed -> ffill forward to daily is look-ahead-free (value at d known at d).
    cn = cn.reindex(columns=keep).reindex(idx, method="ffill")
    oi = oi.reindex(columns=keep).reindex(idx, method="ffill")
    panel = pd.concat({"ret": ret, "comm_net": cn, "oi": oi}, axis=1)
    panel.index.name = "date"
    return panel


def load_data():
    return _build_panel(SEARCH_ROOTS)


def load_gen_data(label):
    return _build_panel(GEN_UNIVERSES[label])


# ----------------------------- signal helpers -----------------------------
def _hp_percentile(panel, lookback_weeks, min_weeks):
    """Per-root 52w rolling percentile of hedging pressure HP = comm_net / OI (low pct = commercials
    most net-short = hedgers paying longs to carry inventory risk). Computed on the native weekly
    grid (resample undoes the daily ffill -> no rolling-window distortion), then ffilled to daily."""
    cn, oi = panel["comm_net"], panel["oi"]
    hp = (cn / oi.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    hp_w = hp.resample("W-FRI").last()
    pct_w = hp_w.rolling(lookback_weeks, min_periods=min_weeks).rank(pct=True)
    return pct_w.reindex(panel.index, method="ffill")


def _xs_masks(hp_pct, quantile, min_roots):
    """Balanced cross-sectional terciles by integer rank (robust on thin gen universes):
    LONG = lowest-HP-percentile names, SHORT = highest. Equal count per leg."""
    r = hp_pct.rank(axis=1, method="first", ascending=True)   # 1 = lowest HP pct = LONG candidate
    n = hp_pct.notna().sum(axis=1)
    k = (n * quantile).round().clip(lower=1)                  # names per leg
    longm  = r.le(k, axis=0) & hp_pct.notna()
    shortm = r.gt((n - k), axis=0) & hp_pct.notna()
    valid = (n >= min_roots).astype(int)
    longm  = longm.mul(valid, axis=0).astype(bool)
    shortm = shortm.mul(valid, axis=0).astype(bool)
    return longm, shortm


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    ret = panel["ret"].astype(float)

    # 1) hedging-pressure deviation percentile (already release-date-lagged via the COT index)
    hp_pct = _hp_percentile(panel, p["hp_lookback_weeks"], p["hp_min_weeks"])

    # 2) cross-sectional sort across roots: LONG commercials-most-net-short / SHORT least
    longm, shortm = _xs_masks(hp_pct, p["quantile"], p["min_roots"])

    # 3) inverse-vol risk weight within each leg (60d realized vol of within-contract returns)
    vol = ret.rolling(p["vol_lb"], min_periods=max(20, p["vol_lb"] // 2)).std()
    iv = 1.0 / vol.replace(0.0, np.nan)
    lw = iv.where(longm, 0.0)
    sw = iv.where(shortm, 0.0)
    lw = lw.div(lw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    sw = sw.div(sw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    raw = lw - sw  # dollar-neutral, gross ~2

    # 4) weekly rebalance: freeze target weights to Friday, hold through the week (costs charged weekly)
    raw_w = raw.resample("W-FRI").last().reindex(ret.index, method="ffill").fillna(0.0)

    # 5) portfolio vol target (10% ann.) + gross cap (2x). Uses only trailing returns; whole book is
    #    lagged one day below, so scale[t] (built from returns <= t) is applied to weights held at t+1.
    unscaled = (raw_w * ret).sum(axis=1)
    ann_vol = unscaled.rolling(p["vol_lb"], min_periods=max(20, p["vol_lb"] // 2)).std() * np.sqrt(252.0)
    scale = (p["target_vol"] / ann_vol).replace([np.inf, -np.inf], np.nan)
    gross = raw_w.abs().sum(axis=1).replace(0.0, np.nan)
    scale = scale.clip(upper=(p["gross_cap"] / gross)).fillna(0.0)
    W = raw_w.mul(scale, axis=0).fillna(0.0)

    # 6) LAG 1 day — the lag is OUR responsibility: weights decided from t-1 info are held on t.
    W_held = W.shift(1).fillna(0.0)

    daily = net_of_cost(W_held, ret, cost_bps=p["cost_bps"], name=SPEC_ID)
    trades = trades_from_weights(W_held, ret, SECTOR_MAP)  # kit stamps entry_regime (contract)
    return daily, trades


# ----------------------------- soft expectations (machine-checkable mechanism claims) -----------------------------
def _check_dispersion(ctx):
    """gate0(3): the cross-section is non-degenerate — both terciles populated on (almost) every
    in-sample rebalance date (else 'all roots in the same tercile' kills the L/S)."""
    panel, hs = ctx["panel"], pd.Timestamp(ctx["holdout_start"])
    hp_pct = _hp_percentile(panel, DEFAULTS["hp_lookback_weeks"], DEFAULTS["hp_min_weeks"])
    longm, shortm = _xs_masks(hp_pct, DEFAULTS["quantile"], DEFAULTS["min_roots"])
    ins = hp_pct.index < hs
    active = hp_pct[ins].notna().sum(axis=1) >= DEFAULTS["min_roots"]
    both = (longm[ins].sum(axis=1) >= 1) & (shortm[ins].sum(axis=1) >= 1)
    frac = float(both[active].mean()) if int(active.sum()) else 0.0
    return {"pass": bool(frac >= 0.90), "observed": round(frac, 3)}


def _check_slow_decay(ctx):
    """Structural hedger demand decays slowly (weekly data) -> held positions should span multiple
    weeks. Median hold >= 10 trading days (~2 weeks). Falsifies a 'fast churner' mis-story."""
    tr = ctx["trades"]
    if not tr:
        return {"pass": False, "observed": 0}
    med = float(pd.Series([t["hold_days"] for t in tr]).median())
    return {"pass": bool(med >= 10.0), "observed": med}


def _check_grid_robust(ctx):
    """The edge is not a single lucky param node: >=75% of the pre-declared grid variants are
    positive on the in-sample search window."""
    grid = ctx["grid"]
    if not grid:
        return {"pass": False, "observed": 0.0}
    pos = sum(1 for s in grid.values() if s is not None and len(s) and float(s.mean()) > 0.0)
    frac = pos / len(grid)
    return {"pass": bool(frac >= 0.75), "observed": round(frac, 3)}


# ----------------------------- spec -----------------------------
SPEC = StrategySpec(
    id=SPEC_ID,
    family="hedging_pressure",
    title="Commodity hedging-pressure premium (Basu-Miffre) — real COT commercial positioning, cross-sectional L/S",
    markets=["commodities"],
    data_desc=("CFTC COT commercial net positioning (release-date indexed, look-ahead pre-closed) + "
               "Databento individual-contract futures curve returns (roll-aware, within-contract), "
               "16 CME roots, 2013+. Both OWNED -> $0 incremental."),
    pre_registration=(
        "FROZEN DESIGN: weekly (COT release / Friday), per root compute hedging pressure "
        "HP = commercial_net / open_interest, convert to a per-root 52-week rolling PERCENTILE (the "
        "signal is the DEVIATION in hedging demand, not the level — handles structurally net-short "
        "commercials). Cross-sectionally sort the roots: LONG the lowest-HP-percentile tercile "
        "(commercials most net-short -> hedgers paying longs to carry inventory risk), SHORT the "
        "highest tercile. Equal-count legs, inverse-60d-vol risk weight within each leg, 10% annualized "
        "vol target, gross<=2x, weekly rebalance on release dates, ~8bps on turnover, signals lagged 1 "
        "day (W.shift(1)). PRIMARY = the percentile-tercile spec; no variant selection (grid is the "
        "honest search burden only). Returns are computed WITHIN contracts via fut_curve's roll-aware "
        "front series (never across a roll). "
        "WHY NOT DUPLICATE: hedging_pressure_footprint_ls_v2 FAILED on a 12-month-return PROXY for "
        "positioning (COT was not owned then) — that null falsified the proxy, not the premium. This is "
        "the canonical axis-change onto fundamentally NEW owned data (real CFTC commercial positioning), "
        "distinct from price-derived basis-momentum and from dead DM FX/bond carry. "
        "SCOPE=broad: the insurance-provision mechanism is universal, so the SEARCH set (energy+grains+"
        "oilseeds, 9 roots / 3 sectors) is held DISJOINT from the generalization sets (metals & livestock, "
        "share NO tickers) — a pass MUST survive the stage-2 battery on those untouched sectors or it is a "
        "sector-bound outlier (BAB lesson). Market-neutral book -> MCPT absolute null applies. "
        "STANDALONE FIRST (2026-06-08 lesson): no trend leg is blended here; the pro-cyclical tail of this "
        "insurance premium pairs naturally with Boreas trend, but a hedge is only added later as a SIZED "
        "tail-overlay if it cuts DD without diluting standalone Sharpe — never a reflexive 50/50. "
        "GATE0 expectations: (1) cot_positioning returns comm_net & oi for all roots with no >4wk gaps; "
        "(2) release-date index joins cleanly to daily returns (zero-Tuesday/no-look-ahead holds); "
        "(3) HP percentile has real cross-sectional dispersion (see signal_dispersion check); "
        "(4) run 2013+ for >=10 live roots; (5) CL roll-chain integrity verified by hand. "
        "DEPLOYMENT: full 16-root book is the research spec; the retail spec is a 6-8 market top/bottom "
        "book at matched risk weights via CME micros (MCL/MGC/SIL/MHG/mini grains), both legs exchange-"
        "listed (no borrow), gross<=2x — checked in deployment_sanity."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},                 # default = frozen primary spec (DEFAULTS applied inside signal)
    grid=GRID,
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "signal_dispersion",
         "claim": "cross-section non-degenerate: both terciles populated on >=90% of in-sample rebalance dates",
         "check": _check_dispersion},
        {"name": "slow_decay_hold",
         "claim": "weekly structural hedger demand -> median held-position length >= 10 trading days (~2 weeks)",
         "check": _check_slow_decay},
        {"name": "grid_robust",
         "claim": ">=75% of pre-declared grid variants are positive on the in-sample search window",
         "check": _check_grid_robust},
    ],
)