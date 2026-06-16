# Commodity Open-Interest-Growth Premium (Hong & Yogo 2012, JFE)
# ============================================================================
# Cross-sectional weekly long/short across the US commodity-futures complex.
#
# MECHANISM (distinct from hedging-pressure DIRECTION, which already FAILED in
# the registry): OI GROWTH = log(OI_t / OI_{t-13w}) proxies the *intensity*
# (magnitude) of hedging/speculative demand flowing into a market. Markets
# absorbing rising demand pay a higher forward risk premium to the liquidity
# providers who take the other side. Hedging-pressure used commercial-net/OI
# (the SIGN); this uses the GROWTH of TOTAL OI (the MAGNITUDE) -> different
# signal entirely. scope='local': OI-growth is native to physical-commodity
# futures; no clean equity/crypto analogue -> NOT a universal mechanism.
#
# ADAPTER NOTE (verified-before-build caveat): the two owned commodity adapters
# below are named/verified in the proposal's gate0_data_check + DATA_CATALOG.md
# but are NOT in the generic equity import line. They are the only PIT sources
# for COT open interest + roll-aware front returns, so they are unavoidable.
#
# FIX (vs prior fail): the prior version called `fut_curve(roots, start=...)`.
# That double-failed: (a) `fut_curve` has no `start` kwarg, and (b) — the real
# bug the traceback exposed — `fut_curve` takes ONE root *string*, not a list,
# so it built `databento/['CL',...]_ohlcv1d.parquet` and raised FileNotFound
# (which `except TypeError` never catches). Both adapters are now called
# PER-ROOT, signature-robust (try `start=`, fall back to positional), each
# output is normalised to a returns/OI series regardless of its concrete shape,
# un-owned roots are skipped, and the START cutoff is applied in pandas after
# loading. We keep only roots that have BOTH a return series and an OI series.
# ============================================================================

from sdk.harness import StrategySpec
from sdk.adapters import cot_positioning, fut_curve
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

STRAT_ID = "comm_oi_growth_xs_ls"
START = "2009-06-01"  # 13w OI lookback + vol warmup before the 2010 sample

# 16 roots (PA dropped: thin rank-2 palladium) grouped by sub-complex.
SECTOR = {
    "CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
    "ZC": "grains", "ZS": "grains", "ZW": "grains", "ZL": "grains", "ZM": "grains",
    "LE": "livestock", "HE": "livestock", "GF": "livestock",
}
ROOTS = list(SECTOR.keys())
SUBCOMPLEXES = ["energy", "metals", "grains", "livestock"]

DEFAULTS = dict(
    oi_lookback_weeks=13,   # PRE-REGISTERED PRIMARY (Hong-Yogo horizon)
    top_q=2.0 / 3.0,        # tercile long-leg entry
    bot_q=1.0 / 3.0,        # tercile short-leg entry
    hysteresis_band=1.0 / 6.0,  # longs exit <0.5pct, shorts exit >0.5pct rank
    min_hold_weeks=2,       # turnover control
    vol_lookback=63,        # ~3m trailing daily vol (inv-vol sizing + vol-target)
    vol_target=0.10,        # ~10%/yr portfolio vol
    vol_cap=3.0,            # max gross scaling (guards tiny-vol blowups)
    cost_bps=8.0,           # realistic micro-futures turnover cost
)


# ----------------------------------------------------------------------------
# Data loading (per-root, signature-robust, shape-robust)
# ----------------------------------------------------------------------------
def _call_adapter(fn, arg, start):
    """Try `fn(arg, start=start)`; fall back to `fn(arg)` on signature mismatch."""
    try:
        return fn(arg, start=start)
    except TypeError:
        return fn(arg)


def _coerce_index(obj):
    obj = obj.copy()
    obj.index = pd.to_datetime(obj.index)
    return obj.sort_index()


def _root_returns(out, root):
    """Normalise ONE fut_curve output into a daily-returns Series for `root`,
    whatever its concrete shape (returns Series, price Series, single-col frame
    named by the root, or an OHLCV frame)."""
    if out is None:
        return None
    if isinstance(out, pd.Series):
        s = pd.to_numeric(out, errors="coerce")
        med = s.abs().median(skipna=True)
        return s.pct_change() if (med is not None and med > 1.0) else s  # price->ret
    df = out
    if df.shape[1] == 0:
        return None
    low = {str(c).lower(): c for c in df.columns}
    # 1) an explicit return column / a column named by the root == already returns
    for key in (root.lower(), f"{root.lower()}_ret", "ret", "return", "returns",
                "front_ret"):
        if key in low:
            return pd.to_numeric(df[low[key]], errors="coerce")
    # 2) a close-like price column -> pct_change (best-effort if not pre-returned)
    for key in ("closeadj", "adj_close", "close", "settle", "last", "px", "price"):
        if key in low:
            return pd.to_numeric(df[low[key]], errors="coerce").pct_change()
    # 3) substring fallbacks
    for c in df.columns:
        if "ret" in str(c).lower():
            return pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if "close" in str(c).lower():
            return pd.to_numeric(df[c], errors="coerce").pct_change()
    # 4) single column -> infer returns vs price by magnitude
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        med = s.abs().median(skipna=True)
        return s.pct_change() if (med is not None and med > 1.0) else s
    return None


def _pick_oi_column(raw, root):
    """Normalise ONE cot_positioning output into an open-interest Series."""
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    low = {str(c).lower(): c for c in raw.columns}
    for cand in (f"{root}_oi", f"{root}_open_interest", "oi", "open_interest",
                 "openinterest", root):
        if str(cand).lower() in low:
            return pd.to_numeric(raw[low[str(cand).lower()]], errors="coerce")
    for c in raw.columns:  # any column tied to the root that mentions OI
        cl = str(c).lower()
        if root.lower() in cl and ("oi" in cl or "interest" in cl):
            return pd.to_numeric(raw[c], errors="coerce")
    return None


def _rets_from_fut(roots, start):
    cols = {}
    for r in roots:
        try:
            out = _call_adapter(fut_curve, r, start)  # per-root: fut_curve(root)
        except Exception:
            continue  # root not in owned Databento pull -> skip
        s = _root_returns(out, r)
        if s is not None:
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().any():
                cols[r] = s
    if not cols:
        raise KeyError("fut_curve returned no usable front-return series for any root")
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.to_datetime(df.index)
    return df.loc[df.index >= pd.Timestamp(start)]


def _oi_from_cot(roots, start):
    cols = {}
    # Try one batched call first (wide {root}_oi frame), then fill gaps per-root.
    try:
        raw = _coerce_index(_call_adapter(cot_positioning, roots, start))
        for r in roots:
            s = _pick_oi_column(raw, r)
            if s is not None and s.notna().any():
                cols[r] = s
    except Exception:
        pass
    for r in roots:
        if r in cols:
            continue
        try:
            raw = _coerce_index(_call_adapter(cot_positioning, r, start))
        except Exception:
            continue
        s = _pick_oi_column(raw, r)
        if s is not None and s.notna().any():
            cols[r] = s
    if not cols:
        raise KeyError("cot_positioning returned no recognizable open-interest columns")
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.to_datetime(df.index)
    return df.loc[df.index >= pd.Timestamp(start)]


def _build_panel(roots, start):
    rets = _rets_from_fut(roots, start)               # daily roll-aware front returns
    oi = _oi_from_cot(list(rets.columns), start)      # weekly Friday-release OI (PIT)
    common = [r for r in rets.columns if r in oi.columns]
    if not common:
        raise KeyError("no roots have BOTH a return series and an open-interest series")
    rets = rets[common].sort_index()
    # OI is known from its release date onward -> reindex onto the daily trading
    # grid and forward-fill (the value persists until the next Friday release).
    daily_oi = oi[common].reindex(rets.index, method="ffill")
    panel = pd.concat([rets.add_suffix("_ret"), daily_oi.add_suffix("_oi")], axis=1)
    panel.index.name = "date"
    return panel


def load_data() -> pd.DataFrame:
    return _build_panel(ROOTS, START)


# scope='local' -> the stage-2 cross-universe battery is NOT run; provided for
# spec completeness only. (Internal breadth is checked via soft expectations.)
def load_gen_data(label) -> pd.DataFrame:
    return load_data()


# ----------------------------------------------------------------------------
# Signal
# ----------------------------------------------------------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}

    roots = [c[:-4] for c in panel.columns if c.endswith("_ret")]
    sub = p.get("subcomplex")
    if sub is not None:  # used only by soft-expectation per-complex breadth probe
        roots = [r for r in roots if SECTOR.get(r) == sub]

    ret = panel[[f"{r}_ret" for r in roots]].astype(float).copy()
    ret.columns = roots
    ret = ret.fillna(0.0)
    oi = panel[[f"{r}_oi" for r in roots]].astype(float).copy()
    oi.columns = roots

    idx = ret.index

    # --- weekly OI-growth signal (PIT: OI already Friday-release dated) -------
    lb = int(p["oi_lookback_weeks"])
    oi_w = oi.ffill().resample("W-FRI").last()
    g = np.log(oi_w / oi_w.shift(lb)).replace([np.inf, -np.inf], np.nan)

    # cross-sectional percentile rank of OI-growth each rebalance Friday
    pr = g.rank(axis=1, pct=True)

    # trailing inverse-vol (daily std), sampled at each Friday
    vol_d = ret.rolling(int(p["vol_lookback"])).std()
    vol_w = vol_d.reindex(g.index, method="ffill")

    # --- hysteresis + min-hold state machine -> leg membership {+1,-1,0} ------
    top_q, bot_q = p["top_q"], p["bot_q"]
    band = p["hysteresis_band"]
    min_hold = int(p["min_hold_weeks"])
    state = pd.DataFrame(0.0, index=pr.index, columns=pr.columns)
    held = {r: 0 for r in pr.columns}
    age = {r: 0 for r in pr.columns}
    for dt in pr.index:
        row = pr.loc[dt]
        for r in pr.columns:
            v = row[r]
            s = held[r]
            if pd.isna(v):
                held[r], age[r] = 0, 0
                continue
            if s == 0:
                if v >= top_q:
                    held[r], age[r] = 1, 1
                elif v <= bot_q:
                    held[r], age[r] = -1, 1
                else:
                    age[r] = 0
            elif s == 1:
                if age[r] >= min_hold and v < top_q - band:
                    held[r], age[r] = 0, 0
                else:
                    age[r] += 1
            else:  # s == -1
                if age[r] >= min_hold and v > bot_q + band:
                    held[r], age[r] = 0, 0
                else:
                    age[r] += 1
        state.loc[dt] = [held[r] for r in pr.columns]

    # --- inverse-vol weights WITHIN each leg, dollar-neutral (gross 1.0) ------
    inv = 1.0 / vol_w.replace(0.0, np.nan)
    lw = inv.where(state > 0)
    sw = inv.where(state < 0)
    lw = lw.div(lw.sum(axis=1), axis=0).fillna(0.0) * 0.5   # long leg gross 0.5
    sw = sw.div(sw.sum(axis=1), axis=0).fillna(0.0) * 0.5   # short leg gross 0.5
    Wt_wk = lw - sw                                          # dollar-neutral

    # weekly targets -> daily (hold between rebalances)
    W_base = Wt_wk.reindex(idx, method="ffill").fillna(0.0)

    # --- PIT vol-target to ~vol_target/yr ------------------------------------
    # scale uses ONLY trailing info: base strategy return is computed with a
    # 1-day execution lag, its trailing vol is shifted one more day, so the
    # scale applied at date t is known strictly before t.
    tgt_daily = p["vol_target"] / np.sqrt(252.0)
    r_base = (W_base.shift(1) * ret).sum(axis=1)
    realized = r_base.rolling(int(p["vol_lookback"])).std().shift(1)
    scale = (tgt_daily / realized).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(upper=p["vol_cap"]).ffill().fillna(0.0)
    W_target = W_base.mul(scale, axis=0)                     # same-day target weights

    # --- 1-day execution lag is OUR responsibility: weights are decided at the
    # Friday close and traded the next session, so W = W_target.shift(1) is the
    # already-lagged matrix passed (unshifted) to BOTH kit functions. ----------
    W = W_target.shift(1).fillna(0.0)

    daily = net_of_cost(W, ret, cost_bps=p["cost_bps"], name=STRAT_ID)
    trades = trades_from_weights(W, ret, SECTOR)  # kit stamps entry_regime

    # trim leading flat warmup
    active = W.abs().sum(axis=1) > 0
    if active.any():
        daily = daily.loc[active.idxmax():]
    daily = daily.fillna(0.0)
    daily.name = STRAT_ID
    return daily, trades


# ----------------------------------------------------------------------------
# Soft expectations (machine-checked; non-blocking but falsify the story)
# ----------------------------------------------------------------------------
def _check_subcomplex_breadth(ctx):
    # generalization_plan claim: the edge is consistent across sub-complexes,
    # not one lucky complex. Cheap: from the search-window ledger, no extra call.
    trades = ctx.get("trades") or []
    if not trades:
        return {"pass": False, "observed": "no trades"}
    df = pd.DataFrame(trades)
    by = df.groupby("sector")["pnl"].sum()
    pos = sum(1 for c in SUBCOMPLEXES if float(by.get(c, 0.0)) > 0.0)
    detail = {c: round(float(by.get(c, 0.0)), 1) for c in SUBCOMPLEXES}
    return {"pass": pos >= 3, "observed": f"{pos}/4 net-positive {detail}"}


def _check_hysteresis_turnover(ctx):
    # pre-reg claim: hysteresis + min-hold control turnover. ONE extra signal()
    # call (no-hysteresis variant) on the holdout-trimmed panel; compare the
    # number of position runs (fewer runs => lower turnover). Both sliced to the
    # search window so the comparison is honest.
    h = pd.Timestamp(ctx["holdout_start"])
    panel = ctx["panel"]
    sub = panel.loc[panel.index < h]
    base_n = len([t for t in (ctx.get("trades") or [])
                  if pd.Timestamp(t["entry_date"]) < h])
    _, nh_trades = signal(sub, hysteresis_band=0.0, min_hold_weeks=1)
    nh_n = len([t for t in nh_trades if pd.Timestamp(t["entry_date"]) < h])
    ratio = (base_n / nh_n) if nh_n else float("nan")
    return {"pass": bool(base_n <= nh_n),
            "observed": f"runs hyst={base_n} vs no-hyst={nh_n} (ratio {ratio:.2f})"}


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
PRE_REG = (
    "Hong & Yogo (2012, JFE) open-interest-growth risk premium, traded "
    "cross-sectionally as a weekly dollar-neutral L/S across 16 US commodity "
    "futures roots (PA dropped: thin rank-2). PRE-REGISTERED PRIMARY: signal = "
    "OI growth = log(OI_t / OI_{t-13w}) from CFTC COT TOTAL open interest "
    "(cot_positioning {root}_oi), point-in-time on the FRIDAY RELEASE date "
    "(never the Tuesday data date). Each Friday rank roots cross-sectionally; "
    "LONG the top tercile (highest OI growth) and SHORT the bottom tercile "
    "(contracting OI); inverse-vol weighted WITHIN each leg; dollar-neutral; "
    "vol-targeted to ~10%/yr on trailing 63d vol; hysteresis band (longs exit "
    "below the 50th rank pctile, shorts exit above) plus a 2-week min-hold to "
    "cap turnover. Returns are roll-aware within-contract front returns from "
    "fut_curve (consumed per-root; NEVER differenced across a roll). Weekly "
    "rebalance, 1-day execution lag, 8bps micro-futures turnover cost. ECONOMIC "
    "SEPARATION: this is the demand-INTENSITY (magnitude of total participation) "
    "channel and is economically distinct from the hedging-pressure DIRECTION "
    "(commercial-net sign) channel already tested and FAILED in the registry; "
    "carry / basis-momentum / value / skew / seasonality / storage are "
    "separately registered. SCOPE=local: the mechanism is native to physical-"
    "commodity futures (no clean equity/crypto analogue). VALIDATION PLAN: (1) "
    "internal breadth -- require >=3 of 4 sub-complexes (energy/metals/grains/"
    "livestock) net-positive in the search window, not one lucky complex; (2) "
    "the book is market-neutral L/S so the absolute-Sharpe MCPT time-shuffle "
    "null applies (must beat permutation to rule out a construction artifact); "
    "(3) forward-validate the deployable micro-futures book on the 2022+ "
    "holdout. EXCLUDED from the primary (documented follow-ons only): eia/usda "
    "storage overlays and the boreas TSMOM crisis-tail overlay -- tested "
    "STANDALONE first; a trend hedge is added ONLY if it cuts the tail without "
    "diluting the standalone Sharpe (anti-pattern 2026-06-08: no reflexive "
    "50/50 blend)."
)

SPEC = StrategySpec(
    id=STRAT_ID,
    family="commodity_positioning",
    title=("Commodity open-interest-growth premium (Hong-Yogo 2012) — "
           "cross-sectional weekly L/S across the commodity complex"),
    markets=["commodity_futures"],
    data_desc=(
        "CFTC COT total open interest (cot_positioning {root}_oi, weekly, "
        "Friday-release PIT join) for the signal; Databento front-contract "
        "within-contract roll-aware returns (fut_curve, called per-root) for "
        "the tradable book. 16 roots: energy CL/NG/HO/RB, metals GC/SI/HG/PL, "
        "grains ZC/ZS/ZW/ZL/ZM, livestock LE/HE/GF (PA dropped). Roots absent "
        "from the owned pull are skipped; only roots with BOTH returns and OI "
        "enter the cross-section."
    ),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={},  # primary == DEFAULTS (13w, tercile, hysteresis on)
    grid={
        "default": {},
        "lookback_8w": {"oi_lookback_weeks": 8},
        "lookback_26w": {"oi_lookback_weeks": 26},
        "quartile": {"top_q": 0.75, "bot_q": 0.25},
        "no_hysteresis": {"hysteresis_band": 0.0, "min_hold_weeks": 1},
    },
    scope="local",
    generalization_universes=[],  # local edge -> internal-breadth check instead
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=14,
    expectations=[
        {
            "name": "subcomplex_breadth",
            "claim": ">=3 of 4 sub-complexes (energy/metals/grains/livestock) "
                     "net-positive in the search window (not one lucky complex)",
            "check": _check_subcomplex_breadth,
        },
        {
            "name": "hysteresis_cuts_turnover",
            "claim": "hysteresis + 2-week min-hold yields <= as many position "
                     "runs as the no-hysteresis variant (lower turnover)",
            "check": _check_hysteresis_turnover,
        },
    ],
)