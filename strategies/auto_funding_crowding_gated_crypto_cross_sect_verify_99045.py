"""
Crypto cross-sectional SHORT-HORIZON REVERSAL (a pro-cyclical liquidity-provision
premium in the liquid majors) fused with a LEADING leverage-crowding RISK-OFF gate
that amputates the deleveraging left tail.

ONE FROZEN SPEC, ONE DOLLAR-NEUTRAL BOOK.

HONEST DATA DISCLOSURE (read pre_registration): the proposal's leverage thermometer is
aggregate perp FUNDING. No crypto-funding adapter is in the tested-import whitelist
(only yf_panel/sep_panel/sf1/fred/trend/inv_vol are owned/tested), so funding is NOT
fetched. The thermometer L_t is implemented as a PRICE-ONLY PROXY that captures the
SAME economic configuration the funding gate targets ("crowded one-sided leverage
building under a calm surface"): crowded leveraged longs grind price higher while
paying funding, so a strong, persistent directional drift accrued under SUPPRESSED
realized vol is the price footprint of an over-levered, deleverage-prone tape.
The harness evaluates the thing actually implemented; this module is that thing.

NO external side effects (no file writes, no capital, no config). Owned/FREE data only.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------------
SPEC_ID    = "crypto_xs_reversal_funding_gate"
DATA_START = "2018-01-01"

# Liquid crypto-USD majors (yfinance pairs; survivorship not an issue at this cap tier
# but eligibility is point-in-time below: a coin enters U_t only after >=60d of history).
SEARCH_UNIVERSE = [f"{k}-USD" for k in
    ["BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "LTC", "LINK", "AVAX", "ADA"]]

# >=3 generalization universes, each DISJOINT from search AND from each other (broad scope).
# A real liquidity-provision/reversal premium should re-appear in secondary deep alts.
GEN_UNIVERSES = {
    "alts_a":  [f"{k}-USD" for k in ["DOT","MATIC","ATOM","NEAR","UNI","ETC","XLM","ALGO","FIL","VET"]],
    "alts_b":  [f"{k}-USD" for k in ["AAVE","MKR","SAND","MANA","AXS","GRT","EGLD","THETA","CHZ","ENJ"]],
    "defi_l2": [f"{k}-USD" for k in ["BCH","XMR","EOS","XTZ","NEO","DASH","ZEC","COMP","SNX","CRV"]],
}

# Honest "sector" taxonomy so the trade ledger's spread / single-name gates have meaning.
_SECTOR_BASE = {
    "BTC":"Store-of-Value","ETH":"Smart-Contract","BNB":"Exchange","XRP":"Payments",
    "SOL":"Smart-Contract","DOGE":"Meme","LTC":"Payments","LINK":"Oracle",
    "AVAX":"Smart-Contract","ADA":"Smart-Contract",
    "DOT":"Interop","MATIC":"L2-Scaling","ATOM":"Interop","NEAR":"Smart-Contract",
    "UNI":"DeFi","ETC":"Smart-Contract","XLM":"Payments","ALGO":"Smart-Contract",
    "FIL":"Storage","VET":"Supply-Chain",
    "AAVE":"DeFi","MKR":"DeFi","SAND":"Gaming","MANA":"Gaming","AXS":"Gaming",
    "GRT":"Infrastructure","EGLD":"Smart-Contract","THETA":"Media","CHZ":"Fan-Token","ENJ":"Gaming",
    "BCH":"Payments","XMR":"Privacy","EOS":"Smart-Contract","XTZ":"Smart-Contract",
    "NEO":"Smart-Contract","DASH":"Privacy","ZEC":"Privacy","COMP":"DeFi","SNX":"DeFi","CRV":"DeFi",
}
SECTOR_MAP = {f"{k}-USD": v for k, v in _SECTOR_BASE.items()}

# default = primary; the rest are the PRE-REGISTERED robustness perturbations
# (formation 3/5/7, rebalance 2/3/5, gate pct 85/90/95, rv-low median/40th, block 5/10/15).
# Declared upfront so the DSR effective-N reflects the true search burden (NOT selection).
DEFAULT_PARAMS = dict(formation=5, rebalance=3, gate_pct=90, rv_low_q=0.50,
                      block_days=10, rv_win=10, lev_win=20, min_names=6, cost_bps=10.0)
GRID = {
    "default":  {},
    "form3":    {"formation": 3},
    "form7":    {"formation": 7},
    "reb2":     {"rebalance": 2},
    "reb5":     {"rebalance": 5},
    "gate85":   {"gate_pct": 85},
    "gate95":   {"gate_pct": 95},
    "rvlow40":  {"rv_low_q": 0.40},
    "block5":   {"block_days": 5},
    "block15":  {"block_days": 15},
}

CRYPTO_YEAR = 365.0  # crypto trades every calendar day


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _eligible(px: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time membership: valid price today AND >=60 prior valid observations."""
    valid = px.notna()
    return valid & (valid.cumsum() >= 60)


def _risk_off_series(rets: pd.DataFrame, eligible: pd.DataFrame, p: dict):
    """
    Composite RISK_OFF, computed same-day (uses data through close t; the strategy's
    single shift(1) lags every position by one day, so the gate acts on t-1 closes).

      L_t  = funding PROXY = annualized trailing-`lev_win` mean return of the EW major
             basket (signed leverage-crowding thermometer; |L| extreme = one-sided crowd).
      RV_t = annualized `rv_win`-day realized vol of the EW major basket.

      LEADING_FLAG     = |L_t| >= trailing-252d `gate_pct` pctile of |L|   (crowded leverage)
                         AND RV_t <= trailing-252d `rv_low_q` quantile of RV (calm surface).
      BLOWOUT_FALLBACK = RV_t >= trailing-252d 90th pctile of RV (Parent-1 lagging gate).
      RISK_OFF         = LEADING_FLAG OR BLOWOUT_FALLBACK.
      blocked          = any RISK_OFF fire within the last `block_days` days  (flatten+block).
    """
    win = 252
    basket = rets.where(eligible).mean(axis=1)
    RV = basket.rolling(int(p["rv_win"])).std() * np.sqrt(CRYPTO_YEAR)
    L  = basket.rolling(int(p["lev_win"])).mean() * CRYPTO_YEAR
    absL = L.abs()

    q_absL = absL.rolling(win, min_periods=60).quantile(float(p["gate_pct"]) / 100.0)
    rv_low = RV.rolling(win, min_periods=60).quantile(float(p["rv_low_q"]))
    rv_hi  = RV.rolling(win, min_periods=60).quantile(0.90)

    leading = (absL >= q_absL) & (RV <= rv_low)
    blowout = (RV >= rv_hi)
    risk_off = (leading | blowout).fillna(False)
    blocked = risk_off.astype(float).rolling(int(p["block_days"]), min_periods=1).max() > 0
    return blocked, risk_off


def _build_weights(panel: pd.DataFrame, p: dict, disable_gate: bool):
    """Same-day target weights (built from closes through t); the 1-day lag is applied
    by the caller via .shift(1). Returns (rets, held_weights_same_day)."""
    px = panel.sort_index().astype(float)
    rets = px.pct_change()
    elig = _eligible(px)

    form = int(p["formation"]); reb = int(p["rebalance"]); min_names = int(p["min_names"])

    # REVERSAL: rank trailing-`form` return; LONG bottom tercile (losers), SHORT top (winners).
    f = (px / px.shift(form) - 1.0).where(elig)
    ranks = f.rank(axis=1, pct=True)                      # high pct = recent winner
    long_mask  = ranks <= (1.0 / 3.0)                     # losers  -> LONG
    short_mask = ranks >= (2.0 / 3.0)                     # winners -> SHORT
    nl = long_mask.sum(axis=1).replace(0, np.nan)
    ns = short_mask.sum(axis=1).replace(0, np.nan)
    Wl = long_mask.div(nl, axis=0) * 0.5                  # +0.5 long leg
    Ws = short_mask.div(ns, axis=0) * 0.5                 # -0.5 short leg
    tw = Wl.sub(Ws, fill_value=0.0).fillna(0.0)           # dollar-neutral, gross = 1.0x
    enough = (elig.sum(axis=1) >= min_names)
    tw = tw.mul(enough.astype(float), axis=0)             # no trade if <6 names

    # rebalance every `reb` days: set target only on rebalance rows, hold (ffill) between.
    pos = np.arange(len(tw))
    reb_rows = (pos % reb) == 0
    tw_reb = tw.copy()
    tw_reb.loc[~reb_rows, :] = np.nan
    held = tw_reb.ffill().fillna(0.0)

    # GATE: when blocked -> FLATTEN (and stay blocked through the window).
    if not disable_gate:
        blocked, _ = _risk_off_series(rets, elig, p)
        held = held.mul((~blocked).astype(float).reindex(held.index).fillna(1.0), axis=0)

    return rets, held


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Daily Close panel for the liquid crypto-USD majors (yf_panel; OWNED/FREE)."""
    return yf_panel(SEARCH_UNIVERSE, start=DATA_START)


def load_gen_data(label) -> pd.DataFrame:
    """Panel for ONE generalization universe (same shape as load_data())."""
    return yf_panel(GEN_UNIVERSES[label], start=DATA_START)


# ----------------------------------------------------------------------------------
# signal
# ----------------------------------------------------------------------------------
def signal(panel, **params):
    """
    Returns (daily net-of-cost returns Series, trades list).
    Lag policy: weights are built SAME-DAY (closes through t), then .shift(1) -> a held
    position on day t reflects decisions from close t-1 (strict 1-day lag, no look-ahead).
    Costs: cost_bps charged per-side on turnover via net_of_cost; default 10bps/side ~=
    20bps round-trip taker (pre-registered). Perp funding on a dollar-neutral book is
    ~symmetric (longs receive ~ shorts pay) so its net drag ~0 and is folded into this
    conservative charge.
    """
    disable_gate = bool(params.pop("disable_gate", False))
    p = dict(DEFAULT_PARAMS); p.update(params)

    rets, held = _build_weights(panel, p, disable_gate)
    W = held.shift(1).fillna(0.0)                         # explicit 1-day lag (responsibility stated)

    name = SPEC_ID + ("__nogate" if disable_gate else "")
    daily = net_of_cost(W, rets, cost_bps=float(p["cost_bps"]), name=name)

    sector_map = {t: SECTOR_MAP.get(t, "Crypto") for t in panel.columns}
    trades = trades_from_weights(W, rets, sector_map)     # kit stamps entry_regime (contract)
    return daily, trades


# ----------------------------------------------------------------------------------
# soft expectations (machine-checkable mechanism claims; <holdout only; <=1 signal call each)
# ----------------------------------------------------------------------------------
def _exp_engine_is_reversal(ctx):
    """Engine is REVERSAL (opposite-sign of momentum): mean cross-sectional rank-corr
    between trailing-5d return and forward-3d return is NEGATIVE on the search window."""
    try:
        h0 = pd.Timestamp(ctx["holdout_start"])
        px = ctx["panel"].sort_index().astype(float)
        px = px[px.index < h0]
        if len(px) < 80:
            return {"pass": False, "observed": "insufficient_history"}
        f = px / px.shift(5) - 1.0
        fwd = px.shift(-3) / px - 1.0                     # sliced first -> never peeks past holdout
        cs = []
        for dt in f.index[::3]:
            a, b = f.loc[dt], fwd.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() >= 6:
                cs.append(a[m].rank().corr(b[m].rank()))
        obs = float(np.nanmean(cs)) if cs else np.nan
        return {"pass": bool(np.isfinite(obs) and obs < 0), "observed": obs}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _exp_gate_cuts_tail(ctx):
    """Gate AMPUTATES the left tail: gated 5% CVaR >= ungated 5% CVaR (less negative)
    on the search window. Uses the one allowed extra signal() call (disable_gate)."""
    try:
        h0 = pd.Timestamp(ctx["holdout_start"])
        gated = ctx["search"].dropna()
        gated = gated[gated.index < h0]
        ung, _ = ctx["spec"].signal(ctx["panel"], disable_gate=True)
        ung = ung[ung.index < h0].dropna()

        def cvar(s, q=0.05):
            if len(s) < 30:
                return np.nan
            thr = s.quantile(q)
            t = s[s <= thr]
            return float(t.mean()) if len(t) else np.nan

        cg, cu = cvar(gated), cvar(ung)
        ok = bool(np.isfinite(cg) and np.isfinite(cu) and cg >= cu)
        return {"pass": ok, "observed": f"gated_CVaR={cg:.5f} | ungated_CVaR={cu:.5f}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _exp_risk_off_targets_drawdowns(ctx):
    """Pre-registered MEASURE (not selection): the RISK_OFF block windows concentrate on
    the UNGATED book's worst days -> overlap on the worst-decile loss days exceeds the
    unconditional block frequency. Uses the one allowed extra signal() call."""
    try:
        h0 = pd.Timestamp(ctx["holdout_start"])
        ung, _ = ctx["spec"].signal(ctx["panel"], disable_gate=True)
        ung = ung[ung.index < h0].dropna()
        if len(ung) < 100:
            return {"pass": False, "observed": "insufficient_history"}
        px = ctx["panel"].sort_index().astype(float)
        rets = px.pct_change()
        blocked, _ = _risk_off_series(rets, _eligible(px), DEFAULT_PARAMS)
        blocked = blocked.reindex(ung.index).fillna(False)
        worst = ung <= ung.quantile(0.10)
        base = float(blocked.mean())
        overlap = float(blocked[worst].mean()) if worst.sum() else np.nan
        ok = bool(np.isfinite(overlap) and overlap > base)
        return {"pass": ok, "observed": f"overlap_on_worst_decile={overlap:.3f} | base_rate={base:.3f}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


def _exp_net_survives_costs(ctx):
    """Reversal is NOT killed by turnover: net (after 20bps round-trip) search Sharpe > 0."""
    try:
        s = ctx["search"].dropna()
        if len(s) < 80 or s.std() == 0:
            return {"pass": False, "observed": "insufficient_or_degenerate"}
        ann = float(s.mean() * CRYPTO_YEAR)
        shp = float(s.mean() / s.std() * np.sqrt(CRYPTO_YEAR))
        return {"pass": bool(shp > 0), "observed": f"net_ann={ann:.3f} | net_sharpe={shp:.2f}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


# ----------------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------------
SPEC = StrategySpec(
    id=SPEC_ID,
    family="xs_reversal",
    title="Funding-crowding-gated crypto cross-sectional reversal (liquidity-provision premium, "
          "left tail amputated by a leading leverage-divergence risk-off gate)",
    markets=["crypto"],
    data_desc="OWNED/FREE: yf_panel daily Close for liquid crypto-USD majors (BTC/ETH/BNB/XRP/SOL/"
              "DOGE/LTC/LINK/AVAX/ADA), 2018+. Spot drives the reversal signal, the realized-vol "
              "surface, the (price-proxied) leverage thermometer, and return accounting (perp~=spot "
              "for EOD signal). NO crypto-funding adapter is whitelisted, so funding is PROXIED from "
              "price (see pre_registration).",
    pre_registration=(
        "MECHANISM. (1) Return engine = pro-cyclical cross-sectional SHORT-HORIZON REVERSAL across "
        "liquid crypto majors: being paid to provide liquidity to transient dislocations (long recent "
        "losers / short recent winners). A risk/liquidity premium (earns in calm range-bound tape, "
        "bleeds in trending deleveraging stress); OPPOSITE sign to naive momentum; funding is NOT "
        "harvested. (2) Defensive overlay = a LEADING leverage-crowding RISK-OFF gate (crypto analog of "
        "a vol-of-vol divergence): leverage thermometer extreme while realized vol is still LOW = "
        "leverage building under a calm surface, the configuration that precedes deleveraging cascades "
        "that steamroll a reversal book. Harvested object = the reversal premium with its deleveraging "
        "left tail amputated.\n"
        "FROZEN SPEC. Universe U_t = majors with >=60d of price history as of t (point-in-time entry; "
        "<6 names => no trade). Every `rebalance` (3) days rank U_t by trailing `formation` (5) day "
        "return; LONG bottom tercile, SHORT top tercile, equal-weight each side, dollar-neutral, "
        "gross=1.0x (0.5 long + 0.5 short); hold to next rebalance. COSTS: cost_bps=10/side on turnover "
        "via net_of_cost ~= 20bps round-trip taker (pre-registered); dollar-neutral perp funding ~"
        "symmetric (net ~0) and folded into this conservative charge. GATE (composite RISK_OFF, "
        "same-day then 1-day-lagged with the whole book): L_t = annualized trailing-20d mean of the EW "
        "basket (funding proxy, see below); RV_t = annualized 10d realized vol of the EW basket. "
        "LEADING_FLAG = |L_t| >= trailing-252d 90th pctile of |L| AND RV_t <= trailing-252d median RV. "
        "BLOWOUT_FALLBACK = RV_t >= trailing-252d 90th pctile RV. RISK_OFF = LEADING_FLAG OR "
        "BLOWOUT_FALLBACK. On RISK_OFF: FLATTEN and BLOCK new entries for a 10-trading-day window, then "
        "resume. PRIMARY metric = NET-of-cost Sharpe of the gated stream.\n"
        "HONEST DATA DEVIATION. The proposal's thermometer is aggregate perp FUNDING; no funding "
        "adapter is in the tested-import whitelist, so funding is NOT fetched. L_t is a PRICE-ONLY "
        "PROXY for the SAME economic state ('crowded one-sided leverage under a calm surface'): "
        "leveraged longs grind price higher while paying funding, so a strong persistent directional "
        "drift accrued under suppressed realized vol is the price footprint of an over-levered, "
        "deleverage-prone tape. This is a CONDITIONING proxy, never a harvested signal, and remains "
        "opposite in role to the reversal return engine. The funding-data version is the future upgrade "
        "if/when a funding adapter is whitelisted; the verdict reflects the proxy actually tested.\n"
        "LAG / NO LOOK-AHEAD. Signal, RV, L, and gate are built from closes through t; the entire book "
        "is shift(1)-lagged so a position on day t reflects decisions from close t-1. Trailing-252d "
        "percentiles use only data through t.\n"
        "GENERALIZATION (broad). The premium is universal (liquidity-provision reversal protected by a "
        "leading crowding gate), so the identical frozen spec is run on 3 DISJOINT secondary-alt "
        "universes (alts_a/alts_b/defi_l2); a real edge must appear there too (>=60% OOS-positive on "
        "their holdouts) and the gate should cut tails in both. ROBUSTNESS (declared in grid, NOT "
        "selection; must share sign): formation 3/5/7, rebalance 2/3/5, gate pctile 85/90/95, RV-low "
        "median/40th, block 5/10/15.\n"
        "POWER / PLACEBO. The RETURN engine does NOT depend on gate-episode count (thousands of "
        "reversal legs across ~10 names x ~daily rebalance over ~7y) — the deliberate fix for the "
        "<15-episode ceiling; distinct RISK_OFF episodes are the explicit caveat to the human gate. "
        "PRE-REGISTERED PLACEBO (prose only, NOT a soft expectation: it needs many randomized "
        "count/season-matched re-runs = MCPT-class, beyond the one-signal-call soft-expectation "
        "budget): random count-matched RISK_OFF windows must NOT improve the stream. Write-once holdout "
        "= final block (2024-01-01+); MCPT mandatory."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope="broad",
    generalization_universes=["alts_a", "alts_b", "defi_l2"],
    load_gen_data=load_gen_data,
    holdout_start="2024-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "engine_is_reversal",
         "claim": "trailing-5d vs forward-3d cross-sectional rank-corr is negative (reversal, not momentum)",
         "check": _exp_engine_is_reversal},
        {"name": "gate_cuts_left_tail",
         "claim": "gated 5% CVaR >= ungated 5% CVaR on search (deleveraging tail amputated)",
         "check": _exp_gate_cuts_tail},
        {"name": "risk_off_targets_drawdowns",
         "claim": "RISK_OFF block windows overlap the ungated book's worst-decile days above base rate",
         "check": _exp_risk_off_targets_drawdowns},
        {"name": "net_survives_costs",
         "claim": "net (20bps round-trip) search Sharpe > 0 (reversal not killed by turnover)",
         "check": _exp_net_survives_costs},
    ],
)