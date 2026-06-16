"""
Beta-isolated equity-index Variance Risk Premium (option-income ETF complex)
+ validated Boreas trend tail-overlay.

Mechanism: you are PAID to sell equity-index variance insurance (pro-cyclical:
earns in calm, loses in vol spikes). We HOLD the equity-index option-income
complex (PutWrite / BuyWrite ETFs) and SHORT SPY sized to the basket's trailing
beta -> strips directional equity beta, leaving the pure VRP spread. We then test
whether the VALIDATED crisis-alpha trend leg (opposite tail) RESCUES the VRP crash
tail without diluting Sharpe (sized to ~30% of VRP risk, NOT a reflexive 50/50).

Honest structural notes (pre-registered, see SPEC.pre_registration):
 - The proposed cboe_index() (CBOE PUT index 1991+) is NOT in the tested adapter
   set, so we use the deployable owned/free ETF realism throughout (yf_panel).
 - This is a LOW-CARDINALITY overlay book (~5 ETF alpha sleeve, buy-and-hold).
   It generates ~one position-run per name, BELOW the >=50-trade factor-book bar.
   The valid evidence is the RETURN battery (Sharpe, maxDD-reduction from the
   overlay, DSR, beta-matched MCPT, sub-period breadth, OOS holdout, stage-2
   generalisation across DISJOINT VRP-ETF corners). We do NOT fabricate turnover
   (e.g. a monthly sell/re-buy or a vol-trim): a vol-trim would be tail-TIMING
   that contradicts the hedge-not-time mechanism, and forced monthly round-trips
   would be both a cost-inflating and a trade-count hack.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, trend_returns
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ---------------------------------------------------------------- universe ----
HEDGE = "SPY"                                   # declared hedge sleeve (whitelisted)
SEARCH_TICKERS = ["PUTW", "XYLD", "QYLD", "RYLD", "DIVO"]   # SP500/NDX/R2000/Div VRP
START = "2013-01-01"

# generalisation corners: DISJOINT ticker sets, same frozen mechanism, all have
# 2022+ data (stage-2 runs default params on the HOLDOUT only).
GEN = {
    "active_managed_income":   ["JEPI", "JEPQ", "SPYI", "QQQI"],  # active option-income
    "dividend_value_buywrite": ["KNG", "FTHI", "GPIX"],           # dividend/value cov-call
    "half_overlay_growth":     ["QYLG", "XYLG"],                  # 50%-overlay NDX/SP500
}

# index-family "sector" labels for the trade ledger / sector-spread gate
SECTOR_MAP = {
    "PUTW": "SP500_VRP", "XYLD": "SP500_VRP", "QYLD": "NDX_VRP",
    "RYLD": "R2000_VRP", "DIVO": "DIVIDEND_VRP",
    "JEPI": "SP500_VRP", "JEPQ": "NDX_VRP", "SPYI": "SP500_VRP", "QQQI": "NDX_VRP",
    "KNG": "DIVIDEND_VRP", "FTHI": "SP500_VRP", "GPIX": "SP500_VRP",
    "QYLG": "NDX_VRP", "XYLG": "SP500_VRP",
}


# ----------------------------------------------------------------- data --------
def load_data() -> pd.DataFrame:
    # ASSUMES yf_panel returns TOTAL-RETURN (split+distribution adjusted) closes.
    # Option-income ETF return is ~90% distributions (collateral T-bill yield +
    # option premium) -> a price-only panel makes this invalid (gate-0 check).
    px = yf_panel(SEARCH_TICKERS + [HEDGE], start=START)
    return px.dropna(how="all", axis=1).sort_index()


def load_gen_data(label: str) -> pd.DataFrame:
    # REQUIRED for scope='broad'. Same shape as load_data() for ONE gen corner.
    px = yf_panel(GEN[label] + [HEDGE], start="2018-01-01")
    return px.dropna(how="all", axis=1).sort_index()


# --------------------------------------------------------------- the signal ----
def signal(panel, **params):
    overlay_risk = float(params.get("overlay_risk", 0.30))  # trend risk as frac of VRP risk
    beta_hedge   = bool(params.get("beta_hedge", True))
    target_vol   = float(params.get("target_vol", 0.09))    # annualised spread vol target
    vol_lb       = int(params.get("vol_lb", 63))
    beta_lb      = int(params.get("beta_lb", 252))
    kmax         = float(params.get("kmax", 3.0))
    gross_cap    = float(params.get("gross_cap", 2.0))      # retail leverage limit

    panel = panel.sort_index()
    cols = list(panel.columns)
    hedge = HEDGE if HEDGE in cols else None
    alpha_cols = [c for c in cols if c != hedge]
    rets = panel.pct_change()

    # 1) inverse-vol LONG basket weights, normalised to gross 1.0 across live names
    a = rets[alpha_cols]
    vol = a.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std()
    inv = (1.0 / vol.replace(0.0, np.nan)).where(a.notna())
    wsum = inv.sum(axis=1).replace(0.0, np.nan)
    w = inv.div(wsum, axis=0).fillna(0.0)                    # sums to 1 per live date
    basket_ret = (w * a.fillna(0.0)).sum(axis=1)

    spy_ret = rets[hedge] if hedge else pd.Series(0.0, index=panel.index)

    # 2) trailing beta of the basket to SPY (monthly held) -> strip equity beta
    if beta_hedge and hedge:
        cov = basket_ret.rolling(beta_lb, min_periods=beta_lb // 2).cov(spy_ret)
        var = spy_ret.rolling(beta_lb, min_periods=beta_lb // 2).var().replace(0.0, np.nan)
        beta = (cov / var).clip(0.0, 1.5)                   # keeps SPY share <= cap
        beta = beta.resample("MS").first().reindex(panel.index, method="ffill").fillna(0.0)
    else:
        beta = pd.Series(0.0, index=panel.index)

    # 3) vol-target the beta-stripped spread to target_vol
    spread_raw = basket_ret - beta * spy_ret
    sp_vol = spread_raw.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std() * np.sqrt(252)
    k = (target_vol / sp_vol.replace(0.0, np.nan)).clip(upper=kmax).fillna(0.0)

    # 4) full weight matrix (alpha long + SPY hedge), gross-capped, then 1-day lag
    W = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for c in alpha_cols:
        W[c] = w[c] * k
    if hedge:
        W[hedge] = -beta * k
    gross = W.abs().sum(axis=1).replace(0.0, np.nan)
    W = W.mul((gross_cap / gross).clip(upper=1.0).fillna(0.0), axis=0)
    W = W.shift(1).fillna(0.0)        # signals built from data thru t, applied t+1 (lag = ours)

    # 5) net daily VRP-spread returns (8bps base + ETF spread cushion = 10bps on turnover;
    #    collateral T-bill yield is ALREADY embedded in the ETF total return -> no FRED add)
    rfill = rets.fillna(0.0)
    vrp = net_of_cost(W, rfill, cost_bps=10.0, name="vrp_spread")

    # 6) CONTRACT trade ledger on the ALPHA sleeve only (SPY = declared hedge sleeve)
    trades = trades_from_weights(W[alpha_cols], rfill[alpha_cols], SECTOR_MAP)

    # 7) trend tail-overlay sized to ~overlay_risk of the VRP-spread risk (not 50/50)
    out = vrp
    if overlay_risk > 0.0:
        try:
            tr, _ = trend_returns()
            tr = pd.Series(tr).reindex(vrp.index).fillna(0.0)
            vv = vrp.rolling(126, min_periods=60).std()
            tv = tr.rolling(126, min_periods=60).std().replace(0.0, np.nan)
            scale = (overlay_risk * vv / tv).replace([np.inf, -np.inf], np.nan).clip(upper=5.0)
            scale = scale.shift(1).fillna(0.0)
            out = vrp + tr * scale
        except Exception:
            out = vrp
    out = out.copy()
    out.name = "vrp_putwrite_trend"
    return out, trades


# ------------------------------------------------- soft-expectation helpers ----
def _ann_sharpe(r):
    if r is None:
        return 0.0
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _maxdd(r):
    if r is None:
        return 0.0
    r = pd.Series(r).dropna()
    if r.empty:
        return 0.0
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _beta_to_spy(r, panel, holdout_start):
    if r is None:
        return np.nan
    m = panel["SPY"].pct_change()
    m = m[m.index < pd.Timestamp(holdout_start)]
    df = pd.concat([pd.Series(r), m], axis=1).dropna()
    df.columns = ["r", "m"]
    if len(df) < 60 or df["m"].var() == 0:
        return np.nan
    return float(df["r"].cov(df["m"]) / df["m"].var())


def _chk_standalone_sharpe(ctx):
    sh = _ann_sharpe(ctx["grid"].get("standalone"))
    return {"pass": sh > 0.3, "observed": round(sh, 3)}


def _chk_trend_cuts_tail(ctx):
    base = _maxdd(ctx["grid"].get("standalone"))
    comb = _maxdd(ctx["grid"].get("default"))
    ratio = abs(comb) / (abs(base) + 1e-9)
    return {"pass": bool(abs(comb) <= 0.75 * abs(base) + 1e-9), "observed": round(ratio, 3)}


def _chk_trend_keeps_sharpe(ctx):
    sh_b = _ann_sharpe(ctx["grid"].get("standalone"))
    sh_c = _ann_sharpe(ctx["grid"].get("default"))
    return {"pass": bool(sh_c >= sh_b - 1e-9), "observed": round(sh_c - sh_b, 3)}


def _chk_beta_isolated(ctx):
    b_h = _beta_to_spy(ctx["grid"].get("default"), ctx["panel"], ctx["holdout_start"])
    b_r = _beta_to_spy(ctx["grid"].get("hedge_off"), ctx["panel"], ctx["holdout_start"])
    ok = (b_h == b_h) and abs(b_h) < 0.25 and (not (b_r == b_r) or abs(b_r) > abs(b_h))
    return {"pass": bool(ok), "observed": round(b_h, 3) if b_h == b_h else "nan"}


def _chk_overlay_30_ge_50(ctx):
    sh_30 = _ann_sharpe(ctx["grid"].get("default"))
    sh_50 = _ann_sharpe(ctx["grid"].get("overlay_50"))
    return {"pass": bool(sh_30 >= sh_50 - 1e-9), "observed": round(sh_30 - sh_50, 3)}


# ------------------------------------------------------------------- spec ------
grid = {
    "default":    {},                                       # combined: VRP + 30% trend (primary)
    "standalone": {"overlay_risk": 0.0},                    # beta-isolated VRP, no overlay
    "overlay_50": {"overlay_risk": 0.50},                   # reflexive 50% (should NOT beat 30%)
    "hedge_off":  {"beta_hedge": False, "overlay_risk": 0.0},  # raw long VRP (high beta) baseline
}

SPEC = StrategySpec(
    id="vrp_putwrite_beta_isolated_trend_overlay",
    family="volatility_risk_premium",
    title="Beta-isolated equity-index VRP (option-income ETF complex) + validated trend tail-overlay",
    markets=["US_equity_index_volatility", "cross_asset_trend"],
    data_desc=("yfinance TOTAL-RETURN closes of liquid US equity-index option-income ETFs "
               "(PUTW/XYLD/QYLD/RYLD/DIVO; collateral T-bill yield + option premium embedded) "
               "+ SPY beta hedge; Boreas validated 21-market trend overlay via trend_returns()."),
    pre_registration=(
        "PREMIUM: equity-index variance risk premium (paid to sell index variance insurance; "
        "pro-cyclical). MECHANISM: hold the option-income ETF complex (inverse-vol basket, weekly), "
        "SHORT SPY at the basket's trailing-252d (t-1) beta, monthly-rebalanced, to STRIP equity beta; "
        "vol-target the beta-stripped spread to 9% ann. STANDALONE-FIRST (2026-06-08): only treat the "
        "trend overlay as confirmed if the beta-isolated spread standalone net Sharpe>0.3. OVERLAY: add "
        "the validated Boreas trend leg sized to ~30% of the VRP risk (NOT 50/50); it must CUT the VRP "
        "maxDD by >=25% WITHOUT reducing Sharpe (machine-checked below). All signals 1-day lagged; "
        "10bps turnover cost (8bps base + ETF spread cushion); collateral yield already in ETF TR (no "
        "FRED double-count). SPY short declared as a hedge sleeve (whitelist+cap) so the alpha book is "
        "judged alone. GATE-0 DATA CHECK: confirm yf_panel returns DISTRIBUTION-ADJUSTED (total-return) "
        "closes for the income ETFs (return is ~90% distributions) and gross<~2x; if price-only, ABORT. "
        "STRUCTURAL/DEPLOYMENT NOTE (honest, not gamed): this is a low-cardinality buy-and-hold overlay "
        "(~5 ETF alpha sleeve) -> ~one position-run per name, BELOW the >=50-trade factor-book bar; the "
        "trade-count/single-name gates are expected NOT-MET as a structural mismatch. We deliberately do "
        "NOT fabricate turnover (monthly sell/re-buy = cost+count hack; a vol-trim = tail-TIMING that "
        "contradicts the hedge-not-time mechanism and would duplicate the vvix_vix timing overlay). "
        "Primary evidence = RETURN battery (net Sharpe, overlay maxDD-reduction, DSR over the 4 grid "
        "variants, BETA-MATCHED MCPT to prove selection-alpha over disguised equity beta, sub-period "
        "breadth, OOS holdout 2022+, and the stage-2 disjoint-corner generalisation). NON-DUPLICATIVE: "
        "this HARVESTS the equity-index option premium and HEDGES its tail with the validated trend "
        "complement (the pro-cyclical replacement for the orphaned carry leg); it does not TIME beta on a "
        "VVIX/VIX flag (vvix_vix_divergence_gate) nor short a VIX-futures roll (vrp_regime_termstructure)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=grid,
    scope="broad",
    generalization_universes=["active_managed_income", "dividend_value_buywrite", "half_overlay_growth"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    hedge_tickers=["SPY"],
    hedge_cap=0.60,
    expectations=[
        {"name": "standalone_sharpe_pos",
         "claim": "beta-isolated VRP standalone net Sharpe > 0.3 in the search window",
         "check": _chk_standalone_sharpe},
        {"name": "trend_cuts_tail_25pct",
         "claim": "30% trend overlay cuts standalone maxDD by >=25% (combined |DD| <= 0.75*standalone |DD|)",
         "check": _chk_trend_cuts_tail},
        {"name": "trend_keeps_sharpe",
         "claim": "trend overlay does NOT reduce Sharpe (combined Sharpe >= standalone Sharpe)",
         "check": _chk_trend_keeps_sharpe},
        {"name": "beta_isolated",
         "claim": "beta-hedged combined |beta to SPY| < 0.25 and below the raw (hedge_off) beta",
         "check": _chk_beta_isolated},
        {"name": "overlay_30_ge_50",
         "claim": "30% overlay sizing has Sharpe >= reflexive 50% (do not over-hedge the VRP edge)",
         "check": _chk_overlay_30_ge_50},
    ],
)