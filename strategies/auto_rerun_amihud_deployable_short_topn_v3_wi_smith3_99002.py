"""
RERUN — Amihud deployable-short topN v3 with DECLARED IWM hedge sleeve.

EXACT rerun of the 2026-06-11 near-miss (experiments/amihud_illiq_topN_short_v3.md:
search 1.66 -> holdout 1.44, DSR~1.0, 15/15 CPCV positive; failed ONLY on
single_name_share 0.47 = the continuously-held IWM residual-beta trim sleeve).
The ONLY change vs the parent is declaring the already-present hedge on the spec:
    hedge_tickers=["IWM"], hedge_cap=0.35
so the deployment gate judges the ~230-name alpha book alone and gates the sleeve
on whitelist+cap. Signal, universe, costs, params, grid: UNCHANGED. Fresh id
(hedge declaration changes the frozen-design hash -> legitimately new config,
not a holdout double-dip).

Construction (deployable at small AUM, unlike the 100-name short leg the
beta-confound gate killed in the ETF-only variant):
  LONG  top-N most-ILLIQUID eligible small/mid names (Amihud 2002 premium),
  SHORT top-n most-LIQUID names (borrowable proxies, small gross),
  plus a residual IWM short sized to trailing portfolio beta, capped at 0.35.
Hard pre-registered gates unchanged: |beta_to_universe| < 0.3 AND
selection_alpha_sharpe > 0; hedge_share must come in under the 0.35 cap or the
run honestly fails on the sleeve being oversized.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2014-01-01"

# ---------------------------------------------------------------- universe ---
_UNI = None


def _universe():
    """Sector-spread survivorship-clean small+mid universe (~1000 names)."""
    global _UNI
    if _UNI is None:
        t_sm, s_sm = sector_universe(marketcap="Small", top_n_per_sector=45)
        t_md, s_md = sector_universe(marketcap="Mid", top_n_per_sector=45)
        tickers = sorted(set(t_sm) | set(t_md))
        sector_map = {**s_sm, **s_md}
        sector_map["IWM"] = "ETF Hedge"  # declared hedge sleeve
        _UNI = (tickers, sector_map)
    return _UNI


# -------------------------------------------------------------------- data ---
def load_data() -> pd.DataFrame:
    """Multi-field panel: (field, ticker) columns. Sharadar SEP (delisted incl.)
    for the stock book; IWM Close from yfinance for the hedge sleeve only."""
    tickers, _ = _universe()
    closeadj = sep_panel(tickers, START, field="closeadj")
    closeraw = sep_panel(tickers, START, field="closeunadj")
    volume = sep_panel(tickers, START, field="volume")
    panel = pd.concat(
        {"closeadj": closeadj, "closeunadj": closeraw, "volume": volume}, axis=1
    )
    iwm = yf_panel(["IWM"], START)
    panel[("hedge", "IWM")] = iwm["IWM"].reindex(panel.index)
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' (defensibly universe-specific deployable construction);
    # forward-validation confirms it. No generalization universes declared.
    raise ValueError("scope='local': no generalization universes for %r" % label)


# ------------------------------------------------------------------ signal ---
def signal(panel, illiq_lb=63, n_long=200, n_short=30, short_gross=0.30,
           vol_lb=63, beta_lb=252, hedge_cap=0.35,
           min_adv=2.0e5, min_price=2.0):
    close = panel["closeadj"]
    raw = panel["closeunadj"]
    vol = panel["volume"]
    iwm_px = panel["hedge"]["IWM"]

    rets = close.pct_change()
    iwm_ret = iwm_px.pct_change().rename("IWM")

    # --- Amihud illiquidity: trailing mean |ret| / $volume (scale-free ranks) ---
    dvol = (raw * vol).replace(0.0, np.nan)
    mp = int(illiq_lb * 0.8)
    illiq = (rets.abs() / dvol).rolling(illiq_lb, min_periods=mp).mean() * 1e9
    adv = dvol.rolling(illiq_lb, min_periods=mp).median()

    # Eligibility: tradable but allowed to be illiquid (no penny/zero-volume junk)
    eligible = (adv >= min_adv) & (raw >= min_price) & illiq.notna()

    # Inverse-vol sizing inputs + trailing per-name beta to IWM (all trailing-only)
    sigma = rets.rolling(vol_lb, min_periods=int(vol_lb * 0.8)).std()
    bmp = int(beta_lb * 0.6)
    beta = (rets.rolling(beta_lb, min_periods=bmp).cov(iwm_ret)
            / iwm_ret.rolling(beta_lb, min_periods=bmp).var())

    # --- weekly rebalance: last trading day of each week ---
    week_last = pd.Series(rets.index, index=rets.index).resample("W-FRI").last().dropna()
    rebal_dates = pd.DatetimeIndex(week_last.values)

    cols = list(close.columns) + ["IWM"]
    rows = {}
    for dt in rebal_dates:
        if dt not in illiq.index:
            continue
        il = illiq.loc[dt].where(eligible.loc[dt]).dropna()
        if len(il) < n_long + n_short + 50:
            continue

        longs = il.nlargest(n_long).index    # most illiquid -> long premium
        shorts = il.nsmallest(n_short).index  # most liquid -> borrowable shorts

        iv = (1.0 / sigma.loc[dt]).replace([np.inf, -np.inf], np.nan)
        wl = iv.reindex(longs).dropna()
        ws = iv.reindex(shorts).dropna()
        if len(wl) < int(n_long * 0.5) or len(ws) < int(n_short * 0.5):
            continue
        wl = wl / wl.sum()                       # long gross 1.0, inverse-vol
        ws = -short_gross * ws / ws.sum()        # short stock gross (small)

        w = pd.concat([wl, ws])
        # Residual book beta -> IWM short, capped (declared sleeve, never long)
        b = float((w * beta.loc[dt].reindex(w.index)).sum())
        w_iwm = -float(np.clip(b, 0.0, hedge_cap))

        row = pd.Series(0.0, index=cols)
        row.loc[w.index] = w.values
        row.loc["IWM"] = w_iwm
        rows[dt] = row

    if not rows:
        empty = pd.Series(dtype=float, name="amihud_topN_short_v3_hedged")
        return empty, []

    # Weekly weights -> daily holdings (ffill), then THE LAG: weights are built
    # on same-day information, so shift(1) before computing returns. NO look-ahead.
    Wr = (pd.DataFrame.from_dict(rows, orient="index")
          .reindex(rets.index).ffill().fillna(0.0))
    W_lag = Wr.shift(1).fillna(0.0)

    all_rets = pd.concat([rets, iwm_ret], axis=1)

    daily = net_of_cost(W_lag, all_rets, cost_bps=8.0,
                        name="amihud_topN_short_v3_hedged")

    _, sector_map = _universe()
    trades = trades_from_weights(W_lag, all_rets, sector_map)

    return daily, trades


# -------------------------------------------------------------------- spec ---
GRID = {
    "default": {},
    "longer_illiq": {"illiq_lb": 126},
    "tighter_book": {"n_long": 150, "n_short": 25},
    "heavier_short": {"short_gross": 0.40},
}

SPEC = StrategySpec(
    id="amihud_illiq_topN_short_v3_hedge_declared",
    family="amihud_illiquidity",
    title=("RERUN — Amihud deployable-short topN v3 with DECLARED IWM hedge "
           "sleeve (hedge_tickers=['IWM'], hedge_cap=0.35)"),
    markets=["US_smallmid_equity"],
    data_desc=("Sharadar SEP small+mid sector-spread universe (~1000 names, "
               "delisted incl.): closeadj/closeunadj/volume via sep_panel(); "
               "IWM Close via yf_panel for the residual-beta hedge sleeve only."),
    pre_registration=(
        "EXACT rerun of experiments/amihud_illiq_topN_short_v3.md (search 1.66 "
        "-> holdout 1.44, DSR~1.0, 15/15 CPCV positive; killed only by "
        "single_name_share on the undeclared IWM trim sleeve). ONLY change: the "
        "IWM residual-beta sleeve is now DECLARED on the spec (hedge_tickers="
        "['IWM'], hedge_cap=0.35) so the deployment gate judges the ~230-name "
        "alpha book alone. Signal/universe/costs/params/grid UNCHANGED; fresh id "
        "because the hedge declaration changes the frozen-design hash. HARD "
        "pre-registered gates unchanged: |beta_to_universe| < 0.3 AND "
        "selection_alpha_sharpe > 0; hedge_share must come in under the 0.35 cap "
        "or the run honestly fails on the sleeve being oversized (informative "
        "either way). If the alpha book itself has a concentration problem, the "
        "gate SHOULD still fail it. Costs 8bps on turnover; weekly rebalance; "
        "inverse-vol sizing; weights shift(1)-lagged."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=30,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)