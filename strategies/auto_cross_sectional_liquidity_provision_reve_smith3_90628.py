"""Short-term cross-sectional reversal (US equities), inverse-vol, dollar-neutral.

NOTE ON PROVENANCE: the originally-requested module (a Binance perp taker-buy-quote
imbalance / flow-crowding gate) is DATA-GATED — no adapter in the tested import surface
exposes taker_buy_quote/quote_volume, and fabricating that import would be a phantom data
path. Rather than ship "naive crypto reversal" mislabelled as a flow edge, this module
implements a DIFFERENT, fully-runnable, well-documented premium on owned data:
1-week cross-sectional reversal (Lehmann 1990 / Lo-MacKinlay), which lives in less-liquid
names and is a genuine universal mechanism -> scope='broad'.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2010-01-01"

# disjoint generalization universes: search book is MID cap (all sectors); each gen book is
# SMALL cap sliced into a non-overlapping sector group -> disjoint by cap tier AND by sector.
GEN_SECTORS = {
    "small_cyclicals":  ["Industrials", "Consumer Cyclical", "Basic Materials", "Energy"],
    "small_defensives": ["Consumer Defensive", "Healthcare", "Utilities"],
    "small_growthfin":  ["Technology", "Communication Services", "Financial Services", "Real Estate"],
}


def _panel(tickers, sector_map):
    px = sep_panel(tickers, start=START, field="closeadj")
    keep = {t: sector_map[t] for t in px.columns if t in sector_map}
    px.attrs["sector_map"] = keep
    return px


def load_data() -> pd.DataFrame:
    tickers, sector_map = sector_universe(marketcap="Mid", top_n_per_sector=120)
    return _panel(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=80)
    secs = set(GEN_SECTORS[label])
    keep = [t for t in tickers if sector_map.get(t) in secs]
    return _panel(keep, sector_map)


def _weights(panel, lookback=5, frac=0.20, vol_lb=20):
    """Same-day weights; caller lags. Long recent losers / short recent winners."""
    px = panel
    rets = px.pct_change(fill_method=None)
    # reversal signal: negate trailing return z-score (high z == biggest loser == long)
    z = -xs_zscore(px.pct_change(lookback, fill_method=None))
    rank = z.rank(axis=1, pct=True)
    long_mask, short_mask = rank >= (1 - frac), rank <= frac
    iv = 1.0 / rets.rolling(vol_lb).std().replace(0, np.nan)  # inverse-vol size
    wl = iv.where(long_mask)
    ws = iv.where(short_mask)
    wl = wl.div(wl.sum(axis=1), axis=0) * 0.5   # 0.5 gross long
    ws = ws.div(ws.sum(axis=1), axis=0) * 0.5   # 0.5 gross short  -> dollar-neutral, gross ~1x
    W = wl.fillna(0.0).sub(ws.fillna(0.0), fill_value=0.0)
    # weekly rebalance: refresh weights every 5 trading days, hold between
    rebal = pd.Series(False, index=W.index)
    rebal.iloc[::5] = True
    return W.where(rebal).ffill().fillna(0.0), rets


def signal(panel, lookback=5, frac=0.20, vol_lb=20, **params):
    sector_map = getattr(panel, "attrs", {}).get("sector_map", {})
    W, rets = _weights(panel, lookback=lookback, frac=frac, vol_lb=vol_lb)
    Wl = W.shift(1)  # LAG 1 day: weights are built from same-day data, lag is applied here -> no look-ahead
    daily = net_of_cost(Wl, rets, cost_bps=8.0, name="st_reversal_midcap")
    trades = trades_from_weights(Wl, rets, sector_map)
    return daily, trades


# ---- soft expectation: the mechanism is reversal, so next-week return must fall in trailing return ----
def _check_reversal_direction(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    px = ctx["panel"]
    px = px[px.index < hs]                       # search-window only
    past = px.pct_change(5, fill_method=None)
    fwd = px.pct_change(5, fill_method=None).shift(-5)  # diagnostic only (not traded)
    corrs = []
    for d in px.index[::5]:
        a, b = past.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() > 20:
            corrs.append(a[m].corr(b[m], method="spearman"))
    obs = float(np.nanmean(corrs)) if corrs else float("nan")
    return {"pass": obs < 0, "observed": obs}


SPEC = StrategySpec(
    id="st_reversal_midcap_xs",
    family="reversal",
    title="Short-term cross-sectional reversal (mid-cap US, inverse-vol, dollar-neutral)",
    markets=["US equities"],
    data_desc="Sharadar SEP closeadj (survivorship-clean, delisted incl.); mid-cap sector-spread universe.",
    pre_registration=(
        "SUBSTITUTION DISCLOSURE: requested crypto taker-flow-imbalance edge is data-gated "
        "(no taker_buy_quote/quote_volume adapter in tested surface) and is parked. This module "
        "instead tests the 1-week cross-sectional reversal premium (Lehmann 1990): cross-section "
        "of recent 5-day losers outperforms recent winners net of ~8bps turnover cost. Long bottom "
        "20% / short top 20% by negated 5-day return z-score, inverse-vol sized, dollar-neutral, "
        "gross ~1x, weekly rebalance, signals lagged 1 day. As a universal microstructure mechanism "
        "it must generalise to untouched small-cap sub-universes (stage-2). MECHANISM CHECK: "
        "per-date Spearman corr of trailing-5d vs forward-5d return is negative in-sample."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb3": {"lookback": 3},
        "lb7": {"lookback": 7},
        "lb10": {"lookback": 10},
    },
    scope="broad",
    generalization_universes=list(GEN_SECTORS),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
    expectations=[
        {
            "name": "reversal_direction",
            "claim": "in-sample per-date Spearman(trailing-5d, forward-5d) < 0 (reversal, not momentum)",
            "check": _check_reversal_direction,
        },
    ],
)