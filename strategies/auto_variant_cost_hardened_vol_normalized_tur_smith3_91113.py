"""Low-volatility anomaly — cross-sectional long/short equity factor book.

MECHANISM (pre-registered, broad): the low-volatility anomaly is a universal
risk-premium distortion — low past-volatility stocks earn higher risk-adjusted
returns than high-volatility stocks (leverage-constraint / lottery-preference
theory). If real it must appear across cap tiers, not just one universe, so this
is scope='broad' and is held to the stage-2 generalization battery.

Construction: trailing realized vol -> cross-sectional z (low vol = high score)
-> dollar-neutral long/short, inverse-vol sized, capped, weekly rebalance,
signals lagged 1 day, ~8bps cost on turnover. The ONLY novel code is _weights();
everything else is harness kit.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

_START = "2010-01-01"

# universe label -> Sharadar marketcap tier. 'mid' is the SEARCH universe;
# the gen tiers are DISJOINT by cap (no shared tickers).
_TIERS = {"mid": "Mid", "small": "Small", "large": "Large", "micro": "Micro"}

_UNIVERSE_CACHE = {}      # label -> (tickers, sector_map)
_SECTOR_MAP = {}          # merged ticker -> sector across all built universes


def _universe(label):
    if label not in _UNIVERSE_CACHE:
        tickers, smap = sector_universe(_TIERS[label], top_n_per_sector=20)
        _UNIVERSE_CACHE[label] = (tickers, smap)
        _SECTOR_MAP.update(smap)
    return _UNIVERSE_CACHE[label]


def _panel(label):
    tickers, _ = _universe(label)
    panel = sep_panel(tickers, start=_START)
    panel.attrs["label"] = label
    return panel


def load_data() -> pd.DataFrame:
    return _panel("mid")


def load_gen_data(label) -> pd.DataFrame:
    return _panel(label)


def _weights(panel, vol_lb=126, max_weight=0.05):
    """Build same-day dollar-neutral inverse-vol low-vol weights (NOT yet lagged)."""
    rets = panel.pct_change()
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()
    # low vol -> high score
    score = xs_zscore(-vol)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    raw = score * inv_vol
    # dollar-neutral (cross-sectional demean)
    raw = raw.sub(raw.mean(axis=1), axis=0)
    # cap single-name weight, then renormalize gross to 1
    gross = raw.abs().sum(axis=1)
    W = raw.div(gross.replace(0, np.nan), axis=0)
    W = W.clip(-max_weight, max_weight)
    W = W.div(W.abs().sum(axis=1).replace(0, np.nan), axis=0)
    # weekly rebalance: only update on Mondays, hold between
    mask = W.index.weekday == 0
    W = W.where(pd.Series(mask, index=W.index), axis=0)
    W = W.ffill()
    return W, rets


def signal(panel, **params):
    vol_lb = int(params.get("vol_lb", 126))
    max_weight = float(params.get("max_weight", 0.05))
    W, rets = _weights(panel, vol_lb=vol_lb, max_weight=max_weight)
    Wl = W.shift(1)  # lag signals 1 day -> no look-ahead
    daily = net_of_cost(Wl, rets, cost_bps=8.0, name="lowvol")
    smap = {t: _SECTOR_MAP.get(t, "Unknown") for t in panel.columns}
    trades = trades_from_weights(Wl, rets, smap)
    return daily, trades


# ---- soft expectation: edge robust to the vol lookback (uses pre-declared grid) ----
def _check_robust_lookback(ctx):
    g = ctx.get("grid", {})
    obs, ok = {}, True
    for k in ("vol_63", "vol_252"):
        s = g.get(k)
        if s is not None and len(s) > 0:
            m = float(s.mean())
            obs[k] = m
            ok = ok and (m > 0)
        else:
            ok = False
    return {"pass": bool(ok), "observed": str(obs)}


SPEC = StrategySpec(
    id="lowvol_xs_ls",
    family="low_volatility",
    title="Cross-sectional low-volatility long/short equity factor",
    markets=["US equities"],
    data_desc="Sharadar SEP survivorship-clean daily prices; sector-spread cap-tier universes.",
    pre_registration=(
        "Low-volatility anomaly: rank stocks by trailing realized vol, go long "
        "low-vol / short high-vol, dollar-neutral inverse-vol sized, weekly "
        "rebalance, 8bps cost, 1-day lag. Theory (leverage constraints / lottery "
        "preference) predicts a UNIVERSAL effect, so it must generalize to other "
        "cap tiers (small/large/micro holdouts). Expect positive net Sharpe "
        "robust across vol lookbacks; if it only works in the mid-cap search "
        "universe it is an overfit outlier and should be rejected."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "vol_63": {"vol_lb": 63},
        "vol_252": {"vol_lb": 252},
    },
    scope="broad",
    generalization_universes=["small", "large", "micro"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=30,
    expectations=[
        {
            "name": "robust_across_lookbacks",
            "claim": "net returns positive for both 63d and 252d vol lookbacks",
            "check": _check_robust_lookback,
        },
    ],
)