"""
Crypto crowding-reversal (liquidity-provision premium), broad liquid USD pairs.

ECONOMIC THESIS (unchanged family): aggressive one-sided crowding in a perp tends to
MEAN-REVERT — over-bid / over-extended names give back, over-sold names bounce. This is
the liquidity-provision premium the original proposal targeted via taker-flow imbalance.

HONEST DATA PIVOT (was GATE-0 blocked): the sanctioned adapter set has NO order-flow
source (binance_klines/TAKER_BUY_QUOTE) and NO implied-vol source (deribit_dvol). Rather
than fabricate adapters or refuse, we implement the SAME mechanism with the owned data:
  - Crowding is proxied by realised short-horizon price EXTENSION (smoothed trailing
    return, xs z-scored) — over-extended == crowded-long. This is a noisier but
    economically-aligned proxy for one-sided taker pressure.
  - The DVOL stress gate is proxied by BTC REALISED-vol stress (point-in-time expanding
    quantile) — gross is halved in high-vol regimes.
Both proxies are stated here so the verdict reflects what is actually computed.

scope=local: the investable liquid crypto-USD universe is ~50 names — too small to form
three DISJOINT 150-400 name generalization universes, so a broad cross-universe battery is
infeasible. The edge is validated by forward/holdout instead. (If owned crypto order-flow
adapters are added later, re-spec with the true taker-flow signal and re-evaluate as broad.)
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np
import pandas as pd

_START = "2018-01-01"

# Liquid crypto USD pairs (yfinance format), grouped into clusters for the trade ledger.
_CLUSTERS = {
    "l1": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD", "AVAX-USD",
           "ATOM-USD", "TRX-USD", "ALGO-USD", "NEAR-USD", "EGLD-USD", "ICP-USD", "EOS-USD",
           "XTZ-USD", "NEO-USD", "QTUM-USD", "FTM-USD", "KSM-USD", "VET-USD", "THETA-USD"],
    "payments": ["XRP-USD", "XLM-USD", "LTC-USD", "BCH-USD", "DASH-USD", "IOTA-USD", "ZIL-USD"],
    "privacy": ["XMR-USD", "ZEC-USD"],
    "defi": ["AAVE-USD", "MKR-USD", "COMP-USD", "SNX-USD", "CRV-USD", "YFI-USD", "UNI-USD",
             "GRT-USD", "LINK-USD"],
    "meme": ["DOGE-USD"],
    "gaming_nft": ["SAND-USD", "MANA-USD", "AXS-USD", "CHZ-USD", "ENJ-USD"],
    "infra": ["FIL-USD", "BAT-USD", "ETC-USD"],
}
_UNIVERSE = [t for v in _CLUSTERS.values() for t in v]
_SECTOR_MAP = {t: c for c, ts in _CLUSTERS.items() for t in ts}


def load_data() -> pd.DataFrame:
    """Daily Close panel for the liquid crypto-USD universe (yfinance; free, crypto-OK)."""
    return yf_panel(_UNIVERSE, start=_START)


def load_gen_data(label) -> pd.DataFrame:
    """Not exercised under scope='local'; provided for contract completeness."""
    return load_data()


def signal(panel, **params):
    lookback = int(params.get("lookback", 7))    # crowding horizon (days)
    smooth = int(params.get("smooth", 3))        # smoothing of the crowding proxy
    gross = float(params.get("gross", 1.5))      # target gross (<=2x)
    vol_lb = int(params.get("vol_lb", 30))       # vol lookback for inverse-vol sizing
    sign = float(params.get("sign", -1.0))       # -1 = reversal (default), +1 = momentum
    floor = 1e-4

    px = panel.astype(float)
    rets = px.pct_change()

    # Crowding proxy: smoothed trailing return (over-extended == crowded long), xs z-scored.
    crowd = px.pct_change(lookback).rolling(smooth).mean()
    raw = sign * xs_zscore(crowd)                # reversal: short crowded, long oversold

    # Inverse-vol sizing.
    vol = rets.rolling(vol_lb).std()
    sized = raw / (vol + floor)

    # Dollar-neutral, then gross-normalize per date.
    sized = sized.sub(sized.mean(axis=1), axis=0)
    g = sized.abs().sum(axis=1)
    W = sized.div(g.replace(0, np.nan), axis=0) * gross

    # Implied-vol-stress gate (DVOL proxy via BTC realised-vol; PIT expanding quantile, lagged).
    btc_ret = px["BTC-USD"].pct_change()
    rv = btc_ret.rolling(vol_lb).std()
    thr = rv.expanding(min_periods=60).quantile(0.80)
    scale = pd.Series(np.where(rv > thr, 0.5, 1.0), index=px.index).shift(1).fillna(1.0)
    W = W.mul(scale, axis=0)

    # Weekly rebalance: hold Monday weights through the week.
    off = W.index.dayofweek != 0
    Wk = W.copy()
    Wk.loc[off] = np.nan
    Wk = Wk.ffill()

    # 1-day lag is OUR responsibility: act on yesterday's signal (no look-ahead).
    Wlag = Wk.shift(1).fillna(0.0)

    daily = net_of_cost(Wlag, rets, cost_bps=10.0, name="crypto_crowding_reversal")
    trades = trades_from_weights(Wlag, rets, _SECTOR_MAP)
    return daily, trades


def _sharpe(r):
    if r is None:
        return float("nan")
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(365.0))


def _exp_reversal_beats_momentum(ctx):
    """Mechanism falsification: if this is a crowding-REVERSAL edge, the reversal sign
    must beat the momentum sign on the search window (free: uses pre-declared grid)."""
    g = ctx.get("grid", {})
    rev = _sharpe(g.get("default"))
    mom = _sharpe(g.get("momentum_sign"))
    return {"pass": bool(np.isfinite(rev) and rev > mom),
            "observed": f"reversal Sharpe {rev:.2f} vs momentum {mom:.2f}"}


SPEC = StrategySpec(
    id="crypto_crowding_reversal",
    family="crowding_reversal_flow",
    title="Crypto crowding reversal, liquid USD pairs (dollar-neutral, vol-stress gated)",
    markets=["crypto"],
    data_desc=("yfinance daily Close for ~48 liquid crypto-USD pairs. Crowding is proxied "
               "by smoothed short-horizon realised return extension (owned data has NO "
               "taker-flow/order-flow field); implied-vol stress is proxied by BTC realised "
               "vol. These proxies replace the unavailable binance_klines TAKER_BUY_QUOTE "
               "and deribit_dvol the original proposal required."),
    pre_registration=(
        "Cross-sectional reversal of one-sided crowding: long the over-sold / crowded-short "
        "names, short the over-extended / crowded-long names, ranked by smoothed trailing "
        "return (3-7d, xs z-scored) as a price-extension proxy for taker-flow imbalance. "
        "Inverse-vol sized, dollar-neutral, weekly rebalance, 1-day signal lag, ~10bps/turn "
        "cost, gross <=1.5x. Halve gross in BTC realised-vol stress regimes (point-in-time, "
        "DVOL proxy). DATA-PIVOT NOTE: the sanctioned adapter set has no crypto order-flow "
        "or implied-vol source, so the original taker-flow/DVOL signals are proxied by "
        "realised price/vol; the family (liquidity-provision crowding reversal) is unchanged. "
        "scope=local because the liquid crypto universe (~50 names) cannot supply three "
        "disjoint 150-400 name generalization universes; forward/holdout validation instead."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "momentum_sign": {"sign": 1.0},
        "lb14": {"lookback": 14},
        "gross1": {"gross": 1.0},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "reversal_beats_momentum",
         "claim": "reversal-sign book out-Sharpes the momentum-sign book on the search window",
         "check": _exp_reversal_beats_momentum},
    ],
)