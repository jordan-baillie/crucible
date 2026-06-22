"""Net share-issuance (supply dilution) premium — US equities.

Pivot note: the originating idea was a crypto "supply inflation / dilution
premium" (token supply growth predicts negative forward return).  The Crucible
SDK exposes NO tested crypto supply/klines adapter in sdk.adapters — only
survivorship-clean equity + fundamentals adapters — so this module tests the
SAME economic mechanism in its well-documented equity analog: the net share
issuance anomaly (Pontiff & Woodgate 2008; Daniel & Titman 2006).  Firms that
INFLATE share supply (issue equity) earn LOW forward returns; firms that
CONTRACT supply (buy back stock) earn HIGH forward returns.  Long low-issuance /
short high-issuance, dollar-neutral, inverse-vol sized, weekly rebalanced.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

START = "2002-01-01"
SHARES_FIELD = "sharesbas"   # Sharadar SF1 basic shares outstanding (PIT via datekey)
LOOKBACK = 252               # ~1y supply-growth window
VOL_LB = 63                  # ~3m vol for inverse-vol sizing
TOP_FRAC = 0.20              # top/bottom quintile long/short


# ---- universe helpers -------------------------------------------------------
def _alt_group(sector_map):
    """Deterministic disjoint sector partition (robust to taxonomy spelling)."""
    secs = sorted({v for v in sector_map.values() if v})
    return set(secs[: len(secs) // 2])


def _panel_for(tickers, sector_map, start=START):
    px = sep_panel(tickers, start=start)             # survivorship-clean adj close
    px = px.dropna(how="all", axis=1)
    cols = list(px.columns)
    sf = sf1(cols, [SHARES_FIELD])                   # datekey-based fundamentals
    shares = pit_panel(sf, SHARES_FIELD, px.index, cols)  # PIT, ffilled, no lookahead
    panel = px.copy()
    panel.attrs["shares"] = shares
    panel.attrs["sector_map"] = {t: sector_map.get(t, "Unknown") for t in cols}
    return panel


# ---- signal -----------------------------------------------------------------
def _issuance_z(shares, lookback):
    growth = shares / shares.shift(lookback) - 1.0   # trailing supply inflation
    return xs_zscore(-growth)                         # long low issuance, short high


def _weights(z, rets, vol_lb, top_frac, gross=1.0):
    vol = rets.rolling(vol_lb).std()
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    rank = z.rank(axis=1, pct=True)
    wl = iv.where(rank.ge(1 - top_frac)).fillna(0.0)
    ws = iv.where(rank.le(top_frac)).fillna(0.0)
    wl = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0) * (gross / 2)
    ws = ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0) * (gross / 2)
    return (wl - ws).fillna(0.0)


def _weekly(W):
    fri = W.index.weekday == 4
    return W.loc[fri].reindex(W.index).ffill().fillna(0.0)


def signal(panel, **params):
    lookback = int(params.get("lookback", LOOKBACK))
    vol_lb = int(params.get("vol_lb", VOL_LB))
    top_frac = float(params.get("top_frac", TOP_FRAC))
    cost_bps = float(params.get("cost_bps", 8.0))

    rets = panel.pct_change().replace([np.inf, -np.inf], np.nan)
    z = _issuance_z(panel.attrs["shares"], lookback)
    W = _weekly(_weights(z, rets, vol_lb, top_frac))
    Wlag = W.shift(1)   # same-day target weights -> trade next day (lag is OURS; no lookahead)

    daily = net_of_cost(Wlag, rets, cost_bps=cost_bps, name="dilution_premium")
    trades = trades_from_weights(Wlag, rets, panel.attrs["sector_map"])
    return daily, trades


# ---- data builders ----------------------------------------------------------
def load_data():
    t, smap = sector_universe(marketcap="Small", top_n_per_sector=40)
    alt = _alt_group(smap)
    sel = [x for x in t if smap.get(x) not in alt]   # search = non-alt sectors
    return _panel_for(sel, {x: smap[x] for x in sel})


def load_gen_data(label):
    if label == "mid_cap":
        t, smap = sector_universe(marketcap="Mid", top_n_per_sector=25)
        return _panel_for(t, smap)
    if label == "large_cap":
        t, smap = sector_universe(marketcap="Large", top_n_per_sector=25)
        return _panel_for(t, smap)
    if label == "small_alt_sectors":
        t, smap = sector_universe(marketcap="Small", top_n_per_sector=40)
        alt = _alt_group(smap)
        sel = [x for x in t if smap.get(x) in alt]   # disjoint sectors from search set
        return _panel_for(sel, {x: smap[x] for x in sel})
    raise ValueError(f"unknown gen universe: {label}")


# ---- soft expectation: the mechanism actually holds in-sample ---------------
def _check_long_beats_short(ctx):
    panel = ctx["panel"]
    hs = pd.Timestamp(ctx["holdout_start"])
    rets = panel.pct_change().replace([np.inf, -np.inf], np.nan)
    z = _issuance_z(panel.attrs["shares"], LOOKBACK)
    rank = z.rank(axis=1, pct=True)
    longs = rank.ge(1 - TOP_FRAC).shift(1, fill_value=False)
    shorts = rank.le(TOP_FRAC).shift(1, fill_value=False)
    spread = rets.where(longs).mean(axis=1) - rets.where(shorts).mean(axis=1)
    spread = spread[spread.index < hs]
    ann = float(spread.mean() * 252)
    return {"pass": ann > 0, "observed": round(ann, 4)}


SPEC = StrategySpec(
    id="auto_crypto_supply_inflation_dilution_premium_smith3_33634",
    family="equity_net_issuance",
    title="Net share-issuance (supply dilution) premium — US small/mid/large equities",
    markets=["US equities"],
    data_desc="Sharadar SEP adj-close (survivorship-clean) + SF1 sharesbas (PIT shares outstanding)",
    pre_registration=(
        "Mechanism (supply/dilution premium): firms that INFLATE share supply (net "
        "equity issuance) earn LOW forward returns; firms that CONTRACT supply "
        "(buybacks) earn HIGH forward returns. Signal = cross-sectional z-score of "
        "negative trailing ~1y growth in basic shares outstanding (PIT, datekey-based). "
        "Long lowest-issuance quintile / short highest, dollar-neutral, inverse-vol "
        "sized, weekly rebalance, 8bps cost, signals lagged 1 day. Scope is BROAD: a "
        "supply premium is a universal mechanism, so it must generalise to untouched "
        "cap tiers (mid, large) and an untouched, disjoint small-cap sector half. "
        "Originated as a crypto token-supply-inflation idea; no tested crypto supply "
        "adapter exists in the SDK, so it is tested in the equity analog "
        "(Pontiff-Woodgate 2008). Expectation: in-sample low-issuance leg beats "
        "high-issuance leg (positive long-minus-short spread)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": LOOKBACK, "vol_lb": VOL_LB, "top_frac": TOP_FRAC, "cost_bps": 8.0},
    grid={
        "default": {},
        "lookback_126": {"lookback": 126},
        "wider_tertile": {"top_frac": 0.33},
        "vol_42": {"vol_lb": 42},
    },
    scope="broad",
    generalization_universes=["mid_cap", "large_cap", "small_alt_sectors"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
    expectations=[
        {
            "name": "low_issuance_beats_high",
            "claim": "in-sample annualised long(low-issuance)-minus-short(high-issuance) spread > 0",
            "check": _check_long_beats_short,
        }
    ],
)