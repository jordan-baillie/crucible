"""
Cross-sectional ILLIQUIDITY PREMIUM (Amihud) on owned, survivorship-clean US small-caps.

NOTE ON PIVOT: this idea was scoped against a crypto venue, but the tested adapter kit
exposes NO crypto panel (no binance/deribit adapter exists — DATA_CATALOG confirms OWNED
data is Sharadar US equities + FRED/yf futures). Rather than reinvent/download raw crypto
(forbidden), the universal illiquidity premium (Amihud 2002 — documented across markets and
asset classes) is implemented where we have clean data AND where the premium is least
arbitraged: small-cap Domestic Common Stock. scope='broad' because the mechanism is universal;
it must generalise to disjoint (sector-partitioned) small-cap universes in the stage-2 battery.

Mechanism: ILLIQ_i = trailing-mean( |ret_i| / dollar-volume_i ). Cross-sectionally, MORE
illiquid names earn higher returns than liquid ones -> long top-illiquidity quintile, short
bottom, dollar-neutral, inverse-vol within legs, weekly rebalance, 1-day lag, 8bps costs.
"""
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2005-01-01"

# Search universe = small-cap names from these sectors.
SEARCH_SECTORS = ["Technology", "Healthcare", "Industrials", "Consumer Cyclical"]
SEARCH_TOP_N = 250  # per sector  (~1000 names, bounded -> CPCV-safe)

# Generalization universes = OTHER sectors (DISJOINT tickers, same illiquid cap tier).
GEN_UNIVERSES = {
    "fin_realestate":    ["Financial Services", "Real Estate"],
    "energy_materials":  ["Energy", "Basic Materials"],
    "staples_utilities": ["Consumer Defensive", "Utilities"],
}
GEN_TOP_N = 175  # per sector  (~350 names each)

# Disjoint sector partitions => disjoint tickers => a union map has no key collisions,
# so this global is a safe fallback if panel.attrs are stripped by the harness.
_SECTOR_MAP = {}


def _build_universe(sectors, top_n_per_sector):
    tickers, sector_map = [], {}
    for sec in sectors:
        ts = us_universe(sector=sec, category="Domestic Common Stock",
                         marketcap="Small", include_delisted=True, top_n=top_n_per_sector)
        for t in ts:
            sector_map[t] = sec
        tickers.extend(ts)
    return tickers, sector_map


def _load_panel(tickers, sector_map):
    px = sep_panel(tickers, START, field="closeadj")
    vol = sep_panel(tickers, START, field="volume")
    common = px.columns.intersection(vol.columns)
    px, vol = px[common], vol[common]
    panel = pd.concat({"price": px, "volume": vol}, axis=1)  # cols = MultiIndex(field, ticker)
    sm = {t: sector_map[t] for t in common if t in sector_map}
    _SECTOR_MAP.update(sm)
    panel.attrs["sector_map"] = sm
    return panel


def load_data() -> pd.DataFrame:
    tickers, sector_map = _build_universe(SEARCH_SECTORS, SEARCH_TOP_N)
    return _load_panel(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    tickers, sector_map = _build_universe(GEN_UNIVERSES[label], GEN_TOP_N)
    return _load_panel(tickers, sector_map)


def _weekly_lag(W, every=5):
    """Weekly rebalance (hold target weights, refresh every 5 trading days) + 1-day lag.
    The lag lives HERE: weights formed from data <=t are applied to t+1 returns (no look-ahead).
    The matrix returned is already lagged -> pass straight to net_of_cost / trades_from_weights."""
    W = W.copy()
    non_rebal = (np.arange(len(W)) % every) != 0
    W.iloc[non_rebal] = np.nan
    return W.ffill().fillna(0.0).shift(1).fillna(0.0)


def signal(panel, **params):
    illiq_lb = int(params.get("illiq_lb", 60))
    vol_lb   = int(params.get("vol_lb", 60))
    top_q    = float(params.get("top_q", 0.2))
    leg_mode = params.get("_leg_mode", "both")  # internal diagnostic only (expectations)

    px = panel["price"].astype(float)
    vol = panel["volume"].astype(float)
    sector_map = {**_SECTOR_MAP, **panel.attrs.get("sector_map", {})}

    rets = px.pct_change()
    dollar_vol = (px * vol).replace(0.0, np.nan)
    illiq = rets.abs() / dollar_vol
    illiq_bar = illiq.rolling(illiq_lb, min_periods=max(10, illiq_lb // 2)).mean()
    z = xs_zscore(np.log(illiq_bar.where(illiq_bar > 0)))  # log to tame Amihud skew

    vol_est = rets.rolling(vol_lb, min_periods=max(10, vol_lb // 2)).std()
    inv_vol = (1.0 / vol_est).replace([np.inf, -np.inf], np.nan)

    pr = z.rank(axis=1, pct=True)
    long_sel = pr >= (1.0 - top_q)   # most illiquid
    short_sel = pr <= top_q          # most liquid

    def _leg(sel):
        w = (sel.astype(float) * inv_vol)
        w = w.where(np.isfinite(w), 0.0)
        s = w.sum(axis=1)
        return w.div(s.where(s > 0), axis=0).fillna(0.0)

    Wl, Ws = _leg(long_sel), _leg(short_sel)

    if leg_mode == "spread":  # diagnostic: long-minus-short premium (one signal() call)
        lr = net_of_cost(_weekly_lag(Wl), rets, cost_bps=8.0, name="long")
        sr = net_of_cost(_weekly_lag(Ws), rets, cost_bps=8.0, name="short")
        return (lr - sr).rename("illiq_spread"), []

    W = _weekly_lag(0.5 * Wl - 0.5 * Ws)  # dollar-neutral, already lagged
    daily = net_of_cost(W, rets, cost_bps=8.0, name="illiq_premium")
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


# ---- soft expectations (machine-checkable mechanism claims) ----
def _chk_premium_sign(ctx):
    h = pd.Timestamp(ctx["holdout_start"])
    spread, _ = ctx["spec"].signal(ctx["panel"], _leg_mode="spread")  # one extra signal() call
    spread = spread[spread.index < h]
    m = float(spread.mean())
    return {"pass": bool(m > 0), "observed": m}


def _chk_weekly_hold(ctx):
    trades = ctx["trades"] or []
    if not trades:
        return {"pass": False, "observed": 0.0}
    mean_hd = float(np.mean([t.get("hold_days", 0) for t in trades]))
    return {"pass": bool(mean_hd >= 4.0), "observed": mean_hd}


SPEC = StrategySpec(
    id="auto_crypto_cross_sectional_illiquidity_premi_smith2_31003",
    family="illiquidity_premium",
    title="Amihud illiquidity premium (cross-sectional, US small-cap equities)",
    markets=["US small-cap equities"],
    data_desc="Sharadar SEP daily closeadj + volume (survivorship-clean, delisted incl.); "
              "small-cap Domestic Common Stock, sector-partitioned.",
    pre_registration=(
        "Universal liquidity premium (Amihud 2002). Signal = trailing-mean |ret|/dollar-volume; "
        "long most-illiquid quintile, short most-liquid, dollar-neutral, inverse-vol within legs, "
        "weekly rebalance, 1-day lag, 8bps. Originally scoped for crypto, but no crypto adapter "
        "exists in the tested kit; the premium is universal so it is tested on owned data in the "
        "small-cap tier where it is least arbitraged. Claims: (1) long(illiquid) leg out-earns "
        "short(liquid) leg in-sample [_chk_premium_sign]; (2) weekly rebalance => mean hold ~1wk "
        "[_chk_weekly_hold]; (3) being broad, it generalises to disjoint-sector small-cap universes."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb30":    {"illiq_lb": 30},
        "lb120":   {"illiq_lb": 120},
        "q10":     {"top_q": 0.10},
        "volb120": {"vol_lb": 120},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "premium_sign_long_beats_short",
         "claim": "in-sample mean of (illiquid long leg - liquid short leg) net return > 0",
         "check": _chk_premium_sign},
        {"name": "weekly_rebalance_hold",
         "claim": "mean trade hold_days >= 4 (consistent with weekly rebalance)",
         "check": _chk_weekly_hold},
    ],
)