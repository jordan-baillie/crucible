import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

# ---------------------------------------------------------------------------
# Cross-sectional Amihud illiquidity premium (market-neutral small-cap book).
# Mechanism: illiquid names earn a compensation premium (Amihud 2002). We rank
# names by trailing mean(|ret| / dollar-volume); go long the most illiquid,
# short the most liquid, dollar-neutral, inverse-vol sized, weekly rebalance.
# A genuine liquidity premium is a UNIVERSAL mechanism -> scope='broad': it
# must also show up in disjoint cap tiers (micro / mid), fading in liquid
# large-caps. All signals are lagged 1 day; costs ~8bps on turnover.
# NO external side effects.
# ---------------------------------------------------------------------------

START = "2006-01-01"

# universe_key -> {ticker: sector}; used as a robust fallback for the trade
# ledger's sector map (ticker->sector is universe-independent, so a merge is safe).
_SECTOR_MAPS = {}


def _make_panel(marketcap, top_n_per_sector, key):
    """Build a (close, dollar-volume) MultiIndex-column panel for one universe."""
    tickers, sector_map = sector_universe(marketcap, top_n_per_sector)
    _SECTOR_MAPS[key] = dict(sector_map)
    close = sep_panel(tickers, START, field="closeadj").astype(float)
    vol = sep_panel(tickers, START, field="volume").astype(float)
    cols = close.columns.intersection(vol.columns)
    close = close[cols]
    vol = vol.reindex(columns=cols)
    dvol = close * vol  # dollar-volume proxy (relative scale only matters)
    panel = pd.concat({"close": close, "dvol": dvol}, axis=1)
    panel.attrs["sector_map"] = dict(sector_map)
    return panel


def load_data():
    # Search universe: sector-spread small-cap (~990 names) — where the
    # illiquidity premium lives, not arbitraged away as in large/liquid names.
    return _make_panel("Small", 90, "small")


# Stage-2 generalization universes: DISJOINT cap tiers (no shared tickers),
# each kept small (~330 names) so they run same-night.
_GEN = {"micro": ("Micro", 30), "mid": ("Mid", 30), "large": ("Large", 30)}


def load_gen_data(label):
    mc, n = _GEN[label]
    return _make_panel(mc, n, label)


def _sector_map_for(panel):
    sm = panel.attrs.get("sector_map")
    if sm:
        return sm
    merged = {}
    for m in _SECTOR_MAPS.values():
        merged.update(m)
    return merged


def _normalize(w):
    """Dollar-neutral weights, gross ~= 1 (0.5 long + 0.5 short)."""
    longs = w.clip(lower=0.0)
    shorts = (-w).clip(lower=0.0)
    longs = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0)
    shorts = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0)
    return (longs.fillna(0.0) - shorts.fillna(0.0)) * 0.5


def signal(panel, **params):
    illiq_lb = int(params.get("illiq_lb", 21))
    vol_lb = int(params.get("vol_lb", 63))
    top_frac = float(params.get("top_frac", 0.2))
    reverse = bool(params.get("reverse", False))  # for soft-expectation check only

    close = panel["close"].astype(float)
    dvol = panel["dvol"].astype(float)
    rets = close.pct_change()

    # Amihud illiquidity: trailing mean(|ret| / dollar-volume). All data <= t.
    amihud = (rets.abs() / dvol.replace(0.0, np.nan)) * 1e6
    illiq = amihud.rolling(illiq_lb, min_periods=max(5, illiq_lb // 2)).mean()

    z = xs_zscore(illiq)            # higher illiquidity -> higher expected return
    if reverse:
        z = -z

    rank = z.rank(axis=1, pct=True)
    long = (rank >= (1.0 - top_frac)).astype(float)
    short = (rank <= top_frac).astype(float)
    raw = long - short

    # Inverse-vol sizing within legs.
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()
    raw = raw * (1.0 / vol.replace(0.0, np.nan))

    W_sig = _normalize(raw)

    # Weekly rebalance: refresh weights only on the first trading day of each
    # ISO week, hold constant in between.
    periods = W_sig.index.to_period("W")
    first = ~periods.duplicated()
    mask2d = np.broadcast_to(first[:, None], W_sig.shape)
    W_held = W_sig.where(mask2d).ffill()

    # Lag 1 day: weights are computed from data through t, so they can only be
    # traded at t+1. The shift is MY responsibility (net_of_cost does not lag).
    W = W_held.shift(1)

    ret = net_of_cost(W, rets, cost_bps=8.0, name="amihud_illiq")
    fvi = ret.first_valid_index()
    if fvi is not None:
        ret = ret.loc[fvi:]
    ret.name = "amihud_illiq"
    trades = trades_from_weights(W, rets, _sector_map_for(panel))
    return ret, trades


# ----------------------------- soft expectations ---------------------------
def _chk_premium(ctx):
    s = ctx["search"].dropna()
    m = float(s.mean())
    return {"pass": bool(m > 0), "observed": m}


def _chk_direction(ctx):
    # ONE extra signal() call; everything recomputed is sliced to < holdout.
    panel = ctx["panel"]
    hs = pd.Timestamp(ctx["holdout_start"])
    rev, _ = signal(panel, reverse=True)
    rev = rev[rev.index < hs].dropna()
    base = ctx["search"]
    base = base[base.index < hs].dropna()
    diff = float(rev.mean() - base.mean())
    return {"pass": bool(rev.mean() < base.mean()), "observed": diff}


SPEC = StrategySpec(
    id="amihud_illiquidity_premium_smallcap",
    family="liquidity",
    title="Cross-sectional Amihud illiquidity premium (small-cap, market-neutral)",
    markets=["US equities"],
    data_desc="Sharadar SEP daily closeadj + volume; sector-spread small-cap universe.",
    pre_registration=(
        "Amihud illiquidity (trailing mean |ret|/dollar-volume, 21d) is "
        "cross-sectionally priced: illiquid names earn a compensation premium. "
        "Long top-quintile illiquidity, short bottom-quintile, dollar-neutral, "
        "inverse-vol sized, weekly rebalance, signals lagged 1 day, 8bps cost. "
        "As a universal liquidity premium it should generalise to disjoint cap "
        "tiers (micro/mid) and weaken in liquid large-caps."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "illiq_lb_42": {"illiq_lb": 42},
        "vol_lb_42": {"vol_lb": 42},
        "top_frac_30": {"top_frac": 0.3},
    },
    scope="broad",
    generalization_universes=["micro", "mid", "large"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {
            "name": "illiquidity_premium_positive",
            "claim": "high-minus-low Amihud illiquidity earns positive search-window return",
            "check": _chk_premium,
        },
        {
            "name": "direction_correct",
            "claim": "reversing the sign (long liquid) underperforms the illiquidity book in-sample",
            "check": _chk_direction,
        },
    ],
)