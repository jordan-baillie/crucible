"""Cross-sectional value (book-to-price) factor in small-cap US equities.

Pre-registered mechanism: the value premium (high book-to-price -> higher
expected return) is a UNIVERSAL risk/behavioural premium. It is strongest in
smaller, less-arbitraged names, so we search in small caps -- but if it is a
real premium it MUST also appear (frozen, OOS) in disjoint cap tiers
(micro / mid / large). Hence scope='broad'.

No side effects: pure data adapters + in-memory computation only.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel

START = "2010-01-01"

# Accumulated ticker -> sector map across every universe we build (search + gen).
# Populated in _build(); read by signal() for the trade ledger. In-memory only.
_SECTORS: dict = {}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _build(tickers, sector_map, start=START) -> pd.DataFrame:
    """Panel with a column MultiIndex: level0 in {'price','bp'}, level1=ticker."""
    _SECTORS.update(sector_map)

    price = sep_panel(tickers, start=start, field="closeadj")          # survivorship-clean
    price = price.dropna(how="all", axis=1)
    cols = list(price.columns)

    sf = sf1(cols, fields=["bvps"], dimension="ARQ")                   # filing-date fundamentals
    bvps = pit_panel(sf, "bvps", price.index, cols)                    # point-in-time, ffilled

    bp = bvps / price                                                  # book-to-price (value)
    bp = bp.replace([np.inf, -np.inf], np.nan)

    panel = pd.concat({"price": price, "bp": bp}, axis=1)
    return panel


def load_data() -> pd.DataFrame:
    # Small-cap is where the value anomaly lives (less arbitraged); bounded universe.
    tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=120)
    return _build(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    # Disjoint cap tiers (no overlap with the small-cap search universe). Kept small.
    mc = {"micro": "Micro", "mid": "Mid", "large": "Large"}[label]
    tickers, sector_map = sector_universe(marketcap=mc, top_n_per_sector=35)
    return _build(tickers, sector_map)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def _weights(panel: pd.DataFrame, q: float, vol_lb: int):
    price = panel["price"]
    bp = panel["bp"]
    rets = price.pct_change()

    z = xs_zscore(bp)                                                  # winsorized x-sec z-score

    inv_vol = 1.0 / rets.rolling(vol_lb).std()
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    ranks = z.rank(axis=1, pct=True)
    long_leg = (ranks >= 1.0 - q)
    short_leg = (ranks <= q)

    lw = (long_leg * inv_vol)
    sw = (short_leg * inv_vol)
    lw = lw.div(lw.sum(axis=1), axis=0)                               # inverse-vol within leg
    sw = sw.div(sw.sum(axis=1), axis=0)
    w = lw.fillna(0.0) - sw.fillna(0.0)                              # dollar-neutral long/short

    # Weekly rebalance: only refresh weights every 5 trading days, hold between.
    rebal = pd.Series(np.arange(len(w.index)) % 5 == 0, index=w.index)
    w = w.where(rebal, np.nan).ffill()
    return w, rets


def signal(panel, q=0.20, vol_lb=63, **params):
    w, rets = _weights(panel, q=q, vol_lb=vol_lb)

    # Weights built same-day from x-sec ranks -> lag 1 day to avoid look-ahead.
    W = w.shift(1)

    daily = net_of_cost(W, rets, cost_bps=8.0, name="value_bp_smallcap")
    sector_map = {t: _SECTORS.get(t, "Unknown") for t in W.columns}
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


# ---------------------------------------------------------------------------
# Soft expectations
# ---------------------------------------------------------------------------
def _check_cutoff_robust(ctx):
    """Mechanism claim: the value edge is not an artefact of the quintile cutoff."""
    g = ctx["grid"]
    obs = {}
    ok = True
    for k in ("q15", "q25"):
        if k in g and g[k] is not None and len(g[k]):
            m = float(g[k].mean())                                    # already search-window only
            obs[k] = m
            ok = ok and (m > 0)
        else:
            ok = False
    return {"pass": ok, "observed": obs}


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
SPEC = StrategySpec(
    id="value_bp_smallcap_xs",
    family="value",
    title="Cross-sectional book-to-price value, small-cap US equities",
    markets=["US equities"],
    data_desc="Sharadar SEP closeadj (survivorship-clean) + SF1 ARQ bvps (filing-date, PIT).",
    pre_registration=(
        "Long high book-to-price, short low book-to-price, dollar-neutral, inverse-vol "
        "weighted within each leg, weekly rebalance, 8bps cost, signal lagged 1 day. "
        "The value premium is a universal premium concentrated in less-arbitraged small "
        "caps (search universe); as a real premium it must also appear OOS in disjoint "
        "cap tiers (micro/mid/large), hence scope='broad'."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"q": 0.20, "vol_lb": 63},
    grid={
        "default": {},
        "q15": {"q": 0.15},
        "q25": {"q": 0.25},
        "vol126": {"vol_lb": 126},
    },
    scope="broad",
    generalization_universes=["micro", "mid", "large"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {
            "name": "cutoff_robust",
            "claim": "value edge positive in-search at both 15% and 25% quintile cutoffs",
            "check": _check_cutoff_robust,
        },
    ],
)