"""Cross-sectional 12-1 momentum, US equities (broad universal-premium book).

Mechanism: relative-strength momentum (skip the most recent month to dodge
short-term reversal) is a textbook UNIVERSAL premium -> scope='broad'. We search
in a sector-spread mid-cap universe and require generalisation to disjoint cap
tiers (small / large / micro) on holdout.

No look-ahead: the momentum signal uses only past prices (shift), and the held
weights come from inv_vol_position, whose output is already 1-day lagged and
weekly-held (adapter contract) -> net_of_cost / trades_from_weights receive an
ALREADY-LAGGED W, so we do NOT shift again.
"""
import functools

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

START = "2005-01-01"

# Disjoint generalisation universes (cap tiers that share no tickers with the
# Mid-cap search universe). label -> (cap_tier, top_n_per_sector).
_GEN = {
    "small_cap": ("Small", 30),
    "large_cap": ("Large", 25),
    "micro_cap": ("Micro", 30),
}


@functools.lru_cache(maxsize=1)
def _search_universe():
    # Sector-spread mid-cap universe + {ticker: sector} map for the trade ledger.
    # ~40 names x ~11 sectors ~= 440 liquid names (bounded -> CPCV-safe).
    return sector_universe("Mid", 40)


def _panel_with_sectors(tickers, sector_map):
    px = sep_panel(tickers, start=START).sort_index()
    px.attrs["sector_map"] = {t: sector_map[t] for t in px.columns if t in sector_map}
    return px


def load_data() -> pd.DataFrame:
    tickers, sector_map = _search_universe()
    return _panel_with_sectors(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    cap, tpn = _GEN[label]
    search = set(_search_universe()[0])
    tickers, sector_map = sector_universe(cap, tpn)
    tickers = [t for t in tickers if t not in search]  # enforce disjointness
    return _panel_with_sectors(tickers, sector_map)


def signal(panel, **params):
    lookback = int(params.get("lookback", 252))
    skip = int(params.get("skip", 21))
    target_vol = float(params.get("target_vol", 0.10))
    vol_lb = int(params.get("vol_lb", 63))
    max_pos = int(params.get("max_pos", 30))
    sector_map = panel.attrs.get("sector_map", {})

    px = panel.sort_index()
    rets = px.pct_change()

    # 12-1 momentum: cumulative return from t-lookback to t-skip (past data only).
    mom = px.shift(skip) / px.shift(lookback) - 1.0
    z = xs_zscore(mom)  # winsorized cross-sectional z, NaN-preserving

    # Inverse-vol sized, weekly-rebalanced, signed long-short positions.
    # inv_vol_position output is ALREADY 1-day lagged -> no extra shift here.
    W = inv_vol_position(z, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=max_pos, rebalance="W")

    daily = net_of_cost(W, rets, cost_bps=8.0, name="xs_momentum_midcap")
    trades = trades_from_weights(W, rets, sector_map)  # auto-stamps entry_regime
    return daily, trades


# --- soft expectation: the skipped-month mechanism must actually help ---------
def _sharpe(r):
    if r is None:
        return 0.0
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _check_skip_helps(ctx):
    grid = ctx.get("grid", {})
    base, skip0 = grid.get("default"), grid.get("skip0")
    if base is None or skip0 is None:
        return {"pass": False, "observed": "missing grid variant (default/skip0)"}
    sb, s0 = _sharpe(base), _sharpe(skip0)
    return {"pass": sb >= s0,
            "observed": f"skip-21 Sharpe {sb:.2f} vs skip-0 Sharpe {s0:.2f}"}


SPEC = StrategySpec(
    id="xs_momentum_midcap_12_1",
    family="momentum",
    title="Cross-sectional 12-1 momentum, mid-cap US equities",
    markets=["US equities"],
    data_desc=("Sharadar SEP split/div-adjusted daily closes; sector-spread "
               "mid-cap search universe (~440 names), gens on disjoint cap tiers."),
    pre_registration=(
        "Relative-strength momentum is a universal cross-sectional premium. We "
        "rank names by their 12-month return skipping the most recent month "
        "(t-252..t-21) to avoid short-term reversal, go long winners / short "
        "losers, inverse-vol sized, weekly rebalanced, 8bps cost. Mechanism "
        "claim (machine-checked): skipping the most recent month improves the "
        "search-window Sharpe vs no-skip (skip0). Being a universal premium, the "
        "frozen default signal must remain OOS-positive on >=60% of the disjoint "
        "small/large/micro-cap generalisation universes on holdout."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "skip0": {"skip": 0},            # falsification leg for the skip claim
        "lb_126": {"lookback": 126},
        "tv_15": {"target_vol": 0.15},
    },
    scope="broad",
    generalization_universes=list(_GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "skip_month_helps",
         "claim": "12-1 (skip=21) search Sharpe >= 12-0 (skip=0) search Sharpe",
         "check": _check_skip_helps},
    ],
)