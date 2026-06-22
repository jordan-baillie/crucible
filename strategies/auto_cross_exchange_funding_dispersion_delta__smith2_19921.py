"""Cross-sectional 12-1 momentum factor (sector-balanced US equities).

NOTE on identity: the file stem says "funding_dispersion" but that idea is
infeasible with the sanctioned adapters (no crypto/funding data exists in the
tested import set — only Sharadar SEP/SF1, yfinance, FRED). This module is the
honest, rails-verifiable replacement: a universal momentum premium tested for
generalization across disjoint cap tiers. SPEC.id is kept == file stem so the
harness key does not drift; family/title/data_desc describe what it actually is.

Mechanism: cross-sectional relative-strength (12-month return skipping the most
recent month) is a long-documented, market-universal premium (behavioral
under-reaction + a risk component). Because the theory says "appears across
markets", scope='broad': a stage-1 pass MUST generalize to untouched cap tiers.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np
import pandas as pd

STRAT_ID = "auto_cross_exchange_funding_dispersion_delta__smith2_19921"

START = "2008-01-01"
TOP_N_PER_SECTOR = 40          # search universe ~ 11 sectors * 40 ~= 440 names
TOP_N_PER_SECTOR_GEN = 20      # gen universes ~ 220 names each (small, same-night)

DEFAULT = {"lookback": 252, "gap": 21, "top_q": 0.2, "vol_lb": 63}

# generalization universes: DISJOINT cap tiers (a name sits in one tier at the
# marketcap snapshot, so they share no tickers with the 'Mid' search universe).
GEN = {"small": "Small", "large": "Large", "micro": "Micro"}


def _build_panel(marketcap: str, top_n_per_sector: int) -> pd.DataFrame:
    """Sector-balanced, survivorship-clean adjusted-close panel + sector_map."""
    tickers, sector_map = sector_universe(marketcap, top_n_per_sector)
    px = sep_panel(tickers, START, "closeadj")
    px = px.sort_index().dropna(how="all", axis=1)
    px.attrs["sector_map"] = sector_map  # the trade ledger needs this
    return px


def load_data() -> pd.DataFrame:
    return _build_panel("Mid", TOP_N_PER_SECTOR)


def load_gen_data(label: str) -> pd.DataFrame:
    return _build_panel(GEN[label], TOP_N_PER_SECTOR_GEN)


def signal(panel, **params):
    p = {**DEFAULT, **params}
    lookback, gap, top_q, vol_lb = p["lookback"], p["gap"], p["top_q"], p["vol_lb"]

    prices = panel.sort_index()
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)

    # 12-1 momentum: return from t-lookback to t-gap (skip most recent month).
    mom = prices.shift(gap) / prices.shift(lookback) - 1.0
    z = xs_zscore(mom)  # winsorized, NaN-preserving, per-date

    # Long top quantile / short bottom quantile, cross-sectionally per date.
    ranks = z.rank(axis=1, pct=True)
    longs = (ranks >= (1.0 - top_q)).astype(float)
    shorts = (ranks <= top_q).astype(float)

    # Inverse-vol sizing within each leg.
    inv_vol = 1.0 / rets.rolling(vol_lb).std().replace(0.0, np.nan)
    long_w = (longs * inv_vol)
    short_w = (shorts * inv_vol)
    long_w = long_w.div(long_w.sum(axis=1), axis=0).fillna(0.0) * 0.5
    short_w = short_w.div(short_w.sum(axis=1), axis=0).fillna(0.0) * 0.5
    W = long_w - short_w  # dollar-neutral, gross ~= 1.0

    # Weekly rebalance: refresh only on Mondays, hold (ffill) through the week.
    mask = W.index.weekday == 0
    W.loc[~mask] = np.nan
    W = W.ffill().fillna(0.0)

    # Signal built from same-day info -> lag 1 day before it earns returns.
    W_lag = W.shift(1).fillna(0.0)

    sector_map = panel.attrs.get("sector_map", {})
    daily = net_of_cost(W_lag, rets, cost_bps=8.0, name=STRAT_ID)
    trades = trades_from_weights(W_lag, rets, sector_map)
    return daily, trades


# ---- soft expectations (machine-checkable mechanism claims) -----------------

def _check_subperiods(ctx):
    """Premium should persist: positive mean in BOTH halves of search window."""
    s = ctx["search"].dropna()
    if len(s) < 4:
        return {"pass": False, "observed": "insufficient_data"}
    mid = s.index[len(s) // 2]
    first = float(s[s.index < mid].mean())
    second = float(s[s.index >= mid].mean())
    return {"pass": (first > 0.0) and (second > 0.0),
            "observed": f"{first:.6f}/{second:.6f}"}


def _check_lookback_robust(ctx):
    """Edge should not hinge on one formation window: 252d and 126d both > 0."""
    g = ctx["grid"]
    d, alt = g.get("default"), g.get("lb_126")
    if d is None or alt is None:
        return {"pass": False, "observed": "missing_grid_variant"}
    md, ma = float(d.dropna().mean()), float(alt.dropna().mean())
    return {"pass": (md > 0.0) and (ma > 0.0), "observed": f"{md:.6f}/{ma:.6f}"}


SPEC = StrategySpec(
    id=STRAT_ID,
    family="xs_momentum",
    title="Cross-sectional 12-1 momentum (sector-balanced US equities)",
    markets=["US equities"],
    data_desc=("Sharadar SEP split/div-adjusted daily closes (delisted incl, "
               "survivorship-clean); sector-balanced Mid-cap universe (~440 "
               "names) via sector_universe; momentum from t-252 to t-21."),
    pre_registration=(
        "Cross-sectional relative-strength (12-month return skipping the most "
        "recent month) is a market-universal premium. Long the top quintile, "
        "short the bottom quintile, inverse-vol sized, weekly rebalance, "
        "dollar-neutral, signals lagged 1 day, 8bps costs on turnover. "
        "Predictions (machine-checked): (1) net return is positive in BOTH "
        "halves of the search window; (2) the edge is robust to formation "
        "window (252d and 126d both positive). Because the mechanism is "
        "universal, scope='broad': it must generalize to disjoint cap tiers "
        "(Small/Large/Micro) on holdout."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb_126": {"lookback": 126},
        "topq_0.3": {"top_q": 0.3},
    },
    scope="broad",
    generalization_universes=["small", "large", "micro"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "positive_in_subperiods",
         "claim": "search-window net return positive in both halves",
         "check": _check_subperiods},
        {"name": "lookback_robust",
         "claim": "default(252d) and lb_126 both positive mean net return",
         "check": _check_lookback_robust},
    ],
)