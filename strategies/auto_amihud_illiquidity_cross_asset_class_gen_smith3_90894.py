"""
Cross-sectional price momentum (12-1) as a long-only factor book.

Mechanism (pre-registration): The momentum premium is a UNIVERSAL behavioural/risk
premium documented across equity markets, cap tiers and sectors (Jegadeesh-Titman,
Asness et al.). Past 12-month-skip-1-month winners continue to outperform on a
weekly-rebalanced, inverse-vol-weighted long book. Because the claim is a universal
mechanism, scope='broad': a stage-1 pass MUST generalise to untouched cap tiers
(small / large / micro) on their holdouts, or it is an overfit outlier and rejected.

Search universe: Mid-cap, sector-spread (~220 names). Generalization universes are
DISJOINT by cap tier (no shared tickers with Mid): Small, Large, Micro.

No look-ahead: momentum uses panel.shift(skip..lookback); positions are weekly,
inverse-vol, and inv_vol_position returns ALREADY-LAGGED weights (held t+1), so we
pass them straight to net_of_cost / trades_from_weights (no further .shift needed).
Costs: 8bps on turnover (net_of_cost default).
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2004-01-01"

# cap tier per universe label (search + generalization)
_CAP = {
    "search": "Mid",
    "small":  "Small",
    "large":  "Large",
    "micro":  "Micro",
}
_TOP_PER_SECTOR = {"Mid": 20, "Small": 25, "Large": 18, "Micro": 30}


def _build_panel(cap: str) -> pd.DataFrame:
    """Survivorship-clean price panel for one cap tier, sector_map stashed in attrs."""
    tickers, sector_map = sector_universe(marketcap=cap,
                                          top_n_per_sector=_TOP_PER_SECTOR[cap])
    panel = sep_panel(tickers, START, field="closeadj")
    # keep only names we have a sector for (trade ledger needs it)
    panel = panel[[c for c in panel.columns if c in sector_map]]
    panel.attrs["sector_map"] = {t: sector_map[t] for t in panel.columns}
    return panel


def load_data() -> pd.DataFrame:
    return _build_panel(_CAP["search"])


def load_gen_data(label: str) -> pd.DataFrame:
    # label is one of generalization_universes; disjoint cap tier from the search panel
    return _build_panel(_CAP[label])


def signal(panel, lookback=252, skip=21, max_pos=15,
           target_vol=0.10, vol_lb=63, **params):
    sector_map = panel.attrs.get("sector_map", {})
    rets = panel.pct_change()

    # 12-1 momentum: cumulative return from t-lookback to t-skip (skip recent month)
    mom = panel.shift(skip) / panel.shift(lookback) - 1.0

    # cross-sectional rank, long the winners only (long-only factor book)
    z = xs_zscore(mom)
    long_sig = z.where(z > 0)

    # inverse-vol, weekly, max_pos -> ALREADY-LAGGED positions (held next day)
    W = inv_vol_position(long_sig, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=int(max_pos), rebalance="W")

    daily = net_of_cost(W, rets, cost_bps=8.0, name="momentum_12_1")
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


# ---- soft expectation: momentum premium present in BOTH halves of search window ----
def _check_subperiod_robust(ctx):
    r = ctx["search"].dropna()
    if len(r) < 60:
        return {"pass": False, "observed": "insufficient_history"}
    mid = len(r) // 2
    h1 = float(r.iloc[:mid].mean())
    h2 = float(r.iloc[mid:].mean())
    return {"pass": bool(h1 > 0 and h2 > 0),
            "observed": f"h1_mean={h1:.6f}, h2_mean={h2:.6f}"}


SPEC = StrategySpec(
    id="xs_momentum_12_1_midcap",
    family="momentum",
    title="Cross-sectional 12-1 momentum, long-only inverse-vol (mid-cap)",
    markets=["US equities"],
    data_desc="Sharadar SEP closeadj, sector-spread mid-cap (~220 names); "
              "gen universes small/large/micro cap (disjoint by cap tier).",
    pre_registration=(
        "Past 12-month-skip-1-month winners continue to outperform. Universal "
        "behavioural/risk momentum premium -> must generalise across cap tiers. "
        "Long-only, weekly, inverse-vol, 8bps costs, signals lagged 1 day. "
        "Expect positive returns in both halves of the search window."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":      {},
        "fast":         {"lookback": 126},
        "slow":         {"lookback": 378},
        "concentrated": {"max_pos": 10},
    },
    scope="broad",
    generalization_universes=["small", "large", "micro"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=15,
    expectations=[
        {"name": "subperiod_robust",
         "claim": "momentum long book positive in BOTH halves of search window",
         "check": _check_subperiod_robust},
    ],
)