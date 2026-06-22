"""Cross-sectional 12-1 price momentum (Jegadeesh-Titman / Asness "Momentum Everywhere").

Replaces a dead premise: the prior module targeted *cross-exchange crypto funding
dispersion* via a `funding_rates()` adapter that does not exist -- we own no crypto
or funding data (see DATA_CATALOG.md). That edge is unimplementable here, so this is
a real, broad universal premium on owned Sharadar SEP equities instead.

Mechanism: delayed diffusion of information -> recent past relative winners keep
outperforming relative losers for ~3-12 months; skipping the most recent month
avoids 1-month short-term reversal contamination. Theory says this is universal, so
scope='broad': a stage-1 pass must generalise to disjoint cap tiers untouched by the
mid-cap search universe (small / large / micro).
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

_START = "2010-01-01"
# Ticker->sector is stable across cap tiers; accumulate across every panel build so
# signal() can recover sectors even if a sliced panel loses DataFrame.attrs.
_SECTOR_MAP: dict[str, str] = {}


def _build_panel(marketcap: str, top_n_per_sector: int) -> pd.DataFrame:
    tickers, smap = sector_universe(marketcap=marketcap, top_n_per_sector=top_n_per_sector)
    px = sep_panel(tickers, start=_START, field="closeadj").astype(float)
    px = px.dropna(axis=1, how="all").sort_index()
    _SECTOR_MAP.update({t: smap.get(t, "Unknown") for t in tickers})
    px.attrs["sector_map"] = {t: _SECTOR_MAP.get(t, "Unknown") for t in px.columns}
    return px


def load_data() -> pd.DataFrame:
    # Search universe: liquid US mid-caps, sector-spread (~11 sectors x 100 ~= 1100 names).
    return _build_panel(marketcap="Mid", top_n_per_sector=100)


def load_gen_data(label: str) -> pd.DataFrame:
    # DISJOINT cap tiers from the mid-cap search universe (Sharadar scalemarketcap
    # buckets share no tickers across tiers). Kept small (~150-400 names each).
    caps = {"small": "Small", "large": "Large", "micro": "Micro"}
    return _build_panel(marketcap=caps[label], top_n_per_sector=30)


def signal(panel, lookback=252, skip=21, target_vol=0.10, vol_lb=63,
           max_pos=40, rebalance="W"):
    px = panel.sort_index()
    sector_map = panel.attrs.get("sector_map") or {
        t: _SECTOR_MAP.get(t, "Unknown") for t in px.columns
    }
    rets = px.pct_change()

    # 12-1 momentum: ratio of price `skip` days ago to `lookback` days ago.
    # Uses only PAST prices at each date -> no look-ahead in the raw signal.
    mom = px.shift(skip) / px.shift(lookback) - 1.0
    sig = xs_zscore(mom)  # cross-sectional, winsorised, NaN-preserving

    # inv_vol_position returns WEEKLY-HELD, LAGGED, inverse-vol positions (it owns the
    # 1-day execution lag). It is already lagged -> pass straight to net_of_cost /
    # trades_from_weights, do NOT .shift(1) again (that would double-lag).
    W = inv_vol_position(sig, rets, target_vol=target_vol, vol_lb=vol_lb,
                         max_pos=max_pos, rebalance=rebalance)

    daily = net_of_cost(W, rets, cost_bps=8.0, name="xs_mom_12_1")
    trades = trades_from_weights(W, rets, sector_map)  # kit stamps entry_regime
    return daily, trades


# ---- soft expectation: the skip-month mechanism claim, machine-checked ----------
def _ann_sharpe(r) -> float:
    if r is None:
        return 0.0
    r = pd.Series(r).dropna()
    if len(r) < 60 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(252.0))


def _check_skip_month(ctx):
    # Compare pre-declared grid variants (free; no extra signal() call). All grid
    # series are search-window only (< holdout_start) per the harness contract.
    g = ctx.get("grid", {}) or {}
    base, noskip = g.get("default"), g.get("skip0")
    if base is None or noskip is None:
        return {"pass": False, "observed": "grid variants unavailable"}
    s_skip, s_noskip = _ann_sharpe(base), _ann_sharpe(noskip)
    return {
        "pass": s_skip >= s_noskip - 0.05,
        "observed": f"skip-month Sharpe={s_skip:.2f} vs no-skip Sharpe={s_noskip:.2f}",
    }


SPEC = StrategySpec(
    id="xs_momentum_12_1_midcap",
    family="cross_sectional_momentum",
    title="Cross-sectional 12-1 price momentum (mid-cap US equities)",
    markets=["US equities"],
    data_desc="Sharadar SEP split/div-adjusted daily closes, delisted included "
              "(survivorship-clean); sector-spread mid-cap universe via sector_universe.",
    pre_registration=(
        "HYPOTHESIS: cross-sectional 12-1 momentum (12-month return skipping the most "
        "recent month) is a UNIVERSAL premium (Jegadeesh-Titman 1993; Asness-Moskowitz-"
        "Pedersen 'Value and Momentum Everywhere' 2013), driven by delayed information "
        "diffusion / underreaction. Long relative winners, short relative losers; "
        "winsorised xs z-score, inverse-vol sizing, weekly rebalance, 8bps cost on "
        "turnover, signal lagged 1 day (positions are next-day-executed). Skipping the "
        "last month avoids 1-month short-term reversal, so the skip-month variant should "
        "match-or-beat the no-skip variant in-sample (machine-checked below). Because the "
        "mechanism is theory-universal, scope='broad': the frozen default signal must "
        "stay OOS-positive on >=60% of three DISJOINT cap tiers (small/large/micro) it "
        "never searched. PRIMARY = default params; grid declares the honest search burden."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(lookback=252, skip=21, target_vol=0.10, vol_lb=63,
                        max_pos=40, rebalance="W"),
    grid={
        "default": {},
        "lb126": {"lookback": 126},   # 6-1 momentum
        "skip0": {"skip": 0},         # 12-0 (no reversal skip) -- mechanism control
        "tv15": {"target_vol": 0.15},
    },
    scope="broad",
    generalization_universes=["small", "large", "micro"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[{
        "name": "skip_month_helps",
        "claim": "12-1 momentum (skip last month) in-sample Sharpe >= 12-0 (no skip) - 0.05",
        "check": _check_skip_month,
    }],
)