"""
Gain-Reversal-Frequency Factor — long-only liquidity-provision tilt in survivorship-clean
US small caps (close-to-close re-registration of the Akbas-style reversal-frequency idea).

FIX (2026-06-12): the original module requested field='open' from sep_panel; the SEP parquet
cache schema does NOT contain 'open' (pyarrow column-projection failure — same SEP-cache
schema gap class as the dividend-month triage finding). The signal is therefore RE-REGISTERED
on closes only; no field outside the known-good set {closeadj, close, volume} is touched
(close+volume validated in production by the Amihud dollar-volume strategies).

FROZEN DESIGN (pre-registered):
- Signal = RF: trailing-252d fraction of days where the PRIOR day's close-to-close return was
  POSITIVE and today's is NEGATIVE — i.e. noise-trader optimism reversed next day; high-RF
  names are persistent reversal venues whose holders earn a liquidity-provision premium.
  Requires >=200 valid day-pairs.
- Universe: Sharadar small caps (delisted included), sector-spread, median 63d dollar volume > $1M.
- LONG-ONLY equal-weight top quintile of RF, monthly rebalance (the monthly cross-sectional
  version that survives costs — NOT the dead daily reversal trade).
- ANTI-BOUNCE: P&L is close-to-close on closeadj only; weights are shift(1)-lagged before
  net_of_cost (signal formed at month-end t is held from t+1).
- Costs: 25bps one-way (small/mid). No hedge sleeve — standalone premium test first (2026-06-08 law).
- scope='broad': stage-2 on 3 DISJOINT universes (mid-cap, large-cap, disjoint small-cap slice);
  prediction = positive but monotonically weaker as cap rises (limits-to-arbitrage gradient).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2004-01-01"
# Known-good SEP cache fields ONLY ('open' is absent from the cache schema — see FIX above).
FIELDS = ["closeadj", "close", "volume"]
GEN_LABELS = ["mid_cap", "large_cap", "small_disjoint"]

# Sector map shared across search + generalization universes; the loaders populate it
# before signal() is ever called on the corresponding panel.
SECTOR_MAP: dict = {}
_UNI: dict = {}


def _search_tickers():
    if "search" not in _UNI:
        tickers, smap = sector_universe(marketcap="Small", top_n_per_sector=35)
        SECTOR_MAP.update(smap)
        _UNI["search"] = list(tickers)
    return _UNI["search"]


def _panel(tickers) -> pd.DataFrame:
    """MultiIndex-column panel: level0 = field, level1 = ticker. Columns restricted to
    tickers present in ALL fields so the dollar-volume filter is always computable."""
    parts = {f: sep_panel(tickers, START, field=f) for f in FIELDS}
    common = sorted(set.intersection(*[set(p.columns) for p in parts.values()]))
    return pd.concat({f: parts[f].reindex(columns=common) for f in FIELDS}, axis=1)


def load_data() -> pd.DataFrame:
    return _panel(_search_tickers())


def load_gen_data(label) -> pd.DataFrame:
    search = set(_search_tickers())
    if label == "mid_cap":
        tickers, smap = sector_universe(marketcap="Mid", top_n_per_sector=25)
    elif label == "large_cap":
        tickers, smap = sector_universe(marketcap="Large", top_n_per_sector=20)
    elif label == "small_disjoint":
        # Same cap tier, deeper per-sector slice; explicit filter below removes any overlap.
        tickers, smap = sector_universe(marketcap="Small", top_n_per_sector=70)
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    tickers = [t for t in tickers if t not in search]  # enforce disjointness for all labels
    SECTOR_MAP.update(smap)
    return _panel(tickers)


def signal(panel, rf_window=252, min_valid=200, top_frac=0.20,
           adv_min=1e6, min_eligible=50, cost_bps=25.0, **_):
    closeadj = panel["closeadj"]
    close = panel["close"].where(panel["close"] > 0)
    volume = panel["volume"]

    # P&L universe: close-to-close on closeadj ONLY.
    rets = closeadj.pct_change()

    # --- Signal: gain-reversal frequency (closes only; trailing data up to t) ---
    valid = rets.notna() & rets.shift(1).notna()
    rev = ((rets.shift(1) > 0) & (rets < 0)) & valid     # yesterday's gain reversed today
    n_valid = valid.rolling(rf_window).sum()
    rf = rev.rolling(rf_window).sum() / n_valid.replace(0, np.nan)
    rf = rf.where(n_valid >= min_valid)

    # Tradability filter: trailing 63d median dollar volume (uses only data up to t).
    dollar_vol = (close * volume).rolling(63).median()
    eligible = rf.notna() & (dollar_vol > adv_min)

    # --- Monthly rebalance: equal-weight top quintile of RF among eligible names ---
    idx = closeadj.index
    reb_dates = idx.to_series().groupby(idx.to_period("M")).max().values

    W = pd.DataFrame(np.nan, index=idx, columns=closeadj.columns)
    for t in reb_dates:
        row = rf.loc[t].where(eligible.loc[t]).dropna()
        if len(row) < min_eligible:
            continue  # too thin: carry previous book (ffill below)
        n_top = max(int(np.ceil(len(row) * top_frac)), 10)
        top = row.nlargest(n_top).index
        W.loc[t] = 0.0
        W.loc[t, top] = 1.0 / n_top
    W = W.ffill().fillna(0.0)

    # Lag is OUR responsibility: signals formed at month-end t are held from t+1.
    W_lag = W.shift(1).fillna(0.0)

    daily = net_of_cost(W_lag, rets, cost_bps=cost_bps, name="gain_reversal_freq_v1")
    trades = trades_from_weights(W_lag, rets, SECTOR_MAP)
    return daily, trades


SPEC = StrategySpec(
    id="gain_reversal_frequency_v1",
    family="liquidity_provision",
    title="Gain-Reversal-Frequency factor — long-only top-quintile RF, small caps, monthly",
    markets=["US_equities_smallcap"],
    data_desc=("Sharadar SEP (survivorship-clean, delisted incl.): closeadj/close/volume only "
               "(SEP cache schema lacks 'open'); sector-spread small-cap search universe "
               "(~385 names), median 63d dollar-volume > $1M filter"),
    pre_registration=(
        "Liquidity-provision premium: holders of names where one-day gains are persistently "
        "reversed the next day intermediate recurring noise-trader optimism and are compensated "
        "for it. Signal = trailing-252d frequency of (ret_{t-1}>0 AND ret_t<0), >=200 valid "
        "day-pairs; a FREQUENCY-of-reversal sort, mechanistically DISTINCT from Amihud "
        "illiquidity-LEVEL (price-impact magnitude) sorts. FROZEN: long-only equal-weight TOP "
        "quintile, monthly rebalance, 25bps one-way costs, close-to-close P&L on closeadj with "
        "shift(1)-lagged weights. Standalone, no hedge sleeve. PREDICTION: stage-1 pass "
        "generalizes with positive-but-monotonically-weaker premium up the cap ladder "
        "(small_disjoint > mid > large); a flat or inverted cap gradient flags a confound and "
        "rejects the mechanism."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"rf_window": 252, "min_valid": 200, "top_frac": 0.20,
                    "adv_min": 1e6, "min_eligible": 50, "cost_bps": 25.0},
    grid={
        "default": {},
        "rf_126": {"rf_window": 126, "min_valid": 100},
        "top_30pct": {"top_frac": 0.30},
        "adv_2m": {"adv_min": 2e6},
    },
    scope="broad",
    generalization_universes=GEN_LABELS,
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=100,
)