"""
Dividend-Month Premium — predictable ex-dividend price pressure (Hartzmark & Solomon, JFE 2013).

MECHANISM (event-flow / liquidity-provision premium, NOT a fundamental forecast):
stocks earn abnormal returns in months they are PREDICTED to pay a dividend, driven by
mechanical demand from income-seeking investors ahead of ex-div dates. We get paid for
holding inventory against that predictable flow.

FIX vs failed runs: the SEP parquet behind sep_panel has NO `dividends` column (run 1)
and NO `close` column either (run 2 — pyarrow schema error on the second sep_panel read).
Only `closeadj` is confirmed present. Ex-div events are therefore inferred from closeadj
vs a COMPANION unadjusted/split-adjusted series, tried in a fallback chain
('closeunadj' -> 'close' — closeunadj is the canonical Sharadar pair-field stored next to
closeadj): on an ex-date the dividend-adjusted return exceeds the companion return by the
dividend yield. Because `closeunadj` is also unadjusted for SPLITS, the event flag uses a
BAND, 10bps < gap < 15%: dividends live well inside it, while split jumps (5:4 split ->
gap ~ +20%; 2:1 -> ~ +100%; reverse splits -> negative) fall outside. The event is stamped
on the ex-date itself (same information timing as before) and remains strictly
point-in-time: the predictor only uses events >= ~11 months old.

FROZEN CONSTRUCTION (declared before any result is seen):
- A stock is a "predicted payer" for calendar month m iff it had an (inferred) ex-dividend
  event in calendar month m-12.
- LONG-ONLY inverse-vol book of predicted-payer mid-caps, weight cap, monthly rebalance,
  residual (post-cap) stays in cash. No hedge leg in the registered primary (the harness
  MCPT beta-adjusted null isolates the long-only/low-vol confound).
- Costs 15bps on turnover (mid-cap; turnover structurally bounded by quarterly payer cycles).

Search universe: sector-spread MID caps. Generalization (scope='broad', disjoint by
construction): large-cap tier, small-cap tier (cost-stressed), and the NEXT mid-cap tier
sharing no tickers with the search set. Breadth bar registered as SAME-SIGN.

The only novel code here is the predicted-payer signal; everything else is the mandated kit.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2004-01-01"   # 12m predictor warm-up before any tradeable signal exists
DIV_GAP_MIN = 0.001    # 10bps adj-vs-companion return gap = ex-div event (filters rounding noise)
DIV_GAP_MAX = 0.15     # band ceiling: rejects split jumps when the companion is closeunadj

# ticker -> sector, filled by the loaders (harness always loads before calling signal)
_SECTOR_MAP: dict = {}


# ----------------------------------------------------------------------------- universes

@lru_cache(maxsize=None)
def _univ(tier: str, per_sector: int):
    tickers, smap = sector_universe(marketcap=tier, top_n_per_sector=per_sector)
    return tuple(tickers), dict(smap)


def _companion_panel(tickers) -> pd.DataFrame:
    """Non-div-adjusted companion to closeadj, with a schema fallback chain.

    The SEP parquet schema has already rejected 'dividends' and 'close'; try the
    canonical Sharadar pair-field 'closeunadj' first, then 'close' for portability.
    """
    last_err = None
    for field in ("closeunadj", "close"):
        try:
            return sep_panel(list(tickers), START, field=field)
        except Exception as err:  # pyarrow schema miss -> try next candidate
            last_err = err
    raise RuntimeError(
        f"SEP parquet exposes neither 'closeunadj' nor 'close'; cannot infer ex-div "
        f"events from owned data. Last error: {last_err!r}"
    )


def _panel(tickers) -> pd.DataFrame:
    """Two-block panel: ('px', t) = closeadj, ('ex', t) = 1.0 on inferred ex-div dates.

    Ex-div inference: closeadj (split+div adjusted) return minus companion return equals
    the dividend yield on ex-dates and ~0 otherwise; split jumps in the unadjusted
    companion are excluded by the (DIV_GAP_MIN, DIV_GAP_MAX) band. Stamped on the
    ex-date, point-in-time by construction.
    """
    adj = sep_panel(list(tickers), START, field="closeadj")
    cmp_px = _companion_panel(tickers)
    cols = [t for t in adj.columns if t in cmp_px.columns]
    adj, cmp_px = adj[cols], cmp_px[cols]
    gap = adj.pct_change() - cmp_px.pct_change()
    ex = ((gap > DIV_GAP_MIN) & (gap < DIV_GAP_MAX)).astype(float)
    return pd.concat({"px": adj, "ex": ex}, axis=1)


def load_data() -> pd.DataFrame:
    """Search universe: sector-spread mid caps (~11 sectors x 35 ≈ 385 names)."""
    tickers, smap = _univ("Mid", 35)
    _SECTOR_MAP.update(smap)
    return _panel(tickers)


def load_gen_data(label: str) -> pd.DataFrame:
    """Generalization universes — all filtered DISJOINT from the search set."""
    search = set(_univ("Mid", 35)[0])
    if label == "large_cap":
        tickers, smap = _univ("Large", 25)          # ~275 names, expect weaker/same-sign
    elif label == "small_cap":
        tickers, smap = _univ("Small", 30)          # ~330 names, signal-only read
    elif label == "mid_next_tier":
        wide, smap = _univ("Mid", 70)               # next mid tier beyond the search names
        tickers = [t for t in wide if t not in search]
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    tickers = [t for t in tickers if t not in search]  # enforce disjointness everywhere
    _SECTOR_MAP.update(smap)
    return _panel(tickers)


# ----------------------------------------------------------------------------- signal

def signal(panel: pd.DataFrame, *, vol_lb: int = 63, max_w: float = 0.04,
           min_names: int = 10, cost_bps: float = 15.0):
    px = panel["px"]
    ex = panel["ex"]
    rets = px.pct_change()

    # --- predicted-payer indicator (the ONLY novel logic) -----------------------------
    # paid[month, t] = had an (inferred) ex-div event that calendar month; predictor for
    # month m is paid[m-12] — information is >= ~11 months old when month m begins. PIT.
    months = ex.index.to_period("M")
    paid = (ex > 0).groupby(months).max()           # PeriodIndex('M') x tickers, bool
    pred = paid.shift(12)                           # predicted payer for each month

    day_month = rets.index.to_period("M")
    pred_daily = pred.reindex(day_month)            # broadcast month row to its days
    pred_daily.index = rets.index
    pred_daily = pred_daily.fillna(False).astype(float)

    # --- inverse-vol sizing, monthly rebalance, cap, cash residual --------------------
    vol = rets.rolling(vol_lb).std()
    raw = pred_daily * (1.0 / vol.replace(0.0, np.nan))

    first_of_month = ~pd.Series(day_month, index=rets.index).duplicated()
    W = raw[first_of_month.values].reindex(rets.index).ffill()   # freeze within month

    gross = W.sum(axis=1)
    W = W.div(gross.replace(0.0, np.nan), axis=0)
    W = W.clip(upper=max_w)                          # cap; residual stays in cash
    n_names = (W > 0).sum(axis=1)
    W = W.where(n_names >= min_names, other=0.0)     # too-thin book -> cash, no 2-name books
    W = W.fillna(0.0)

    # --- lag is MY responsibility: weights act from the NEXT close --------------------
    W_lag = W.shift(1)

    net = net_of_cost(W_lag, rets, cost_bps=cost_bps, name="div_month_premium_mid")
    smap = {t: _SECTOR_MAP.get(t, "Unknown") for t in W.columns}
    trades = trades_from_weights(W_lag, rets, smap)  # kit stamps entry_regime — never by hand
    return net, trades


# ----------------------------------------------------------------------------- spec

GRID = {
    "default": {},                                  # primary, registered
    "tight_cap": {"max_w": 0.03},                   # concentration robustness
    "slow_vol": {"vol_lb": 126},                    # sizing-lookback robustness
    "stress_costs": {"cost_bps": 25.0},             # mid-cap cost stress
}

SPEC = StrategySpec(
    id="div_month_premium_mid",
    family="event_flow_pressure",
    title="Dividend-Month Premium (predicted ex-div payers, mid-cap long-only)",
    markets=["US_EQ_MID"],
    data_desc=("Sharadar SEP closeadj + unadjusted companion (closeunadj/close fallback "
               "chain); ex-div events inferred from the dividend-adjusted-vs-companion "
               "return gap on the ex-date (10bps<gap<15% band rejects split jumps); "
               "survivorship-clean (delisted incl.) sector-spread mid-cap universe"),
    pre_registration=(
        "H: stocks predicted to go ex-dividend this calendar month (had an ex-div event in the "
        "same month 12 months ago — event inferred on its ex-date from the SEP closeadj-vs-"
        "companion return gap in the 10bps-15% band; info >=11 months old) earn abnormal returns "
        "from predictable income-seeking flow (Hartzmark & Solomon 2013). Frozen rule: long-only "
        "inv-vol book of predicted mid-cap payers, 4% cap, monthly rebalance, cash residual, "
        "15bps costs. PRIMARY STATISTIC: MCPT beta-adjusted selection alpha vs equal-weight-"
        "universe null (mandatory — long-only book is structurally low-vol/quality tilted; raw "
        "Sharpe is NOT the registered claim). Generalization bar: same-SIGN selection alpha in "
        ">=60% of disjoint universes (large caps may null per the liquidity lesson; sign, not "
        "significance, is the bar). No hedge leg tested until the standalone premium clears "
        "stage 1."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope="broad",
    generalization_universes=["large_cap", "small_cap", "mid_next_tier"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
)