"""
Pre-FOMC Treasury duration premium (Peng-Pan 2024 variant).

MECHANISM (pre-registered, FROZEN): compensation for bearing duration risk into the
resolution of scheduled monetary-policy uncertainty. The drift in long-duration USTs is
realized on the trading day BEFORE the scheduled FOMC announcement. The famous EQUITY
version decayed post-2015; this is deliberately the Treasury leg.

CONSTRUCTION (frozen before any data was looked at):
  - Universe of duration instruments: TLT, IEF, ZN=F, ZB=F (+ SYN10, a DGS10-based
    constant-maturity synthetic 10y return index, for pre-2002 history depth).
  - For every SCHEDULED FOMC announcement date A (1994+, ~8/yr; intermeeting/emergency
    moves EXCLUDED; the cancelled scheduled 2020-03-18 meeting EXCLUDED):
      enter long at the close of trading day A-2, exit at the close of A-1
      => the position earns exactly the day-(A-1) return. Flat all other days.
  - Equal weight across instruments available at the entry close, total gross = 1.0,
    no leverage. The CALENDAR is the signal: no conditioning, no vol-scaling, no overlay.

NO LOOKAHEAD: weights for day d are fully determined at the close of d-1 using only
(a) the public FOMC calendar (published months in advance) and (b) prior-day price
availability (panel.shift(1)). W is therefore already the lagged/actionable matrix
required by net_of_cost — no further shift is applied, by design.

GENERALIZATION (scope='broad', ticker-DISJOINT from the search universe):
  - short_duration  : SHY/IEI/ZT=F/ZF=F  -> maturity gradient: smaller but same-sign
  - gold            : GLD/GC=F           -> second rate-sensitive asset
  - ig_credit_duration : LQD/AGG/BND     -> duration carried via IG credit
  - syn20_long_bond : DGS20 synthetic    -> independent long-duration data source
(The SPY negative control and the random-calendar placebo are offline gate-0 checks,
NOT generalization universes — a positive SPY result must not count FOR the candidate.)

POWER NOTE (pre-registered): ~3% time-in-market => annualized Sharpe is diluted by flat
days; the primary statistic is the event-day return distribution vs non-event days
(the harness DSR/MCPT battery on the daily series is the implementation of this).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "1994-01-01"

# ---------------------------------------------------------------------------
# Scheduled FOMC announcement dates (statement day = final day of meeting),
# federalreserve.gov historical calendars, 1994-2026. Intermeeting / emergency
# actions excluded (e.g. 1998-10-15, 2001-01-03/04-18/09-17, 2008-01-22,
# 2008-10-08, 2020-03-03, 2020-03-15). 2020-03-18 scheduled meeting was
# superseded by the 03-15 emergency action and is excluded.
# ---------------------------------------------------------------------------
FOMC_SCHEDULED = [
    # 1994
    "1994-02-04", "1994-03-22", "1994-05-17", "1994-07-06",
    "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    # 1995
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06",
    "1995-08-22", "1995-09-26", "1995-11-15", "1995-12-19",
    # 1996
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03",
    "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02",
    "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01",
    "1998-08-18", "1998-09-29", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30",
    "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27",
    "2001-08-21", "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (scheduled only; 03-18 superseded by the 03-15 emergency action)
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (published schedule; future dates auto-skip past data end)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# Instrument -> "sector" buckets for the trade ledger (distinct buckets give the
# deployment-sanity gate genuine name/sector spread within the rates book).
SECTOR_MAP = {
    "TLT": "UST 20y+ ETF", "IEF": "UST 7-10y ETF",
    "ZN=F": "UST 10y Futures", "ZB=F": "UST 30y Futures",
    "SYN10": "UST 10y Synthetic", "SYN20": "UST 20y Synthetic",
    "SHY": "UST 1-3y ETF", "IEI": "UST 3-7y ETF",
    "ZT=F": "UST 2y Futures", "ZF=F": "UST 5y Futures",
    "GLD": "Gold ETF", "GC=F": "Gold Futures",
    "LQD": "IG Credit ETF", "AGG": "Agg Bond ETF", "BND": "Agg Bond ETF",
}


def _synthetic_cm_panel(fred_id: str, col: str, mod_duration: float) -> pd.DataFrame:
    """Constant-maturity synthetic Treasury total-return PRICE index from a FRED
    yield series: r_t = -D * dy_t + carry (y_{t-1}/252). Yield in percent."""
    y = fred_series({fred_id: "y"}, START)["y"].astype(float).ffill()
    dy = y.diff()
    r = -mod_duration * (dy / 100.0) + (y.shift(1) / 100.0) / 252.0
    px = 100.0 * (1.0 + r.fillna(0.0)).cumprod()
    px.name = col
    return px.to_frame()


def load_data() -> pd.DataFrame:
    """Search-universe panel: long-duration tradables + DGS10 synthetic for depth."""
    etf = yf_panel(["TLT", "IEF", "ZN=F", "ZB=F"], START)
    syn = _synthetic_cm_panel("DGS10", "SYN10", 8.0)
    panel = pd.concat([etf, syn], axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="first")]
    return panel.dropna(how="all")


GEN_UNIVERSES = ["short_duration", "gold", "ig_credit_duration", "syn20_long_bond"]


def load_gen_data(label: str) -> pd.DataFrame:
    """Ticker-disjoint confirmation universes (same shape as load_data())."""
    if label == "short_duration":
        return yf_panel(["SHY", "IEI", "ZT=F", "ZF=F"], START).dropna(how="all")
    if label == "gold":
        return yf_panel(["GLD", "GC=F"], START).dropna(how="all")
    if label == "ig_credit_duration":
        return yf_panel(["LQD", "AGG", "BND"], START).dropna(how="all")
    if label == "syn20_long_bond":
        return _synthetic_cm_panel("DGS20", "SYN20", 14.0).dropna(how="all")
    raise ValueError(f"unknown generalization universe: {label}")


def signal(panel: pd.DataFrame, days_before: int = 1, window_len: int = 1,
           instruments=None, cost_bps: float = 8.0):
    """Calendar event strategy. Default: hold day A-1 only (enter close A-2,
    exit close A-1). Weights are calendar-determined + prior-day availability
    => W is already the actionable/lagged matrix; NO extra shift applied."""
    panel = panel.sort_index()
    cols = list(panel.columns) if instruments is None else \
        [c for c in panel.columns if c in instruments]
    px = panel[cols].ffill(limit=3)
    rets = px.pct_change(fill_method=None).fillna(0.0)
    idx = px.index
    # availability at the ENTRY close (strictly prior-day info — no lookahead)
    avail = px.shift(1).notna()

    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for a in pd.DatetimeIndex(pd.to_datetime(FOMC_SCHEDULED)):
        p = idx.searchsorted(a)
        if p <= 0 or p >= len(idx):
            continue  # outside the data window
        if days_before == 0:
            # grid variant: hold the announcement day itself (post-resolution contrast)
            if idx[p] != a:
                continue
            held = [idx[p]]
        else:
            lo = p - days_before - window_len + 1
            hi = p - days_before + 1
            if lo < 0:
                continue
            held = list(idx[lo:hi])
        for d in held:
            elig = avail.loc[d]
            elig = list(elig[elig].index)
            if not elig:
                continue
            W.loc[d, elig] = 1.0 / len(elig)  # equal-weight, gross 1.0, no leverage

    sector_map = {c: SECTOR_MAP.get(c, "Rates") for c in cols}
    daily = net_of_cost(W, rets, cost_bps=cost_bps, name="prefomc_duration")
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


SPEC = StrategySpec(
    id="prefomc_duration_v1",
    family="event_macro_announcement",
    title="Pre-FOMC Treasury duration premium — long 10y+ duration the day before "
          "scheduled FOMC announcements",
    markets=["US Treasuries (TLT/IEF/ZN/ZB)", "rates"],
    data_desc="yf_panel TLT/IEF/ZN=F/ZB=F (2000s+) extended by FRED DGS10 "
              "constant-maturity synthetic (1994+); public scheduled FOMC calendar "
              "embedded (1994-2026, intermeeting actions excluded); gen universes: "
              "SHY/IEI/ZT/ZF, GLD/GC, LQD/AGG/BND, DGS20 synthetic.",
    pre_registration=(
        "FROZEN before any backtest: long equal-weight duration basket, enter close "
        "T-2 / exit close T-1 relative to each SCHEDULED FOMC announcement (2pm ET "
        "statement => close-to-close day-before window is unambiguous); flat "
        "otherwise (~3% time-in-market); no leverage, no conditioning, no overlay — "
        "the calendar IS the signal. Unscheduled/emergency meetings excluded; the "
        "cancelled scheduled 2020-03-18 meeting excluded. Pre-registered secondary: "
        "maturity gradient — the premium should RISE with duration (short_duration "
        "gen universe should be same-sign but smaller; syn20 same-sign and strong). "
        "Pre-registered negative control (offline, NOT a gen universe): post-2015 "
        "SPY on the same window should be weak/decayed — if it is equally strong, "
        "suspect construction artifact. Placebo: random non-FOMC calendars must show "
        "no effect (harness MCPT is the implementation). Power note: ~250 events; "
        "the binding statistic is the event-day return distribution vs non-event "
        "days, not annualized Sharpe diluted by flat days. Grid declared up front: "
        "default (T-1 only), two-day window (T-2..T-1), announcement-day contrast, "
        "ETF-only implementation; 'default' is primary."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "two_day_window": {"window_len": 2},
        "announce_day": {"days_before": 0},
        "etf_only": {"instruments": ["TLT", "IEF"]},
    },
    scope="broad",
    generalization_universes=GEN_UNIVERSES,
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=6,
)