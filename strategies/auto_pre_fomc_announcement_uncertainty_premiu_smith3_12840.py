"""
pre_fomc_duration_v1 — Pre-FOMC announcement-uncertainty premium in long-duration Treasuries.

MECHANISM (event-conditional risk premium, not a Fed-decision forecast):
Compensation for bearing duration risk into the scheduled resolution of monetary-policy
uncertainty (Peng-Pan 2024 Treasury-side variant of the Lucca-Moench drift). FROZEN RULE
(exactly as pre-registered): long TLT — and ONLY TLT — on the trading day immediately
BEFORE each scheduled FOMC announcement (enter close T-2, exit close T-1 — deliberately
NOT holding through the announcement). Cash (0) on all other days (~8 events/yr).
IEF is the single PRE-REGISTERED secondary expression (declared grid variant), NOT part
of the default rule. No multi-ETF basket: the thesis is a pure max-duration expression,
and blending TLH/IEF into the default was a deviation from the frozen rule.

Position sizing: the single position is inverse-vol scaled to a 10% annualized vol
target (capped at 1.0 — no leverage), using trailing vol through the PRIOR close only.

Calendar dates are public and known years in advance, so the only information used at
the close of T-2 is the calendar + trailing vol through T-2.

EXCLUSIONS: unscheduled/emergency moves (1994-04-18, 1998-10-15, 2001-01-03/04-18/09-17,
2008-01-22/10-08, 2020-03-03/03-15) are NOT in the calendar — they are not pre-scheduled
uncertainty and would be lookahead.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------------
# Scheduled FOMC announcement dates (decision day = day 2 of two-day meetings),
# 1994 (first public same-day announcements) through the published 2026 schedule.
# Emergency/intermeeting actions deliberately excluded (see module docstring).
# ----------------------------------------------------------------------------------
FOMC_SCHEDULED = [
    # 1994
    "1994-02-04", "1994-03-22", "1994-05-17", "1994-07-06", "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    # 1995
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06", "1995-08-22", "1995-09-26", "1995-11-15", "1995-12-19",
    # 1996
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03", "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02", "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01", "1998-08-18", "1998-09-29", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30", "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27", "2001-08-21", "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (March scheduled meeting replaced by the 2020-03-15 emergency action -> excluded)
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (published schedule)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

SECTOR_MAP = {
    "TLT": "UST 20+Y",
    "IEF": "UST 7-10Y",
    "UST5Y_PROXY": "UST 5Y proxy",
    "UST10Y_PROXY": "UST 10Y proxy",
    "UST30Y_PROXY": "UST 30Y proxy",
}


def _held_days(idx, window_days):
    """Trading days HELD: the `window_days` trading days strictly before each announcement."""
    held = set()
    last = idx[-1]
    for d in FOMC_SCHEDULED:
        a = pd.Timestamp(d)
        pos = idx.searchsorted(a)  # first trading day >= announcement date
        if pos == len(idx) and (a - last).days > 4:
            continue  # announcement far beyond data end -> skip (avoid spurious last-day holds)
        for k in range(1, int(window_days) + 1):
            j = pos - k
            if 0 <= j < len(idx):
                held.add(idx[j])
    return pd.DatetimeIndex(sorted(held))


def load_data():
    # TLT = the frozen default instrument; IEF carried in the panel ONLY for the
    # pre-registered secondary grid variant (never blended into the default rule).
    # yf_panel is the sanctioned source for ETFs per the data catalog.
    return yf_panel(["TLT", "IEF"], start="2002-01-01")


def _duration_proxy(fred_id, duration, name, start="1994-01-01"):
    """Constant-maturity total-return proxy: ret_t = -D * dY_t + Y_{t-1}/252 (carry + price)."""
    y = fred_series({fred_id: "y"}, start=start)["y"].ffill().dropna() / 100.0
    ret = -duration * y.diff() + y.shift(1) / 252.0
    px = 100.0 * (1.0 + ret.fillna(0.0)).cumprod()
    return pd.DataFrame({name: px})


def load_gen_data(label):
    # Internal-consistency battery (ticker-DISJOINT from the search universe): a real
    # announcement-uncertainty premium must show the same sign across the duration curve,
    # with magnitude increasing in duration. FRED proxies extend the sample to 1994.
    if label == "dgs5_proxy_1994":
        return _duration_proxy("DGS5", 4.6, "UST5Y_PROXY")
    if label == "dgs10_proxy_1994":
        return _duration_proxy("DGS10", 8.5, "UST10Y_PROXY")
    if label == "dgs30_proxy_1994":
        return _duration_proxy("DGS30", 17.5, "UST30Y_PROXY")
    raise KeyError(f"unknown generalization universe: {label}")


def signal(panel, window_days=1, instrument="TLT", vol_lb=63, target_vol=0.10, cost_bps=8.0):
    """FROZEN RULE: long ONE instrument (default TLT) on the pre-announcement day(s) only.

    Sizing: single-position inverse-vol scaling to `target_vol` annualized, capped at 1.0
    (no leverage), using trailing vol through the PRIOR close only (shift(1)). On the
    generalization proxy panels (single column, name != TLT) the panel's sole column is
    used — same frozen rule, different duration point.
    """
    px = panel.sort_index().dropna(how="all")
    col = instrument if instrument in px.columns else px.columns[0]
    rets = px.pct_change(fill_method=None)
    idx = rets.index

    held = _held_days(idx, int(window_days))
    mask = idx.isin(held)

    # Inverse-vol size of the SINGLE position: vol through the prior close only
    # (.shift(1) -> at the close of decision day T-2 we size with data through T-2;
    # the calendar itself is public years ahead, so day-selection has no lookahead).
    ann_vol = rets[col].rolling(int(vol_lb), min_periods=21).std() * np.sqrt(252.0)
    size = (float(target_vol) / ann_vol).clip(upper=1.0).shift(1)
    avail = px[col].shift(1).notna()  # had a price yesterday -> tradeable today
    size = size.where(np.isfinite(size))
    size = size.fillna(1.0).where(avail, 0.0)  # warm-up fallback: full unit position

    w = pd.Series(0.0, index=idx)
    w[mask] = size[mask]

    # W.loc[t] = weights HELD during day t's return (decided at close t-1). This is already
    # the lagged matrix net_of_cost expects — no further shift needed.
    W = pd.DataFrame(0.0, index=idx, columns=px.columns)
    W[col] = w

    r = rets.fillna(0.0)
    daily = net_of_cost(W, r, cost_bps=cost_bps, name="pre_fomc_duration")
    sectors = {c: SECTOR_MAP.get(c, "Rates") for c in px.columns}
    trades = trades_from_weights(W, r, sectors)
    return daily, trades


SPEC = StrategySpec(
    id="pre_fomc_duration_v1",
    family="macro_announcement_premium",
    title="Pre-FOMC announcement-uncertainty premium in long-duration Treasuries",
    markets=["us_treasury_etf"],
    data_desc=(
        "Hardcoded public FOMC scheduled-announcement calendar 1994-2026 (emergency meetings "
        "excluded); TLT (default) and IEF (pre-registered secondary) closes via yf_panel (2002+); "
        "FRED DGS5/DGS10/DGS30 constant-maturity total-return proxies (1994+) for the "
        "duration-curve generalization battery."
    ),
    pre_registration=(
        "CLAIM: holding long duration ONLY on the trading day before scheduled FOMC announcements "
        "earns a positive premium (uncertainty-resolution compensation), exiting at the close "
        "before the announcement (no decision-direction bet). FROZEN: long TLT ONLY (no basket; "
        "IEF is the single pre-registered secondary expression, declared as a grid variant), "
        "window=1 pre-announcement day, single-position inverse-vol sizing to 10% ann. vol "
        "capped at 1x, ~8 events/yr, 8bps costs. NULL (MCPT): random placebo date sets of equal "
        "count — tests whether the FOMC calendar adds anything over generic long-duration "
        "exposure; unconditional non-event TLT drift is NOT part of the claim. POWER: ~190 "
        "event-days 2002-2026 (TLT), ~260 with the 1994 FRED proxy battery. CONSISTENCY "
        "REQUIREMENT: same sign across the disjoint duration-curve proxies (5y/10y/30y), "
        "magnitude increasing in duration. DECAY TEST: post-2015 subsample (the equity drift's "
        "documented decay point) inspected at review; equity pre-FOMC drift decay is known and "
        "is explicitly not the claim. Grid (full declared search burden): default TLT, "
        "pre-registered IEF secondary, window=2 concentration check."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"window_days": 1, "instrument": "TLT", "vol_lb": 63, "target_vol": 0.10, "cost_bps": 8.0},
    grid={
        "default": {},
        "ief_secondary": {"instrument": "IEF"},  # the pre-registered secondary expression
        "window2": {"window_days": 2},           # enter T-3: is the premium concentrated in T-1?
    },
    scope="local",
    generalization_universes=["dgs5_proxy_1994", "dgs10_proxy_1994", "dgs30_proxy_1994"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=1,
)