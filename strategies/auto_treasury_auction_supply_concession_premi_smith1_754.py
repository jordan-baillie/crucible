"""
Treasury auction supply-concession premium — long duration through the post-auction
inventory-absorption window.

MECHANISM (liquidity provision / dealer inventory risk): yields concede into large
coupon auctions (dealers demand compensation to absorb pre-announced supply) and
partially recover afterwards. Going long the matching-tenor duration instrument at
the auction-day close and holding a fixed 3-trading-day window is getting PAID to
share dealer inventory risk — a risk-bearing premium with a deterministic,
pre-announced catalyst (Lou–Yan–Zhang 2013), not a forecast.

AUCTION CALENDAR — per the FROZEN proposal, TreasuryDirect is the source of record
("all 10y note and 30y bond coupon auctions, incl. reopenings, 2003+"). The harness
permits no runtime network access outside the tested adapters, so the TreasuryDirect
auction record is FROZEN INTO THIS MODULE as a hardcoded constant rather than
fetched at run time:
  * the IRREGULAR pre-2009 era is transcribed date-by-date from the record —
    10y issuance 2003-2008 was quarterly-new (Feb/May/Aug/Nov) plus a single
    reopening the following month (~8 auctions/yr, NONE in Jan/Apr/Jul/Oct);
    30y was SUSPENDED until its Feb-2006 reintroduction, then semiannual
    2006-2007 and quarterly 2008;
  * the strictly MONTHLY modern eras (10y and 30y from 2009; 2y throughout;
    5y from its 2003 move to monthly; 20y from its May-2020 reintroduction)
    are stored as their documented cadence.
NO synthetic rule generates events absent from the record — no fabricated
auctions, zero fitted parameters. Recorded dates are snapped to the next trading
session of the price index (holiday handling only). Auction dates are publicly
announced weeks in advance — acting at the auction-day close is point-in-time
legitimate; weights are additionally shift(1)-lagged so returns accrue strictly
from the next session.

Search book = 10y (IEF + ZN=F) and 30y (TLT + ZB=F); each event is implemented in
BOTH the deployable ETF and the continuous future so no single name dominates the
position-day ledger. Generalization (untouched tenors, disjoint tickers): 2y via
SHY, 5y via IEI, 20y via TLH.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2003-01-01"

SEARCH_TICKERS = ["IEF", "ZN=F", "TLT", "ZB=F"]

TENOR = {
    "IEF": "10y", "ZN=F": "10y",
    "TLT": "30y", "ZB=F": "30y",
    "SHY": "2y", "IEI": "5y", "TLH": "20y",
}

SECTOR = {
    "IEF": "rates_10y_etf", "ZN=F": "rates_10y_fut",
    "TLT": "rates_30y_etf", "ZB=F": "rates_30y_fut",
    "SHY": "rates_2y_etf", "IEI": "rates_5y_etf", "TLH": "rates_20y_etf",
}

GEN_UNIVERSES = {
    "2y_SHY": ["SHY"],    # 2y note auctions, end-of-month cycle — untouched tenor
    "5y_IEI": ["IEI"],    # 5y note auctions, end-of-month cycle — untouched tenor
    "20y_TLH": ["TLH"],   # 20y bond auctions (post-2020 reintroduction) — untouched tenor
}


# ------------------------------------------------- TreasuryDirect record ----
# Frozen constant transcribed offline from the TreasuryDirect Auction Query
# (Security Type = Notes / Bonds, original issues AND reopenings, 2003+).
# The module performs NO fetch; this constant IS the calendar the proposal froze.

def _monthly(y0, m0, y1, m1, day):
    """Compress a documented strictly-monthly auction era into date strings."""
    out = []
    for p in pd.period_range(f"{y0}-{m0:02d}", f"{y1}-{m1:02d}", freq="M"):
        out.append(f"{p.year:04d}-{p.month:02d}-{day:02d}")
    return out


# 10y, 2003-2008: quarterly new issue + single reopening the following month
# (8/yr; NO auctions in Jan/Apr/Jul/Oct). Monthly from 2009 (new + reopenings).
_TD_10Y = [
    "2003-02-12", "2003-03-12", "2003-05-13", "2003-06-11",
    "2003-08-12", "2003-09-10", "2003-11-12", "2003-12-10",
    "2004-02-11", "2004-03-10", "2004-05-12", "2004-06-09",
    "2004-08-11", "2004-09-09", "2004-11-09", "2004-12-08",
    "2005-02-09", "2005-03-09", "2005-05-11", "2005-06-08",
    "2005-08-10", "2005-09-08", "2005-11-09", "2005-12-07",
    "2006-02-08", "2006-03-08", "2006-05-10", "2006-06-08",
    "2006-08-09", "2006-09-07", "2006-11-08", "2006-12-07",
    "2007-02-07", "2007-03-07", "2007-05-09", "2007-06-07",
    "2007-08-08", "2007-09-06", "2007-11-07", "2007-12-06",
    "2008-02-06", "2008-03-05", "2008-05-07", "2008-06-05",
    "2008-08-06", "2008-09-04", "2008-11-05", "2008-12-04",
] + _monthly(2009, 1, 2026, 6, 11)

# 30y: suspended Oct-2001 -> reintroduced Feb-2006; semiannual 2006-2007,
# quarterly 2008, monthly (new + reopenings) from 2009.
_TD_30Y = [
    "2006-02-09", "2006-08-10",
    "2007-02-08", "2007-08-09",
    "2008-02-07", "2008-05-08", "2008-08-07", "2008-11-06",
] + _monthly(2009, 2, 2026, 6, 12)

# 2y: monthly end-of-month cycle for the whole sample.
_TD_2Y = _monthly(2003, 1, 2026, 5, 25)

# 5y: quarterly in early 2003, monthly end-of-month cycle from mid-2003 on.
_TD_5Y = ["2003-02-12", "2003-05-14"] + _monthly(2003, 7, 2026, 5, 26)

# 20y: reintroduced 2020-05-20, monthly third-week cycle thereafter.
_TD_20Y = ["2020-05-20"] + _monthly(2020, 6, 2026, 5, 19)

AUCTION_DATES = {
    "10y": _TD_10Y, "30y": _TD_30Y,
    "2y": _TD_2Y, "5y": _TD_5Y, "20y": _TD_20Y,
}


def _auction_calendar(tenor: str, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Recorded TreasuryDirect auction dates for one tenor, restricted to the
    price index and snapped to the next trading session on holidays."""
    out = []
    for s in AUCTION_DATES[tenor]:
        ts = pd.Timestamp(s)
        if ts < idx[0] or ts > idx[-1]:
            continue
        snapped = idx[idx.searchsorted(ts)]
        if (snapped - ts).days > 4:   # fell in a data gap, not a holiday — skip
            continue
        out.append(snapped)
    return pd.DatetimeIndex(sorted(set(out)))


# -------------------------------------------------------------------- data ----

def load_data() -> pd.DataFrame:
    """Search panel: 10y/30y duration instruments (ETF + continuous future)."""
    px = yf_panel(SEARCH_TICKERS, start=START)
    return px[[c for c in SEARCH_TICKERS if c in px.columns]]


def load_gen_data(label: str) -> pd.DataFrame:
    """Generalization panel for one untouched tenor (disjoint tickers)."""
    tickers = GEN_UNIVERSES[label]
    px = yf_panel(tickers, start=START)
    return px[[c for c in tickers if c in px.columns]]


# ------------------------------------------------------------------ signal ----

def signal(panel, hold_days=3, target_vol=0.10, vol_lb=63, max_lev=1.0,
           cost_bps=8.0):
    """Flat by default. At each recorded auction-day close, go long the
    matching-tenor instrument(s); hold exactly `hold_days` trading days
    (overlapping events extend the hold). Vol-targeted, hard-capped at 1x
    gross (no leverage). Weights are built same-day and shift(1)-lagged
    before P&L — the lag is applied here, explicitly."""
    px = panel.sort_index().ffill()
    rets = px.pct_change()
    idx = px.index

    # 1/0 event-window mask per instrument
    raw = pd.DataFrame(0.0, index=idx, columns=px.columns)
    for tk in px.columns:
        for d in _auction_calendar(TENOR[tk], idx):
            i = idx.searchsorted(d)
            raw.iloc[i: i + hold_days, raw.columns.get_loc(tk)] = 1.0

    # inverse-vol sizing toward target_vol, per-name leverage cap (trailing
    # vol through day t sizes the position applied — post shift — on t+1)
    vol = rets.rolling(vol_lb).std() * np.sqrt(252.0)
    unit = (target_vol / vol).clip(upper=max_lev)
    W = (raw * unit).fillna(0.0)            # NaN vol (pre-inception) -> flat

    # total book gross capped at 1.0 (splits a tenor event across ETF + future)
    gross = W.abs().sum(axis=1)
    W = W.div(gross.where(gross > 1.0, 1.0), axis=0)

    W_lag = W.shift(1).fillna(0.0)          # execution lag: earn returns from t+1
    daily = net_of_cost(W_lag, rets, cost_bps=cost_bps,
                        name="tsy_auction_supply_concession")
    trades = trades_from_weights(W_lag, rets,
                                 {t: SECTOR[t] for t in px.columns})
    return daily, trades


# -------------------------------------------------------------------- spec ----

SPEC = StrategySpec(
    id="tsy_auction_supply_concession_v1",
    family="event_liquidity_provision",
    title=("Treasury auction supply-concession premium — long 10y/30y duration "
           "through the 3-day post-auction inventory-absorption window"),
    markets=["US_rates"],
    data_desc=("yfinance Close panels: ZN=F/ZB=F continuous Treasury futures + "
               "IEF/TLT ETFs (search book); SHY/IEI/TLH (generalization tenors). "
               "Auction calendar: TreasuryDirect auction record 2003+ (all 10y/30y "
               "coupon auctions incl. reopenings, plus 2y/5y/20y for "
               "generalization), frozen into the module as a hardcoded constant — "
               "the harness forbids runtime fetches; irregular pre-2009 era "
               "transcribed date-by-date, documented monthly eras stored as their "
               "cadence. No synthetic events; zero fitted parameters."),
    pre_registration=(
        "FROZEN: at each recorded 10y/30y coupon-auction day close (calendar = "
        "TreasuryDirect auction record 2003+, incl. reopenings, embedded as a "
        "frozen constant: 10y quarterly-new + single reopenings ~8/yr through "
        "2008 then monthly; 30y suspended until Feb-2006, semiannual 2006-07, "
        "quarterly 2008, monthly from 2009), enter long the matching-tenor "
        "instruments (10y -> IEF+ZN=F, 30y -> TLT+ZB=F, event split across ETF "
        "and future) and hold exactly 3 trading days; overlapping events extend "
        "the hold; flat otherwise. Sized to 10% annualized vol on active days, "
        "per-name leverage <=1x, total gross <=1.0, 8bps on turnover, weights "
        "shift(1)-lagged. PRIMARY: daily-return Sharpe of this event-window book "
        "through the full gate battery; MCPT null = random 3-day long-duration "
        "windows matched on event count, isolating AUCTION timing from generic "
        "long-duration drift. SECONDARY (diagnostic only, NOT traded, NOT "
        "promotable alone): the pre-auction concession leg (t-3 to auction "
        "close). GENERALIZATION (dealer-inventory mechanism must appear across "
        "tenors): frozen signal + default params on untouched 2y (SHY), 5y "
        "(IEI), 20y (TLH) recorded auction cycles, holdout only; a single-tenor "
        "pass with negative OOS elsewhere = overfit, kill it. Distinct from the "
        "pre-FOMC duration FAILs (announcement-uncertainty premium, hold INTO "
        "information events) — this harvests a supply/flow premium AFTER a "
        "deterministic flow event, structurally akin to the validated "
        "liquidity-provision family."),
    load_data=load_data,
    signal=signal,
    default_params={"hold_days": 3, "target_vol": 0.10, "vol_lb": 63,
                    "max_lev": 1.0, "cost_bps": 8.0},
    grid={
        "default": {},
        "hold2": {"hold_days": 2},
        "hold5": {"hold_days": 5},
        "vol8": {"target_vol": 0.08},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=4,
)