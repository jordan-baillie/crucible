"""
Amihud illiquidity premium — DEPLOYABLE-SHORT, LOW-TURNOVER OVERLAPPING-COHORT variant.

Direct evolution of the live elite Amihud deployable-short strategy. Everything proven is
KEPT: within-size-tercile Amihud sort, top-N most-liquid short leg per size bucket
($10-$500 price filter), small residual IWM beta-trim (|beta|<0.3 trim only, declared
hedge sleeve), realistic costs incl. borrow. The ONLY mutation is the portfolio formation
schedule: the single monthly full-rebalance is replaced by 3 Jegadeesh-Titman overlapping
monthly cohorts, each held 3 months. The traded book is the equal-weight union (mean) of
the live cohorts, netted before execution — per-month turnover on the expensive illiquid
long leg drops ~2/3 mechanically, and results no longer hinge on one formation date.

FIX vs previous run: IWM is an ETF and is NOT in Sharadar SEP (equities only) — the
sep_panel(["IWM"]) call returned an EMPTY frame and .iloc[:, 0] raised IndexError.
The hedge price now comes from yf_panel (the sanctioned source for ETFs), with a
defensive fallback to an all-NaN column (which simply disables the trim — beta hedge
is an overlay, never a dependency).

Lookahead discipline: signals use trailing data only; weights are formed at each
formation close and applied via W.shift(1) into net_of_cost / trades_from_weights.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2005-01-01"
NAME = "amihud_cohort_overlap_v1"

# Accumulating ticker -> sector map, populated by the loaders (disjoint universes, no clashes).
_SECTORS = {"IWM": "ETF-Hedge"}


# ----------------------------------------------------------------------------- universes

def _search_universe():
    """Small + Mid cap, sector-spread, survivorship-clean (delisted included). ~1500 names."""
    t_s, m_s = sector_universe(marketcap="Small", top_n_per_sector=80)
    t_m, m_m = sector_universe(marketcap="Mid", top_n_per_sector=60)
    smap = {**m_s, **m_m}
    tickers = sorted(set(t_s) | set(t_m))
    return tickers, smap


def _build_panel(tickers, smap):
    """MultiIndex-column panel: (field, ticker) for closeadj/close/volume + ('hedge','IWM')."""
    _SECTORS.update(smap)
    parts = {}
    for f in ("closeadj", "close", "volume"):
        parts[f] = sep_panel(tickers, START, field=f)
    panel = pd.concat(parts, axis=1)  # columns: (field, ticker)
    # IWM is an ETF -> yf_panel, NOT sep_panel (SEP is equities-only; empty frame before).
    hedge_col = pd.Series(np.nan, index=panel.index)
    try:
        iwm = yf_panel(["IWM"], START)
        if iwm is not None and getattr(iwm, "shape", (0, 0))[1] > 0:
            hedge_col = iwm.iloc[:, 0].reindex(panel.index)
    except Exception:
        pass  # hedge is an optional overlay; missing data just disables the trim
    panel[("hedge", "IWM")] = hedge_col
    return panel


def load_data() -> pd.DataFrame:
    tickers, smap = _search_universe()
    return _build_panel(tickers, smap)


def load_gen_data(label) -> pd.DataFrame:
    """Three universes DISJOINT from the search universe (different cap tier or tail slice)."""
    search, _ = _search_universe()
    search = set(search)
    if label == "large":
        t, m = sector_universe(marketcap="Large", top_n_per_sector=30)       # ~330 names
    elif label == "micro":
        t, m = sector_universe(marketcap="Micro", top_n_per_sector=35)       # ~385 names
    elif label == "small_tail":
        # Small-cap names BEYOND the search slice (ranks 81+ per sector) -> zero overlap.
        t, m = sector_universe(marketcap="Small", top_n_per_sector=200)
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    t = [x for x in t if x not in search and x != "IWM"][:400]
    m = {k: v for k, v in m.items() if k in set(t)}
    return _build_panel(t, m)


# ----------------------------------------------------------------------------- signal

def _form_cohort(fd, amihud, close, med_dvol, n_short, min_price, max_price, min_dvol):
    """One cohort's weight vector at formation date fd. Long total +1, short total -1."""
    a = amihud.loc[fd]
    px = close.loc[fd]
    mdv = med_dvol.loc[fd]
    elig = a.notna() & mdv.notna() & px.between(min_price, max_price) & (mdv >= min_dvol)
    names = a.index[elig]
    if len(names) < 90:
        return None
    # Size proxy: trailing median dollar volume terciles (controls the size confound).
    tercile = pd.qcut(mdv[names].rank(method="first"), 3, labels=False)
    w = pd.Series(0.0, index=names)
    for t in range(3):
        g = names[tercile == t]
        if len(g) < 30:
            return None
        # LONG: most-illiquid quintile within the tercile, equal weight (1/3 of long book).
        n_long = max(int(round(len(g) * 0.20)), 5)
        longs = a[g].nlargest(n_long).index
        w[longs] += (1.0 / 3.0) / len(longs)
        # SHORT: top-N most-liquid (highest dollar volume) within the tercile — the least
        # squeeze-prone, most borrowable short book possible (1/3 of short book).
        ns = min(n_short, len(g) - n_long)
        shorts = mdv[g].nlargest(ns).index
        w[shorts] -= (1.0 / 3.0) / ns
    return w


def signal(panel, n_cohorts=3, hold_months=3, form_lag_days=0, n_short=15,
           amihud_lb=63, min_price=10.0, max_price=500.0, min_dvol=2e5,
           beta_band=0.30, hedge_cap=0.35, cost_bps=30.0, borrow_bps=50.0):
    """
    Overlapping-cohort Amihud long/short book.
      - Every month ONE cohort re-forms (at the month-end close, shifted earlier by
        form_lag_days trading days for the pre-registered formation-date-invariance test)
        and is held hold_months months. Traded book = mean of live cohort weights.
      - Residual IWM beta-trim recomputed monthly on the AGGREGATE book: hedge only the
        beta beyond +/-beta_band, capped at hedge_cap (declared sleeve on the spec).
      - Costs: cost_bps per side on turnover (30bps ~ 60bps RT on the illiquid long leg,
        conservative for the liquid short leg) + borrow_bps/yr on short notional.
    """
    adj = panel["closeadj"]
    close = panel["close"]
    vol = panel["volume"]
    hedge_px = panel["hedge"]["IWM"]
    dates = adj.index
    stock_cols = list(adj.columns)

    rets = adj.pct_change()
    hedge_ret = hedge_px.pct_change()
    dvol = (close * vol).replace(0.0, np.nan)
    amihud = (rets.abs() / dvol).rolling(amihud_lb, min_periods=40).mean()
    med_dvol = dvol.rolling(amihud_lb, min_periods=40).median()

    # Formation dates: last trading day of each month, shifted back form_lag_days days.
    ds = dates.to_series()
    month_ends = ds.groupby([dates.year, dates.month]).max().sort_values()
    form_dates = []
    for me in month_ends:
        loc = dates.get_loc(me) - int(form_lag_days)
        if loc >= amihud_lb + 5:
            form_dates.append(dates[loc])

    rows = {}
    live = []  # (expiry_formation_index, weight Series)
    for i, fd in enumerate(form_dates):
        live = [(e, w) for (e, w) in live if e > i]
        w_new = _form_cohort(fd, amihud, close, med_dvol,
                             n_short, min_price, max_price, min_dvol)
        if w_new is not None:
            live.append((i + hold_months, w_new))
            live = live[-n_cohorts:]
        if not live:
            continue
        # Equal-weight union of live cohorts, netted into ONE book.
        agg = pd.concat([w for (_, w) in live], axis=1).fillna(0.0).mean(axis=1)
        agg = agg.reindex(stock_cols).fillna(0.0)

        # Residual beta trim on the aggregate book (trailing data only).
        hedge_w = 0.0
        hist = rets.loc[:fd].tail(126)
        hr = hedge_ret.reindex(hist.index)
        ok = hr.notna()
        if ok.sum() >= 60:
            pr = hist.fillna(0.0).mul(agg, axis=1).sum(axis=1)
            b = np.cov(pr[ok], hr[ok])[0, 1] / (hr[ok].var() + 1e-12)
            if abs(b) > beta_band:
                hedge_w = -np.sign(b) * min(abs(b) - beta_band, hedge_cap)

        row = agg.copy()
        row["IWM"] = hedge_w
        rows[fd] = row

    cols = stock_cols + ["IWM"]
    W = pd.DataFrame(rows).T.reindex(columns=cols).reindex(dates).ffill().fillna(0.0)
    rets_full = rets.copy()
    rets_full["IWM"] = hedge_ret

    W_lag = W.shift(1)  # the lag — weights formed at close t trade/earn from t+1
    daily = net_of_cost(W_lag, rets_full, cost_bps=cost_bps, name=NAME)
    short_notional = W_lag.clip(upper=0.0).abs().sum(axis=1)
    daily = daily - short_notional * (borrow_bps / 1e4) / 252.0
    daily.name = NAME

    smap = {t: _SECTORS.get(t, "Unknown") for t in cols}
    trades = trades_from_weights(W_lag, rets_full, smap)

    # Trim warmup before first held position.
    held = W_lag.abs().sum(axis=1) > 0
    if held.any():
        daily = daily.loc[held.idxmax():]
    return daily, trades


# ----------------------------------------------------------------------------- spec

SPEC = StrategySpec(
    id=NAME,
    family="illiquidity-premium",
    title=("Amihud illiquidity premium — overlapping 3-cohort (3-month hold) low-turnover "
           "deployable-short variant, long illiquid quintile / short top-N liquid per size "
           "tercile, residual IWM beta-trim"),
    markets=["US_smallmid_equities"],
    data_desc=("Sharadar SEP close/closeadj/volume (survivorship-clean, delisted incl.), "
               "Small+Mid sector-spread universe ~1500 names; IWM Close via yf_panel for "
               "the small residual hedge sleeve only. All OWNED/FREE."),
    pre_registration=(
        "PREMIUM: illiquidity (limits-to-arbitrage risk premium, not a forecast). MUTATION "
        "under test: replace single monthly rebalance with 3 overlapping monthly cohorts "
        "held 3 months (equal-weight union, netted) to cut illiquid-long-leg turnover ~2/3 "
        "and remove single-formation-date luck. FROZEN PRIMARY: 3 cohorts x 3-month hold, "
        "formation at month-end (form_lag_days=0), n_short=15/tercile, $10-$500 price "
        "filter, 60bps RT long / borrow 50bps/yr, |beta|<0.3 IWM trim only (declared "
        "sleeve, cap 0.35). PRE-REGISTERED TESTS: (a) formation-date invariance — "
        "selection_alpha_sharpe positive at form_lag_days {0,7,14} with cross-offset spread "
        "<50% of mean, else FAIL; (b) realized turnover must drop >=40% vs the parent "
        "monthly-rebalance construction while retaining >=80% of parent net "
        "selection_alpha_sharpe; (c) Amihud-quintile monotonicity within both size "
        "terciles; (d) long and short legs each span >=4 sectors on average. HARD GATES "
        "unchanged: write-once holdout 2022+, MCPT within-size permutation null, "
        "beta-confound (|beta_to_universe|<0.3 AND selection_alpha_sharpe>0 — classical "
        "PROMOTE with sel-alpha<=0 is an automatic FAIL). Standalone first; trend overlay "
        "(<=25%) considered only after a holdout pass and only if it cuts the crisis tail."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "form_off1": {"form_lag_days": 7},    # pre-registered formation-date invariance
        "form_off2": {"form_lag_days": 14},   # pre-registered formation-date invariance
        "nshort20": {"n_short": 20},          # short-leg breadth sensitivity
    },
    scope="broad",
    generalization_universes=["large", "micro", "small_tail"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=60,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)