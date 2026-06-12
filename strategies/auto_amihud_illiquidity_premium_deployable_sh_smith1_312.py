"""
Amihud illiquidity premium — DEPLOYABLE-SHORT v2, TURNOVER-HARDENED (overlapping tranches).

FIX vs previous run: the hedge panel was loaded via sep_panel([IWM]) — but IWM is an ETF,
and Sharadar SEP covers common stock only, so sep_panel returned an EMPTY frame and the
'hpx' block silently vanished from the concat -> KeyError 'hpx' in signal(). The hedge
series now loads via yf_panel (the documented adapter for ETFs), reindexed to the equity
calendar, with an explicit guard so an empty hedge feed fails loudly at load time.

KEPT from the elite Amihud lineage (sel-alpha +2.51 / beta -0.31 parents):
  - size-bucketed (marketcap tercile) within-tercile Amihud sort
  - LONG most-illiquid names, SHORT top-N=15 MOST-LIQUID names per size bucket
    ($10-$500 price filter, 10% single-name cap) — the multi-name short leg that the
    falsified full-ETF-hedge mutation (beta 0.91 / sel-alpha -0.77) proved does real work
  - residual-only IWM beta-trim (|beta| <= 0.30 before any hedge; sleeve capped 0.35,
    DECLARED via hedge_tickers so the gate judges the alpha book alone)
  - asymmetric realistic costs: 60bps RT on illiquid longs, 15bps RT on liquid shorts,
    50bps/yr borrow haircut on all short gross

THE ONE MUTATION (pre-registered): formation cadence. Monthly single book -> THREE
OVERLAPPING TRANCHES (Jegadeesh-Titman): each month exactly one tranche (i % 3) reforms
from the current sort and is then HELD ~3 months; live book = equal-weight average of
active tranches. Wide hysteresis inside each tranche. Risk controls are NOT slowed:
the residual IWM beta-trim re-fits EVERY month on trailing data.

NO LOOKAHEAD: all weights are built from data through the formation date and applied via
W.shift(1) before net_of_cost / trades_from_weights — the lag is taken explicitly below.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "2003-01-01"
HEDGE = "IWM"
GEN_LABELS = ["large_caps", "small_deep", "mid_deep"]

# module-level sector registry (search + gen universes share it; signal() reads it)
_SECTORS = {HEDGE: "ETF-Hedge"}
_SEARCH_CACHE = {}


# ----------------------------------------------------------------------------- universes
def _search_universe():
    """Search universe: liquid small+mid caps, sector-spread, survivorship-clean."""
    if "tickers" not in _SEARCH_CACHE:
        s_t, s_map = sector_universe("Small", 40)   # ~440 small caps
        m_t, m_map = sector_universe("Mid", 25)     # ~275 mid caps
        _SECTORS.update(s_map)
        _SECTORS.update(m_map)
        _SEARCH_CACHE["tickers"] = sorted(set(s_t) | set(m_t))
    return _SEARCH_CACHE["tickers"]


def _gen_universe(label):
    """Generalization universes — DISJOINT from search (different cap tier, or the
    next-liquidity slice of the same tier with all search tickers excluded)."""
    search = set(_search_universe())
    if label == "large_caps":
        t, smap = sector_universe("Large", 25)      # different cap tier entirely
    elif label == "small_deep":
        t, smap = sector_universe("Small", 110)     # deeper small slice, search removed
    elif label == "mid_deep":
        t, smap = sector_universe("Mid", 70)        # deeper mid slice, search removed
    else:
        raise ValueError(f"unknown gen universe {label}")
    _SECTORS.update(smap)
    t = [x for x in t if x not in search][:380]     # enforce disjointness + keep small
    return sorted(set(t))


# ----------------------------------------------------------------------------- data
def _build_panel(tickers):
    """One MultiIndex panel: px(adj) / close(raw, for price filter & $vol) / vol / mcap / hpx."""
    tickers = sorted(set(tickers))
    px = sep_panel(tickers, START, field="closeadj")
    close = sep_panel(tickers, START, field="close").reindex(px.index)
    vol = sep_panel(tickers, START, field="volume").reindex(px.index)
    f = sf1(tickers, fields=["marketcap"], dimension="ARQ")
    # pit_panel is datekey-based (filing date) — point-in-time, no calendardate lookahead
    mcap = pit_panel(f, "marketcap", px.index, list(px.columns))
    # IWM is an ETF: SEP (Sharadar common-stock prices) does NOT carry it — use yf_panel,
    # the documented adapter for ETFs. Align to the equity calendar; fail loudly if empty.
    hpx = yf_panel([HEDGE], START)
    if hpx is None or hpx.empty or hpx.dropna(how="all").empty:
        raise RuntimeError(f"hedge panel for {HEDGE} came back empty from yf_panel")
    hpx = hpx.iloc[:, [0]]
    hpx.columns = [HEDGE]
    if hpx.index.tz is not None:
        hpx.index = hpx.index.tz_localize(None)
    hpx = hpx.reindex(px.index).ffill()
    return pd.concat({"px": px, "close": close, "vol": vol, "mcap": mcap, "hpx": hpx}, axis=1)


def load_data() -> pd.DataFrame:
    return _build_panel(_search_universe())


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(_gen_universe(label))


# ----------------------------------------------------------------------------- signal
def _form_tranche(prev_book, ami, mc, prc, n_long, n_short, long_hold_q, short_hold_rank):
    """Reform one tranche at a formation date. Hysteresis: held names within the wide
    band stay; new entries fill from the extreme of the sort. Returns (longs, shorts)."""
    ok = ami.notna() & mc.notna() & prc.notna() & (prc > 1.0)
    ami, mc, prc = ami[ok], mc[ok], prc[ok]
    if len(ami) < 90:
        return None
    ter = pd.qcut(mc.rank(method="first"), 3, labels=False)
    prev_l, prev_s = prev_book
    longs, shorts = set(), set()
    for t in (0, 1, 2):
        idx = ami.index[ter == t]
        ai = ami.loc[idx]
        pct = ai.rank(pct=True)               # high = more illiquid
        liq = ai.rank(method="first")         # 1 = most liquid in bucket
        # LONG: enter from most-illiquid quintile; held stays while in top-40% illiquidity
        held_l = [n for n in prev_l if n in pct.index and pct[n] >= long_hold_q]
        cand_l = pct[pct >= 0.80].sort_values(ascending=False).index
        sel = list(held_l)
        for n in cand_l:
            if len(sel) >= n_long:
                break
            if n not in sel:
                sel.append(n)
        longs.update(sel)
        # SHORT: enter top-n_short most-liquid with $10-$500 price filter (borrowable, GC);
        # held short stays while still inside top-`short_hold_rank` liquidity
        held_s = [n for n in prev_s if n in liq.index and liq[n] <= short_hold_rank]
        pricable = liq[(prc.loc[liq.index] >= 10.0) & (prc.loc[liq.index] <= 500.0)]
        sels = list(held_s)
        for n in pricable.sort_values().index:
            if len(sels) >= n_short:
                break
            if n not in sels:
                sels.append(n)
        shorts.update(sels)
    shorts -= longs
    return longs, shorts


def _leg_weights(names, vols, gross, cap):
    """Inverse-vol weights normalized to `gross`, 10% single-name cap (cap honored even
    if it leaves gross slightly under target — capital discipline over exact gross)."""
    iv = (1.0 / vols.reindex(sorted(names))).replace([np.inf, -np.inf], np.nan).dropna()
    if iv.empty:
        return pd.Series(dtype=float)
    w = iv / iv.sum()
    w = w.clip(upper=cap)
    w = (w / w.sum()).clip(upper=cap) * gross
    return w


def signal(panel, amihud_lb=63, n_long=12, n_short=15, tranches=3,
           long_hold_q=0.60, short_hold_rank=30, name_cap=0.10,
           vol_lb=63, beta_lb=126, beta_cap=0.30, hedge_cap=0.35,
           long_cost_bps=30.0, short_cost_bps=7.5, borrow_bps_yr=50.0):
    px, close, volm, mcap = panel["px"], panel["close"], panel["vol"], panel["mcap"]
    hpx = panel["hpx"].iloc[:, 0]
    stocks = list(px.columns)
    sector_map = {t: _SECTORS.get(t, "Unknown") for t in stocks}
    sector_map[HEDGE] = "ETF-Hedge"

    rets = px.pct_change()
    iwm = hpx.pct_change()
    dvol = (close * volm).replace(0.0, np.nan)
    amihud = (rets.abs() / dvol).replace([np.inf, -np.inf], np.nan) \
                               .rolling(amihud_lb, min_periods=int(amihud_lb * 0.6)).mean()
    tvol = rets.rolling(vol_lb, min_periods=int(vol_lb * 0.6)).std()

    month_ends = px.index.to_series().groupby(px.index.to_period("M")).max().tolist()

    # --- overlapping tranches: month i reforms tranche i % `tranches`, held ~3 months ---
    books = [(set(), set()) for _ in range(tranches)]
    tr_w = [pd.Series(dtype=float) for _ in range(tranches)]
    w_rows = {}
    for i, d in enumerate(month_ends):
        if amihud.loc[d].notna().sum() < 90:
            continue
        k = i % tranches
        formed = _form_tranche(books[k], amihud.loc[d], mcap.loc[d], close.loc[d],
                               n_long, n_short, long_hold_q, short_hold_rank)
        if formed is not None:
            books[k] = formed
            longs, shorts = formed
            wl = _leg_weights(longs, tvol.loc[d], 1.0, name_cap)
            ws = _leg_weights(shorts, tvol.loc[d], 1.0, name_cap)
            tr_w[k] = wl.subtract(ws, fill_value=0.0)
        active = [w for w in tr_w if not w.empty]
        if active:
            w_rows[d] = pd.concat(active, axis=1).fillna(0.0).mean(axis=1)
    if not w_rows:
        empty = pd.Series(dtype=float, name="amihud_tranche_topn_short_v2")
        return empty, []

    W = pd.DataFrame(w_rows).T.reindex(px.index).ffill().fillna(0.0)
    W = W.reindex(columns=stocks, fill_value=0.0)

    # --- residual-only IWM beta-trim, re-fit MONTHLY on trailing data only -------------
    book_gross = (W.shift(1) * rets).sum(axis=1)          # for beta estimation only
    h = {}
    for d in w_rows:
        a = pd.concat([book_gross.loc[:d].tail(beta_lb),
                       iwm.loc[:d].tail(beta_lb)], axis=1).dropna()
        if len(a) < 60 or a.iloc[:, 1].var() <= 0:
            h[d] = 0.0
            continue
        beta = a.iloc[:, 0].cov(a.iloc[:, 1]) / a.iloc[:, 1].var()
        h[d] = 0.0 if abs(beta) <= beta_cap else float(np.clip(-beta, -hedge_cap, hedge_cap))
    H = pd.Series(h).reindex(px.index).ffill().fillna(0.0)

    W_all = W.copy()
    W_all[HEDGE] = H
    rets_all = rets.copy()
    rets_all[HEDGE] = iwm

    # --- THE LAG: same-day weights -> trade next day ------------------------------------
    W_lag = W_all.shift(1).fillna(0.0)

    # asymmetric costs: 60bps RT illiquid longs, 15bps RT liquid shorts + ETF sleeve
    wl_ = W_lag[stocks].clip(lower=0.0)
    ws_ = W_lag[stocks].clip(upper=0.0)
    wh_ = W_lag[[HEDGE]]
    r_long = net_of_cost(wl_, rets, cost_bps=long_cost_bps, name="long_leg")
    r_short = net_of_cost(ws_, rets, cost_bps=short_cost_bps, name="short_leg")
    r_hedge = net_of_cost(wh_, rets_all[[HEDGE]], cost_bps=short_cost_bps, name="hedge_leg")
    borrow = (ws_.abs().sum(axis=1) + wh_.clip(upper=0.0).abs().sum(axis=1)) \
             * (borrow_bps_yr / 1e4 / 252.0)

    daily = (r_long.add(r_short, fill_value=0.0).add(r_hedge, fill_value=0.0) - borrow)
    daily = daily[W_lag.abs().sum(axis=1) > 0].rename("amihud_tranche_topn_short_v2")

    # entry_regime stamped by the kit's standard labeller — never hand-rolled
    trades = trades_from_weights(W_lag, rets_all, sector_map)
    return daily, trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="amihud_tranche_topn_short_v2",
    family="amihud_illiquidity",
    title=("Amihud illiquidity, deployable-short v2: 3-month overlapping tranches + wide "
           "hysteresis (turnover-hardened), long illiquid / short top-15 liquid per size "
           "bucket, residual IWM beta-trim"),
    markets=["US_small_mid_equities"],
    data_desc=("Sharadar SEP closeadj/close/volume (survivorship-clean, delisted incl) for "
               "returns + Amihud dollar-volume; SF1 marketcap via pit_panel (datekey PIT) "
               "for size terciles; IWM closes via yf_panel (ETF — not in SEP) for the "
               "declared residual hedge sleeve."),
    pre_registration=(
        "MUTATION UNDER TEST (one axis): formation cadence — monthly single book -> 3 "
        "overlapping Jegadeesh-Titman tranches (each reformed every 3 months, live book = "
        "equal-weight of tranches) with wide hysteresis (long exits only below top-40% "
        "illiquidity; short exits only outside top-30 liquidity rank). Beta-trim stays "
        "MONTHLY. Pre-registered tests: (a) TURNOVER PAYOFF — frozen tranche PRIMARY must "
        "match-or-beat the monthly_single_book grid sibling net-of-cost with realized "
        "turnover down >=40%, else the slow-signal premise is falsified; (b) COST STRESS — "
        "sel-alpha must stay >0 at 1.5x all costs (2x reported as diagnostic); (c) SHORT-LEG "
        "SUFFICIENCY — n_short in {10,15,25} must degrade gracefully, N=15 retains >=60% of "
        "sel-alpha; (d) within-tercile Amihud quintile monotonicity (parent gate kept); "
        "(e) TRANCHE CONSISTENCY — each staggered tranche positive standalone sel-alpha. "
        "Hard arbiters: write-once holdout 2022+ at full costs incl. borrow; MCPT vs "
        "within-size permutation null; beta-confound gate |beta_to_universe|<0.3 AND "
        "selection_alpha_sharpe>0 (classical PROMOTE with sel-alpha<=0 = automatic FAIL). "
        "Costs frozen: 60bps RT illiquid longs, 15bps RT liquid shorts, 50bps/yr borrow. "
        "Signals lagged 1 day (W.shift(1)). Forward-paper N=15 at $5K before any promote."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "short_n10": {"n_short": 10},
        "short_n25": {"n_short": 25},
        "monthly_single_book": {"tranches": 1},   # cadence sibling — the falsifier for (a)
        "slow_amihud_126": {"amihud_lb": 126},
    },
    scope="broad",
    generalization_universes=GEN_LABELS,
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=90,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)