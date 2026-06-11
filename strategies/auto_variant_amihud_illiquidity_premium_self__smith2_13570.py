"""
amihud_illiq_compressed_short_v3 — Amihud illiquidity premium with a DEPLOYABLE
COMPRESSED SHORT LEG.

Evolution of the elite Amihud family and the pre-registered follow-up to the
FALSIFIED amihud_illiq_etf_hedged_v2 (pure index hedge -> beta 0.91-1.07,
sel-alpha -0.77/-1.12). The FROZEN PRIMARY CONFIG of the elite parent is KEPT:
survivorship-clean small+mid universe built via us_universe(include_delisted=True)
across ALL sectors (no hand-picked sector subset), trailing-12m Amihud, monthly
rebalance, 60-80th pct long band with hysteresis. The SINGLE mutation: replace
the ~100-name most-liquid short quintile with a pre-registered top-N
(N=10/15/25) SECTOR- AND SIZE-TERCILE-MATCHED basket of the MOST liquid names
(dollar-volume = borrow proxy), equal-weighted, plus a SMALL residual
beta-scaled IWM short for only the leftover beta (hard cap 35% of gross —
a binding cap means N is too small to do the cross-sectional work and the
construction has failed).

FIX vs failed run: IWM/MDY are ETFs — they live in Sharadar SFP, NOT SEP, so
sep_panel(["IWM","MDY"]) returned an empty/partial panel and the column select
raised KeyError. The hedge ETF panel now comes from yf_panel (the sanctioned
source for ETFs), with a robust candidate scan and a graceful NO-HEDGE
degradation (hedge weight forced to 0) if neither candidate has data — the
core long/short book never depends on the hedge sleeve existing.

PRE-REGISTERED (frozen before any run):
  - Universe: us_universe(marketcap in {Small, Mid}, include_delisted=True,
    top_n per sector) over ALL Sharadar sectors — survivorship-clean, unchanged
    from the elite parent.
  - N in {10,15,25}, default 15; long band = 60-80th Amihud percentile,
    hysteresis +/-5pp; monthly rebalance; trailing-12m Amihud; sub-$1 and
    dollar-volume floors; trailing-60d beta vs equal-weight universe.
  - SIZE-TERCILE NEUTRALITY: short basket greedily fills the long leg's
    (sector x PIT-marketcap-tercile) weight targets; sector-only fallback when
    a joint bucket starves (liquid-quintile names skew large — documented),
    then global most-liquid fallback. Marketcap is point-in-time (sf1 datekey
    via pit_panel — never calendardate).
  - Impact gate (self-consistent): round-trip own-Amihud impact at
    $T/n_long ($T=$5K) must be <= 35bps or the name is dropped from the long leg.
  - Residual hedge cap: |IWM weight| <= 0.35 * gross (gross ~ 2.0).
  - ACCEPTANCE: |beta_to_universe| <= 0.25 AND selection_alpha_sharpe > 0.
    The beta-confound gate is the PRIMARY arbiter (read the wiki verdict, not
    the run-log tier). Compression sweep 25->15->10 must degrade gracefully.

NO LOOK-AHEAD: all signals built from trailing data only; same-day weight
matrix W is shifted by 1 day before net_of_cost / trades_from_weights (the
lag is taken explicitly below).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "2000-01-01"
HEDGE_CANDIDATES = ["IWM", "MDY"]   # ETFs -> yf_panel (NOT SEP; SEP = common stock)

# ALL Sharadar sectors — the frozen elite universe is NOT sector-filtered.
ALL_SECTORS = [
    "Technology", "Healthcare", "Industrials", "Consumer Cyclical",
    "Consumer Defensive", "Financial Services", "Energy", "Utilities",
    "Basic Materials", "Real Estate", "Communication Services",
]


# ----------------------------------------------------------------------------
# Universe + panel construction (survivorship-clean: us_universe, delisted incl)
# ----------------------------------------------------------------------------
def _universe(caps, per_sector):
    """Frozen elite construction: us_universe(include_delisted=True) per
    sector x cap tier (the sector loop only provides the sector tags the trade
    ledger needs; selection itself is us_universe's liquidity-bounded top_n)."""
    tickers, smap = [], {}
    for cap in caps:
        for sec in ALL_SECTORS:
            try:
                tk = us_universe(sector=sec, category="Domestic Common Stock",
                                 marketcap=cap, include_delisted=True,
                                 top_n=per_sector)
            except Exception:
                tk = []
            for t in tk:
                if t not in smap:
                    smap[t] = sec
                    tickers.append(t)
    return tickers, smap


def _hedge_panel(dates):
    """Hedge ETF closeadj proxy via yf_panel (ETFs are NOT in SEP). Returns a
    single-column '_HEDGE_' frame aligned to dates; all-NaN if no candidate
    has usable history (signal() then degrades gracefully to NO hedge)."""
    try:
        hpx = yf_panel(HEDGE_CANDIDATES, START)
    except Exception:
        hpx = pd.DataFrame(index=dates)
    pick = None
    for t in HEDGE_CANDIDATES:
        if t in hpx.columns and hpx[t].notna().sum() > 1000:
            pick = t
            break
    if pick is None:
        return pd.DataFrame({"_HEDGE_": np.nan}, index=dates)
    return hpx[[pick]].rename(columns={pick: "_HEDGE_"}).reindex(dates)


def _panel(tickers, smap):
    """One DataFrame with MultiIndex columns: px (closeadj), cl (raw close),
    vol (raw share volume), mc (PIT marketcap), hg (hedge ETF close).
    Sector map rides in attrs."""
    px = sep_panel(tickers, START, field="closeadj")
    cl = sep_panel(tickers, START, field="close")      # unadjusted: true $ volume
    vol = sep_panel(tickers, START, field="volume")
    # Point-in-time size (datekey-based — no calendardate lookahead)
    mc = pit_panel(sf1(list(px.columns), ["marketcap"]), "marketcap",
                   px.index, list(px.columns))
    hg = _hedge_panel(px.index)
    panel = pd.concat({"px": px, "cl": cl, "vol": vol, "mc": mc, "hg": hg}, axis=1)
    panel.attrs["sector_map"] = dict(smap)
    return panel


def _search_universe():
    # ~11 sectors x 2 caps x 35 = ~700-770 liquid small+mid names
    return _universe(("Small", "Mid"), per_sector=35)


def load_data() -> pd.DataFrame:
    tickers, smap = _search_universe()
    return _panel(tickers, smap)


def load_gen_data(label) -> pd.DataFrame:
    """Generalization universes — DISJOINT from search: a different cap tier,
    or the NEXT-liquidity slice of the same tier with all search tickers
    force-dropped (shares NO tickers by construction)."""
    if label == "gen_largecap_allsec":
        tickers, smap = _universe(("Large",), per_sector=20)
    elif label == "gen_small_deep":
        tickers, smap = _universe(("Small",), per_sector=70)
    elif label == "gen_mid_deep":
        tickers, smap = _universe(("Mid",), per_sector=70)
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    search_t, _ = _search_universe()
    search_set = set(search_t)
    tickers = [t for t in tickers if t not in search_set]
    smap = {t: smap[t] for t in tickers}
    return _panel(tickers, smap)


# ----------------------------------------------------------------------------
# Greedy sector x size-tercile matched short basket (pre-registered fill rule)
# ----------------------------------------------------------------------------
def _greedy_basket(dv_cands, sec, terc, w_st, w_s, n):
    """Pick n names from the liquid quintile: highest dollar-volume within the
    (sector, size-tercile) bucket with the largest remaining deficit vs the
    long leg's joint weights; fallback to sector-only deficit, then to the
    globally most liquid remaining name."""
    pools = {}
    for t in dv_cands.sort_values(ascending=False).index:
        pools.setdefault((sec[t], terc[t]), []).append(t)
    chosen, cnt_st, cnt_s = [], {}, {}
    while len(chosen) < n:
        best, best_d = None, 1e-9
        # 1) joint (sector, tercile) deficit
        for k, tgt in w_st.items():
            if pools.get(k):
                d = tgt * n - cnt_st.get(k, 0)
                if d > best_d:
                    best_d, best = d, k
        # 2) sector-only deficit (joint bucket starved)
        if best is None:
            for s, tgt in w_s.items():
                ks = [k for k in pools if k[0] == s and pools[k]]
                if ks:
                    d = tgt * n - cnt_s.get(s, 0)
                    if d > best_d:
                        best_d = d
                        best = max(ks, key=lambda k: dv_cands[pools[k][0]])
        # 3) global most-liquid fallback
        if best is None:
            rem = [(dv_cands[lst[0]], k) for k, lst in pools.items() if lst]
            if not rem:
                break
            best = max(rem)[1]
        t = pools[best].pop(0)
        chosen.append(t)
        cnt_st[best] = cnt_st.get(best, 0) + 1
        cnt_s[best[0]] = cnt_s.get(best[0], 0) + 1
    return chosen


# ----------------------------------------------------------------------------
# Signal
# ----------------------------------------------------------------------------
def signal(panel, n_short=15, band_lo=0.60, band_hi=0.80, hyst=0.05,
           t_dollars=5000.0, impact_cap_bps=35.0, hedge_cap_frac=0.35,
           max_long=30, dv_floor=2e5, min_price=1.0, cost_bps=8.0, **_):
    px = panel["px"]
    cl = panel["cl"]
    vol = panel["vol"]
    mc = panel["mc"]
    hg = panel["hg"]["_HEDGE_"]
    smap = panel.attrs.get("sector_map") or {t: "Other" for t in px.columns}
    hedge_ok = hg.notna().sum() > 1000   # graceful no-hedge degradation

    rets = px.pct_change(fill_method=None)
    hret = hg.pct_change(fill_method=None).fillna(0.0).rename("_HEDGE_")
    dv = cl * vol                                                   # true $ volume
    # Trailing-12m Amihud: mean(|ret| / dollar_volume) = price impact per $ traded
    amihud = (rets.abs() / dv.replace(0.0, np.nan)).rolling(252, min_periods=126).mean()
    dv63 = dv.rolling(63, min_periods=21).median()
    vol60 = rets.rolling(60, min_periods=30).std()
    mkt = rets.mean(axis=1)                                          # eq-wt universe
    beta = rets.rolling(60, min_periods=30).cov(mkt).div(
        mkt.rolling(60, min_periods=30).var(), axis=0)

    idx = px.index
    month_ends = idx.to_series().groupby([idx.year, idx.month]).max()
    warmup_cut = idx[min(265, len(idx) - 1)]
    reb_dates = [d for d in month_ends if d >= warmup_cut]

    rows = {}
    held = set()
    for d in reb_dates:
        am = amihud.loc[d]
        dvd = dv63.loc[d]
        pr = cl.loc[d]
        valid = am.index[am.notna() & (dvd > dv_floor) & (pr >= min_price)]
        if len(valid) < 60:
            continue
        pct = am[valid].rank(pct=True)   # high pct = more illiquid
        # PIT size terciles among valid names (dollar-volume pct fallback if mc NaN)
        size_pct = mc.loc[d, valid].rank(pct=True)
        size_pct = size_pct.fillna(dvd[valid].rank(pct=True))
        terc = np.ceil(size_pct * 3.0).clip(1, 3).astype(int)

        # LONG: mid-illiquid band with hysteresis on incumbents
        in_band = set(pct.index[(pct >= band_lo) & (pct <= band_hi)])
        keep = {t for t in held if t in pct.index
                and (band_lo - hyst) <= pct[t] <= (band_hi + hyst)}
        longs = list(in_band | keep)
        # Self-consistent impact-cost gate: round-trip own-impact at $T/n_long
        per_name_dollars = t_dollars / float(max(len(longs), 1))
        imp_bps = am[longs] * per_name_dollars * 2.0 * 1e4
        longs = [t for t in longs if np.isfinite(imp_bps[t]) and imp_bps[t] <= impact_cap_bps]
        if len(longs) < 10:
            continue
        if len(longs) > max_long:
            center = 0.5 * (band_lo + band_hi)
            longs = sorted(longs, key=lambda t: abs(pct[t] - center))[:max_long]
        held = set(longs)

        # Long weights: inverse trailing vol, gross long = 1.0
        iv = (1.0 / vol60.loc[d, longs]).replace([np.inf, -np.inf], np.nan)
        if iv.notna().sum() == 0:
            iv = pd.Series(1.0, index=longs)
        iv = iv.fillna(iv.median())
        wl = iv / iv.sum()

        # SHORT: compressed top-N from the most-liquid quintile, greedy
        # SECTOR x SIZE-TERCILE matched to the long leg's weights, equal-weight
        liquid = pct.index[pct <= 0.20]
        if len(liquid) < max(5, n_short // 2):
            continue
        sec_long = pd.Series({t: smap.get(t, "Other") for t in wl.index})
        terc_long = terc.reindex(wl.index).fillna(2).astype(int)
        w_st = wl.groupby([sec_long, terc_long]).sum()
        w_st = {(s, k): v for (s, k), v in w_st.items()}
        w_s = wl.groupby(sec_long).sum().to_dict()
        shorts = _greedy_basket(dvd[liquid],
                                {t: smap.get(t, "Other") for t in liquid},
                                {t: int(terc[t]) for t in liquid},
                                w_st, w_s, n_short)
        if len(shorts) < max(5, n_short // 2):
            continue
        ws = 1.0 / len(shorts)

        # Residual beta only -> small IWM short, HARD CAP 35% of gross;
        # zero if no hedge ETF data is available (core book unaffected)
        b = beta.loc[d].fillna(1.0)
        b_res = float((wl * b[wl.index]).sum() - ws * b[shorts].sum())
        gross = float(wl.sum() + ws * len(shorts))
        h = float(np.clip(-b_res, -hedge_cap_frac * gross, hedge_cap_frac * gross))
        if not hedge_ok:
            h = 0.0

        row = pd.Series(0.0, index=list(px.columns) + ["_HEDGE_"])
        row.loc[wl.index] = wl.values
        row.loc[shorts] = row.loc[shorts] - ws
        row.loc["_HEDGE_"] = h
        rows[d] = row

    if not rows:
        empty = pd.Series(dtype=float, name="amihud_compressed_short")
        return empty, []

    # Monthly weights held until next rebalance; THEN lag 1 day (no look-ahead)
    W = pd.DataFrame(rows).T.reindex(idx).ffill().fillna(0.0)
    rets_all = pd.concat([rets, hret], axis=1)
    W = W.reindex(columns=rets_all.columns).fillna(0.0)
    W_lag = W.shift(1).fillna(0.0)   # explicit lag — weights act from next day

    daily = net_of_cost(W_lag, rets_all, cost_bps=cost_bps,
                        name="amihud_compressed_short")
    smap_tr = dict(smap)
    smap_tr["_HEDGE_"] = "ETF-Hedge"
    trades = trades_from_weights(W_lag, rets_all, smap_tr)
    return daily, trades


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
SPEC = StrategySpec(
    id="amihud_illiq_compressed_short_v3",
    family="amihud_illiquidity",
    title=("Amihud illiquidity premium — compressed top-N sector/size-matched "
           "single-name short leg + residual IWM hedge ($5K deployable)"),
    markets=["US_EQ_SMALL", "US_EQ_MID"],
    data_desc=("Sharadar SEP closeadj/close/volume (survivorship-clean: "
               "us_universe include_delisted=True, all sectors, small+mid) + "
               "sf1 marketcap (PIT via datekey/pit_panel) for size terciles; "
               "IWM (MDY fallback) from yf_panel (ETFs are not in SEP) for the "
               "residual hedge sleeve, degrading to no-hedge if unavailable. "
               "Borrowability proxied by dollar-volume rank within the liquid "
               "quintile. $0 / owned."),
    pre_registration=(
        "Frozen before any run: elite-parent universe KEPT — "
        "us_universe(marketcap Small+Mid, include_delisted=True, top_n per "
        "sector) over ALL sectors, survivorship-clean. N in {10,15,25} "
        "(default 15); long band 60-80th Amihud pct with +/-5pp hysteresis; "
        "monthly rebalance; trailing-12m Amihud; impact gate: round-trip "
        "own-Amihud impact at $5K/n_long <= 35bps; short basket greedily fills "
        "the long leg's (sector x PIT-marketcap-tercile) weights, sector-only "
        "then most-liquid fallback; residual IWM hedge hard cap 35% of gross "
        "(binding cap = construction failure -> expect beta-gate demotion). "
        "ACCEPTANCE is NOT classical metrics: |beta_to_universe| <= 0.25 AND "
        "selection_alpha_sharpe > 0 (the beta-confound gate that demoted the "
        "falsified pure-ETF mutation is the PRIMARY arbiter; read the wiki "
        "verdict, not the run-log tier). Compression sweep 25->15->10 must "
        "keep sel-alpha positive and degrade gracefully — a cliff/beta blow-up "
        "at small N falsifies the deployable variant (acceptable, informative). "
        "Standalone test; trend tail-overlay deferred."),
    load_data=load_data,
    signal=signal,
    default_params={"n_short": 15, "band_lo": 0.60, "band_hi": 0.80,
                    "t_dollars": 5000.0, "impact_cap_bps": 35.0,
                    "hedge_cap_frac": 0.35},
    grid={
        "default": {},
        "n10": {"n_short": 10},
        "n25": {"n_short": 25},
        "band_55_75": {"band_lo": 0.55, "band_hi": 0.75},
    },
    scope="broad",
    generalization_universes=["gen_largecap_allsec", "gen_small_deep",
                              "gen_mid_deep"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=40,
)