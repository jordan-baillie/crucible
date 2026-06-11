"""
Amihud illiquidity premium — VARIANT v3 (FIXED): self-consistent impact-cost
gate + DEPLOYABLE multi-name short leg (top-N borrowable liquid-quintile
basket, sector-balanced) with only the RESIDUAL beta trimmed via a small IWM
sleeve.

FIX vs failed run: IWM is an ETF — it is NOT in Sharadar SEP (common stocks),
so sep_panel returned no 'IWM' column -> KeyError. The hedge price now comes
from yf_panel (free, correct source for ETFs) and is joined onto the SEP
panel; the signal also degrades gracefully (hedge weight = 0) if the hedge
series is unavailable, instead of crashing.

Lesson encoded from the falsified sibling (one-line ETF short replacement →
beta 0.91/1.07, sel-alpha -0.77/-1.12, demoted): the multi-name short leg does
real selection work. Here we COMPRESS it (N in {15,25}, dollar-volume floor
$20M/day so every short is borrowable at retail) instead of replacing it.
|beta_to_universe| < 0.3 and selection_alpha_sharpe > 0 are PRE-REGISTERED
in-search gates (see pre_registration), not post-hoc demotion criteria.

All weights are built same-day and lagged via W.shift(1) before net_of_cost —
the 1-day execution lag is applied explicitly below. Monthly rebalance with
hysteresis (pre-registered design for this family), 8bps cost on turnover.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "2000-01-01"
HEDGE_TICKER = "IWM"

DEFAULTS = dict(
    band_lo=0.60, band_hi=0.80,     # long band: mid-illiquid pct within tercile x sector
    hys_lo=0.55, hys_hi=0.85,       # hysteresis exit band (monthly, cuts churn)
    n_short=25,                     # pre-registered N in {15, 25}
    short_dv_floor=2.0e7,           # $20M/day median dollar volume -> borrowable shorts
    long_dv_floor=5.0e5,            # $500k/day floor on the long leg
    price_floor=1.0,                # sub-$1 names excluded
    trade_T=25000.0,                # pre-registered per-name capacity trade size ($)
    prem_bps_month=40.0,            # pre-registered expected monthly premium for cost gate
    hedge_cap=0.30,                 # IWM residual trim hard cap (<30% of book)
    gross_cap=2.0,                  # pre-registered gross leverage ceiling
    amihud_lb=252, amihud_min=126,  # trailing-12m Amihud
    vol_lb=63,                      # inverse-vol sizing lookback
    beta_lb=60,                     # trailing residual-beta window for the IWM trim
)

_CACHE = {}


# ---------------------------------------------------------------- universes
def _tiers_universe(tiers, top_n_per_sector, exclude=(), per_sector_cap=None):
    """Sector-spread universe across cap tiers; optional exclusion + per-sector cap."""
    excl = set(exclude)
    tickers, smap = [], {}
    for tier in tiers:
        tk, m = sector_universe(marketcap=tier, top_n_per_sector=top_n_per_sector)
        per_sec = {}
        for t in tk:
            if t in excl or t in smap:
                continue
            s = m.get(t, "Unknown")
            if per_sector_cap is not None and per_sec.get(s, 0) >= per_sector_cap:
                continue
            per_sec[s] = per_sec.get(s, 0) + 1
            tickers.append(t)
            smap[t] = s
    return tickers, smap


def _search_universe():
    # Small + Mid, survivorship-clean (sector_universe -> us_universe w/ delisted),
    # ~50/sector/tier -> ~1000 names total: bounded, rails-safe.
    if "search" not in _CACHE:
        _CACHE["search"] = _tiers_universe(("Small", "Mid"), 50)
    return _CACHE["search"]


def _hedge_close(index):
    """IWM close from yfinance (ETF — NOT in Sharadar SEP). Past-only ffill onto index."""
    try:
        h = yf_panel([HEDGE_TICKER], START)
        s = h[HEDGE_TICKER] if HEDGE_TICKER in h.columns else h.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").reindex(index).ffill()
    except Exception:
        return pd.Series(np.nan, index=index)


def _build_panel(tickers, smap):
    tickers = list(tickers)
    closeadj = sep_panel(tickers, START, field="closeadj")
    close = sep_panel(tickers, START, field="close")     # RAW close for dollar volume
    volume = sep_panel(tickers, START, field="volume")

    # ETF hedge price from yfinance (Close, adjusted) joined onto the SEP grid.
    hedge = _hedge_close(closeadj.index)
    closeadj = closeadj.copy()
    close = close.copy()
    volume = volume.copy()
    closeadj[HEDGE_TICKER] = hedge
    close[HEDGE_TICKER] = hedge
    volume[HEDGE_TICKER] = np.nan                        # never used in dv (names only)

    fund = sf1(tickers, fields=["marketcap"], dimension="ARQ")
    mcap = pit_panel(fund, "marketcap", closeadj.index, tickers)     # datekey PIT
    mcap = mcap.reindex(columns=closeadj.columns)                    # IWM -> NaN, fine
    panel = pd.concat(
        {"closeadj": closeadj, "close": close, "volume": volume, "mcap": mcap}, axis=1
    )
    panel.attrs["sector_map"] = dict(smap)
    return panel


def load_data():
    tk, sm = _search_universe()
    return _build_panel(tk, sm)


# Stage-2 generalization universes: DISJOINT from search by construction
# (next-liquidity-tier slices exclude search tickers; Large is a different cap tier).
GEN_SPECS = {
    "small_next_tier": dict(tiers=("Small",), top_n_per_sector=110, cap=30),
    "mid_next_tier": dict(tiers=("Mid",), top_n_per_sector=110, cap=30),
    "large_cap": dict(tiers=("Large",), top_n_per_sector=35, cap=30),
}


def load_gen_data(label):
    g = GEN_SPECS[label]
    search_tk, _ = _search_universe()
    tk, sm = _tiers_universe(
        g["tiers"], g["top_n_per_sector"], exclude=set(search_tk), per_sector_cap=g["cap"]
    )
    return _build_panel(tk, sm)


# ------------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)
    sector_map = dict(panel.attrs.get("sector_map", {}))

    closeadj = panel["closeadj"]
    close = panel["close"]
    volume = panel["volume"]
    mcap = panel["mcap"]
    names = [c for c in closeadj.columns if c != HEDGE_TICKER]

    # Hedge availability guard (graceful degrade -> unhedged residual, no crash).
    have_hedge = (
        HEDGE_TICKER in closeadj.columns
        and closeadj[HEDGE_TICKER].notna().sum() > 252
    )

    rets_all = closeadj.pct_change(fill_method=None)
    rets = rets_all[names]
    dv = (close[names] * volume[names])
    dv = dv.where(dv > 0)                                            # zero/NaN $vol masked
    amihud = (rets.abs() / dv).rolling(p["amihud_lb"], min_periods=p["amihud_min"]).mean()
    dv_med = dv.rolling(63, min_periods=21).median()
    vol63 = rets.rolling(p["vol_lb"], min_periods=21).std()
    mkt_ew = rets.mean(axis=1)                                       # universe EW return

    idx = rets.index
    mo = pd.Series(idx.to_period("M"), index=idx)
    is_rebal = mo.ne(mo.shift(1))
    rebal_dates = idx[is_rebal.values & (idx >= idx[0] + pd.Timedelta(days=400))]

    prem_frac = p["prem_bps_month"] / 1e4
    n_short = int(p["n_short"])
    cols = names + ([HEDGE_TICKER] if have_hedge else [])
    rows, prev_long = {}, set()

    for dt in rebal_dates:
        am = amihud.loc[dt]
        dvm = dv_med.loc[dt]
        px = close.loc[dt, names]
        mc = mcap.loc[dt, names]
        vv = vol63.loc[dt]

        elig = am.notna() & dvm.notna() & mc.notna() & (px > p["price_floor"]) & (
            dvm > p["long_dv_floor"]
        )
        e = am.index[elig]
        if len(e) < 60:
            continue

        sec = pd.Series({t: sector_map.get(t, "Unknown") for t in e})
        ter = pd.qcut(mc[e].rank(method="first"), 3, labels=False)
        sub = pd.DataFrame({"am": am[e], "sec": sec, "ter": ter})

        # illiquidity percentile within size-tercile x sector; tercile fallback if thin
        pct = pd.Series(np.nan, index=e)
        for (_, _), g in sub.groupby(["ter", "sec"]):
            if len(g) >= 6:
                pct[g.index] = g["am"].rank(pct=True)
        for _, g in sub.groupby("ter"):
            miss = g.index[pct[g.index].isna()]
            if len(miss):
                pct[miss] = g["am"].rank(pct=True)[miss]

        # self-consistent impact-cost gate: own Amihud x $T round trip vs premium
        gate = (2.0 * am[e] * p["trade_T"]) < prem_frac
        in_band = pct.between(p["band_lo"], p["band_hi"])
        keep = pct.index.isin(prev_long) & pct.between(p["hys_lo"], p["hys_hi"])
        longs = pct.index[(in_band | keep) & gate]
        if len(longs) < 10:
            continue

        iv = 1.0 / vv[longs].replace(0, np.nan)
        iv = iv.fillna(iv.median())
        w_long = (iv / iv.sum()).clip(upper=0.05)
        w_long = w_long / w_long.sum()                               # gross long = 1.0

        # SHORT leg: top-N most-liquid borrowable names, sector-balanced to long leg
        am_all = am.dropna()
        pct_all = am_all.rank(pct=True)
        pool_mask = (
            (pct_all < 0.20)
            & (dvm.reindex(am_all.index) > p["short_dv_floor"])
            & (close.loc[dt].reindex(am_all.index) > p["price_floor"])
            & ~am_all.index.isin(longs)
        )
        pool = list(am_all.index[pool_mask])
        sec_share = w_long.groupby(sec.loc[longs]).sum()
        pool_by_sec = {}
        for t in pool:
            pool_by_sec.setdefault(sector_map.get(t, "Unknown"), []).append(t)
        shorts = []
        for s_, share in sec_share.sort_values(ascending=False).items():
            k = int(round(share * n_short))
            cand = sorted(pool_by_sec.get(s_, []), key=lambda t: am_all[t])
            shorts += [t for t in cand[:k] if t not in shorts]
        if len(shorts) < n_short:                                    # global liquid top-up
            rest = sorted((t for t in pool if t not in shorts), key=lambda t: am_all[t])
            shorts += rest[: n_short - len(shorts)]
        shorts = sorted(set(shorts), key=lambda t: am_all[t])[:n_short]
        if len(shorts) < max(5, n_short // 2):
            continue

        w = pd.Series(0.0, index=cols)
        w[w_long.index] = w_long.values
        w[shorts] = -1.0 / len(shorts)                               # gross short = 1.0

        # residual-ONLY IWM trim: trailing-60d beta of the candidate book vs EW universe
        if have_hedge:
            hist = rets.loc[:dt].tail(p["beta_lb"])
            book = (hist[w_long.index] * w_long).sum(axis=1) - hist[shorts].mean(axis=1)
            mh = mkt_ew.loc[:dt].tail(p["beta_lb"])
            var = mh.var()
            beta = float(book.cov(mh) / var) if var and var > 0 else 0.0
            w[HEDGE_TICKER] = float(np.clip(-beta, -p["hedge_cap"], p["hedge_cap"]))

        gross = w.abs().sum()
        if gross > p["gross_cap"]:
            w *= p["gross_cap"] / gross

        rows[dt] = w
        prev_long = set(longs)

    if not rows:
        empty = pd.Series(dtype=float, name="amihud_illiq_topN_short_v3")
        return empty, []

    W = pd.DataFrame(rows).T.reindex(idx).ffill().fillna(0.0)[cols]
    R = rets_all[cols]
    Wlag = W.shift(1).fillna(0.0)                                    # EXPLICIT 1-day exec lag

    daily = net_of_cost(Wlag, R, cost_bps=8.0, name="amihud_illiq_topN_short_v3")
    sec_full = dict(sector_map)
    sec_full[HEDGE_TICKER] = "ETF-Hedge"
    trades = trades_from_weights(Wlag, R, sec_full)
    return daily, trades


# --------------------------------------------------------------------- SPEC
SPEC = StrategySpec(
    id="amihud_illiq_topN_borrowable_short_v3",
    family="amihud_illiquidity",
    title=("Amihud illiquidity premium: self-consistent impact-cost gate + top-N "
           "borrowable liquid-quintile short basket, residual-only IWM trim"),
    markets=["US small/mid-cap equities (Sharadar SEP, survivorship-clean)"],
    data_desc=("Sharadar SEP close/closeadj/volume (delisted included) + SF1 marketcap "
               "(datekey PIT) for size terciles; IWM hedge from yfinance (ETF — not in "
               "SEP), residual trim only"),
    pre_registration=(
        "Illiquidity (Amihud 2002) is a limits-to-arbitrage RISK PREMIUM, not a forecast. "
        "LONG the 60-80th illiquidity percentile band within size-tercile x sector buckets, "
        "admitted only if the pre-registered expected premium (40bps/mo) survives the name's "
        "OWN Amihud x $T round-trip impact at $T=$25k (stress $100k in grid). SHORT leg "
        "COMPRESSED, not replaced — the one-line ETF substitution sibling was FALSIFIED "
        "(beta 0.91/1.07, sel-alpha -0.77/-1.12, demoted), proving the multi-name short "
        "carries real selection alpha: short the top-N (N pre-registered in {15,25}) "
        "most-liquid-quintile names with >$20M/day median dollar volume (borrowable at "
        "retail on IB/Alpaca, ~$100-170/name at $5K), sector-balanced to mirror the long "
        "leg, equal weight. Only the RESIDUAL trailing-60d beta is hedged via IWM, hard "
        "cap 30% of book (a trim, not the leg; >30% in >10% of months = falsified ETF "
        "variant in disguise). HARD PRE-REGISTERED GATES promoted from post-hoc demotion "
        "criteria: |beta_to_universe| < 0.3 AND selection_alpha_sharpe > 0 at the FDR bar, "
        "evaluated IN-SEARCH before any holdout read. Gross <= 2x. Monthly rebalance, "
        "hysteresis exit band 55-85. Expected: net-of-own-impact premium positive and "
        "monotonically WEAKENING (not reversing) toward liquid/large tiers; selection "
        "alpha degrades GRACEFULLY (no cliff) from full-quintile to N=15 — a cliff means "
        "the alpha lives in breadth and the retail-deployable claim fails (report honestly, "
        "keep full-quintile as scale-up target). STANDALONE first; trend tail-overlay "
        "(<=25%) only after a clean pass and only if it cuts crisis drawdown without "
        "diluting Sharpe."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "nshort15": {"n_short": 15},
        "t_stress": {"trade_T": 100000.0},
        "hysteresis_tight": {"hys_lo": 0.58, "hys_hi": 0.82},
    },
    scope="broad",
    generalization_universes=list(GEN_SPECS.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=60,
)