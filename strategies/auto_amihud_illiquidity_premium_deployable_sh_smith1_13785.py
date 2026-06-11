"""
Amihud illiquidity premium — DEPLOYABLE-SHORT variant (v3 of the elite Amihud family).

LONG  : most-illiquid Amihud quintile within each size tercile (equal-weight, fractional-share OK).
SHORT : top-N (N=15, frozen PRIMARY) MOST-liquid names per size tercile, $10–$500 whole-share
        price band, 10% single-name cap — concentrated, easy-to-borrow short book.
TRIM  : single IWM short sized to the RESIDUAL trailing-60d beta only (the ETF is a beta-trim,
        NOT a replacement for stock selection — the wholesale ETF-hedge mutation was FALSIFIED
        2026-06-10: beta 0.91 / sel-alpha −0.77).

FIX vs failed run: IWM is an ETF — it is NOT in Sharadar SEP (US single stocks only), so
sep_panel([IWM]) returned an empty frame and the 'etf' MultiIndex block had no columns ->
KeyError 'etf' in signal(). The ETF sleeve now loads via yf_panel (the sanctioned ETF source),
is reindexed to the equity calendar, and signal() degrades gracefully (zero trim) if the ETF
series is unavailable.

Pre-registered costs: 60bps round-trip long leg (illiquid names), 15bps round-trip short leg,
50bps/yr borrow haircut on all short notional (GC names), 4bps round-trip on the ETF trim.
Pre-registered hard gates (enforced by the harness beta-confound gate): |beta_to_universe|<0.3
AND selection_alpha_sharpe>0 — a classical PROMOTE with sel-alpha<=0 is an automatic FAIL.
Grid variants are the pre-registered short-leg SUFFICIENCY CURVE (N in {10,15,25,full-quintile})
— diagnostics only, N=15 is the frozen primary; sel-alpha must degrade gracefully across N.

All weights are built from data through day t and shifted ONE day before net_of_cost — the
shift(1) lag is applied explicitly in signal().
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2003-01-01"
_HEDGE_ETF = "IWM"

# module-level sector registry (merged across search + gen universes; gens are ticker-disjoint)
_SECTORS = {}


# ----------------------------------------------------------------------------- universes / panels
def _search_universe():
    """Search universe: small + mid cap, sector-spread, survivorship-clean (~1000 names)."""
    t_small, m_small = sector_universe(marketcap="Small", top_n_per_sector=60)
    t_mid, m_mid = sector_universe(marketcap="Mid", top_n_per_sector=35)
    smap = {**m_small, **m_mid}
    tickers = sorted(set(t_small) | set(t_mid))
    return tickers, smap


def _load_etf(index):
    """ETF beta-trim sleeve via yf_panel (ETFs are NOT in Sharadar SEP). Graceful fallback."""
    try:
        etf = yf_panel([_HEDGE_ETF], START)
    except Exception:
        etf = None
    if etf is None or etf.empty or _HEDGE_ETF not in etf.columns:
        etf = pd.DataFrame(index=index, columns=[_HEDGE_ETF], dtype=float)
    else:
        etf = etf[[_HEDGE_ETF]].reindex(index).ffill()
    return etf


def _build_panel(tickers, sector_map):
    """MultiIndex-column panel: close/closeadj/volume for the equity universe + ETF sleeve."""
    tickers = sorted(set(tickers) - {_HEDGE_ETF})
    close = sep_panel(tickers, START, field="close")        # unadjusted: $-band + dollar-volume
    cadj = sep_panel(tickers, START, field="closeadj")      # adjusted: returns
    vol = sep_panel(tickers, START, field="volume")
    cols = sorted(set(close.columns) & set(cadj.columns) & set(vol.columns))
    etf = _load_etf(close.index)                            # residual beta-trim sleeve only
    panel = pd.concat(
        {"close": close[cols], "closeadj": cadj[cols], "volume": vol[cols], "etf": etf},
        axis=1,
    )
    smap = {k: v for k, v in sector_map.items() if k in set(cols)}
    smap[_HEDGE_ETF] = "ETF"
    _SECTORS.update(smap)
    panel.attrs["sector_map"] = smap
    return panel


def load_data() -> pd.DataFrame:
    tickers, smap = _search_universe()
    return _build_panel(tickers, smap)


def load_gen_data(label) -> pd.DataFrame:
    """Three pre-registered generalization universes, ticker-DISJOINT from the search universe."""
    search_set = set(_search_universe()[0])
    if label == "micro_top300":
        t, m = sector_universe(marketcap="Micro", top_n_per_sector=30)
    elif label == "large_top300":
        t, m = sector_universe(marketcap="Large", top_n_per_sector=30)
    elif label == "small_deep_slice":
        # deeper small-cap liquidity tier: per-sector ranks ~61-110, minus the search names
        t, m = sector_universe(marketcap="Small", top_n_per_sector=110)
        t = [x for x in t if x not in search_set][:350]
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    t = [x for x in t if x not in search_set]
    m = {k: v for k, v in m.items() if k in set(t)}
    return _build_panel(t, m)


# ----------------------------------------------------------------------------- signal
def signal(panel, lookback=63, n_short=15, entry_pct=0.80, hyst_pct=0.70,
           long_cost_bps=30.0, short_cost_bps=7.5, etf_cost_bps=2.0, borrow_bps_yr=50.0,
           beta_lb=60, beta_trim_cap=0.5, price_min=10.0, price_max=500.0,
           short_name_cap=0.10, **_):
    close = panel["close"]
    cadj = panel["closeadj"]
    volume = panel["volume"]
    if "etf" in panel.columns.get_level_values(0):
        etf_px = panel["etf"]
    else:  # defensive: missing sleeve -> zero trim
        etf_px = pd.DataFrame(index=close.index, columns=[_HEDGE_ETF], dtype=float)
    etf_ok = etf_px.notna().any().any()

    rets = cadj.pct_change()
    etf_rets = etf_px.pct_change().fillna(0.0)  # NaN sleeve contributes zero
    dates = rets.index

    dollar_vol = (close * volume).where(lambda x: x > 0)
    minp = max(20, int(lookback * 0.6))
    illiq = (rets.abs() / dollar_vol).rolling(lookback, min_periods=minp).mean()  # Amihud
    adv = dollar_vol.rolling(lookback, min_periods=minp).median()                 # size/liq proxy

    # per-name trailing beta to the equal-weight universe (for the RESIDUAL trim only)
    mkt = rets.mean(axis=1)
    mvar = mkt.rolling(beta_lb, min_periods=40).var()
    beta = rets.rolling(beta_lb, min_periods=40).cov(mkt).div(mvar, axis=0)

    # monthly rebalance dates (first trading day of month) after warmup
    firsts = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().tolist()
    warm = dates[min(lookback + beta_lb, len(dates) - 1)]
    rebal = [d for d in firsts if d >= warm]

    WL = pd.DataFrame(np.nan, index=dates, columns=rets.columns)
    WS = pd.DataFrame(np.nan, index=dates, columns=rets.columns)
    WE = pd.DataFrame(np.nan, index=dates, columns=etf_px.columns)
    held_long = set()

    for d in rebal:
        am, sz, px, bt = illiq.loc[d], adv.loc[d], close.loc[d], beta.loc[d]
        valid = am.notna() & sz.notna() & (sz > 0)
        names = am.index[valid]
        if len(names) < 60:
            continue
        terc = pd.qcut(sz[names].rank(method="first"), 3, labels=False)

        longs, shorts = [], []
        for b in range(3):
            bn = terc.index[terc == b]
            if len(bn) < 15:
                continue
            # LONG: most-illiquid quintile, with hysteresis (hold down to 70th pct)
            rk = am[bn].rank(pct=True)
            entries = set(bn[rk >= entry_pct])
            holds = {t for t in held_long if t in set(bn) and rk.get(t, 0.0) >= hyst_pct}
            longs += sorted(entries | holds)
            # SHORT: top-N most-liquid of the liquid quintile, $-band whole-share filter
            liq_q = [t for t in bn[sz[bn].rank(pct=True) >= 0.80]
                     if price_min <= px.get(t, np.nan) <= price_max]
            liq_q.sort(key=lambda t: -sz[t])
            shorts += liq_q[: min(int(n_short), len(liq_q))]

        if not longs or not shorts:
            continue
        held_long = set(longs)

        wl = pd.Series(1.0 / len(longs), index=longs)                       # equal-weight long
        ws = pd.Series(1.0 / len(shorts), index=shorts).clip(upper=short_name_cap)
        ws = ws / ws.sum()                                                  # 10% cap, renorm

        # residual beta of the long-short book -> single-ETF trim (clipped, expected small)
        pbeta = float((wl * bt.reindex(wl.index)).sum() - (ws * bt.reindex(ws.index)).sum())
        if not np.isfinite(pbeta):
            pbeta = 0.0
        trim = float(np.clip(pbeta, -beta_trim_cap, beta_trim_cap)) if etf_ok else 0.0

        rowL = pd.Series(0.0, index=rets.columns); rowL[wl.index] = wl.values
        rowS = pd.Series(0.0, index=rets.columns); rowS[ws.index] = -ws.values
        WL.loc[d], WS.loc[d] = rowL, rowS
        WE.loc[d, _HEDGE_ETF] = -trim

    WL = WL.ffill().fillna(0.0)
    WS = WS.ffill().fillna(0.0)
    WE = WE.ffill().fillna(0.0)

    # --- the one-day lag is applied HERE (weights built from data through t, traded t+1)
    WL_lag, WS_lag, WE_lag = WL.shift(1).fillna(0.0), WS.shift(1).fillna(0.0), WE.shift(1).fillna(0.0)

    r_long = net_of_cost(WL_lag, rets, cost_bps=long_cost_bps, name="long")
    r_short = net_of_cost(WS_lag, rets, cost_bps=short_cost_bps, name="short")
    r_etf = net_of_cost(WE_lag, etf_rets, cost_bps=etf_cost_bps, name="etf")

    # pre-registered 50bps/yr borrow haircut on ALL short notional (stocks + ETF trim)
    short_gross = WS_lag.clip(upper=0.0).abs().sum(axis=1) + WE_lag.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_gross * (borrow_bps_yr / 1e4 / 252.0)

    daily = (r_long.reindex(dates).fillna(0.0)
             + r_short.reindex(dates).fillna(0.0)
             + r_etf.reindex(dates).fillna(0.0)
             - borrow.reindex(dates).fillna(0.0))
    if rebal:
        daily = daily.loc[rebal[0]:]
    daily.name = "amihud_illiq_topN_short_v3"

    # trade ledger (kit labeller stamps entry_regime — never hand-rolled)
    smap = dict(_SECTORS)
    smap.update(panel.attrs.get("sector_map") or {})
    smap.setdefault(_HEDGE_ETF, "ETF")
    W_all = pd.concat([WL_lag + WS_lag, WE_lag], axis=1)
    rets_all = pd.concat([rets, etf_rets], axis=1)
    trades = trades_from_weights(W_all, rets_all, smap)

    return daily.dropna(), trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="amihud_illiq_topN_short_v3",
    family="amihud_illiquidity",
    title=("Amihud illiquidity — deployable-short: long illiquid quintile / short top-15 "
           "most-liquid per size tercile + residual IWM beta-trim"),
    markets=["us_equities_small_mid"],
    data_desc=("Sharadar SEP close/closeadj/volume (cache v2), survivorship-clean small+mid "
               "sector-spread universe (~1000 names, delisted incl); IWM Close via yf_panel "
               "for the residual beta-trim sleeve only (ETFs are not in SEP)"),
    pre_registration=(
        "Direct evolution of the elite Amihud parents (sel-alpha +2.51/beta -0.31). The "
        "falsified 2026-06-10 sibling proved the multi-name short leg carries real selection "
        "alpha; this variant tests the OPPOSITE hypothesis: a concentrated top-N=15 borrowable "
        "short leg per size tercile retains >=60% of the full-quintile selection alpha while "
        "being $5K-tradable (whole shares, $10-$500 band, 10% name cap), with the ETF demoted "
        "to a residual beta-trim. Frozen PRIMARY: lookback=63, n_short=15, monthly rebal with "
        "70th-pct hysteresis. Costs pre-registered: 60bps RT long, 15bps RT short, 50bps/yr "
        "borrow on short notional, 4bps RT ETF. HARD GATES: |beta_to_universe|<0.3 AND "
        "selection_alpha_sharpe>0 net of all costs — classical PROMOTE with sel-alpha<=0 is an "
        "automatic FAIL. Grid = pre-registered short-leg sufficiency curve (N=10/15/25/full), "
        "diagnostics only; sel-alpha must degrade gracefully across N. Standalone first; any "
        "trend tail-overlay only after holdout+MCPT pass, sized <=25%."),
    load_data=load_data,
    signal=signal,
    default_params={"lookback": 63, "n_short": 15},
    grid={
        "default": {},
        "n_short_10": {"n_short": 10},
        "n_short_25": {"n_short": 25},
        "full_liquid_quintile": {"n_short": 10**6},
        "lookback_126": {"lookback": 126},
    },
    scope="broad",
    generalization_universes=["micro_top300", "large_top300", "small_deep_slice"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=100,
)