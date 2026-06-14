"""
Amihud illiquidity premium — DEPLOYABLE-SHORT v3, TRANCHED-OVERLAP variant.

FROZEN (inherited from the elite parent, sel-alpha +2.51 / beta -0.31 lineage):
  - Size-bucketed (dollar-volume tercile) within-tercile Amihud sort.
  - LONG  = most-illiquid quintile per size tercile, equal weight (fractional shares at deploy).
  - SHORT = top-N=15 most-liquid names per size tercile, $10-$500 price filter,
    10% single-name cap (the 2026-06-10 falsification proved this multi-name short
    leg carries the selection alpha; full-ETF replacement is dead).
  - Residual-only IWM beta trim (act only when trailing |beta| > 0.30), declared
    hedge sleeve (hedge_tickers=["IWM"], hedge_cap=0.35) so the gate judges alpha alone.
  - Costs: 60bps RT long leg (30/side), 15bps RT short leg (7.5/side) + 50bps/yr borrow.
  - Per-tranche hysteresis band (keep incumbents still inside a 1.5x/1.6x band).

THE MUTATION (the only change): monthly full rebalance -> 3 OVERLAPPING MONTHLY
TRANCHES (Jegadeesh-Titman). Book = equal-weight average of 3 cohorts on staggered
calendar-month phases, each held ~3 months, 1/3 reformed per month. Pre-registered
effects: (a) expensive long-leg turnover drops ~50-65%; (b) the tranche average IS
the rebalance-phase-averaged portfolio (date-luck removed structurally); (c) short
leg rotates ~5 names/month (borrow/squeeze pressure down, $5K whole-share feasible).

Sizing note: equal weight WITHIN legs is the frozen pre-registered design (the
illiquid quintile is the asset; inv-vol tilting it toward calm names dilutes the
premium) — overall book is dollar-neutral 1.0 gross/leg.

RUNTIME FIX (vs failed run): IWM is an ETF — it is NOT in Sharadar SEP (equities
only), so sep_panel returned no 'IWM' column -> KeyError in signal(). The hedge
instrument is now loaded via yf_panel (the sanctioned FREE source for ETFs) and
spliced into the panel; signal() additionally degrades gracefully (zero hedge)
if the column is ever missing/empty.

NO LOOKAHEAD: all signals are trailing; every weight matrix is shift(1)-lagged
before net_of_cost / trades_from_weights; hedge beta uses only past returns.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2001-01-01"
HEDGE = "IWM"

# Search universe: small+mid in 5 big sectors. Gen universes are DISJOINT:
# different sectors (gen1/gen2) or different cap tier (gen3). ~300-400 names each.
SEARCH_SECTORS = ["Technology", "Industrials", "Consumer Cyclical",
                  "Healthcare", "Financial Services"]
SEARCH_CFG = ([("Small", 70), ("Mid", 45)], SEARCH_SECTORS)

GEN_UNIVERSES = {
    "smallmid_energy_materials_utils": ([("Small", 70), ("Mid", 45)],
                                        ["Energy", "Basic Materials", "Utilities"]),
    "smallmid_defensive_comms_re": ([("Small", 70), ("Mid", 45)],
                                    ["Consumer Defensive", "Communication Services",
                                     "Real Estate"]),
    "largecap_all_sectors": ([("Large", 30)], None),
}

# ticker -> sector, accumulated by the load_* functions; signal() reads it for the ledger.
_SECTORS = {HEDGE: "ETF"}


def _build_universe(cap_specs, sectors):
    tickers, smap = [], {}
    for mc, n_per_sector in cap_specs:
        tks, sm = sector_universe(marketcap=mc, top_n_per_sector=n_per_sector)
        for t in tks:
            sec = sm.get(t, "Unknown")
            if sectors is None or sec in sectors:
                tickers.append(t)
                smap[t] = sec
    return sorted(set(tickers)), smap


def _make_panel(tickers):
    """SEP panel for the stocks + IWM (ETF) spliced in from yf_panel.

    Sharadar SEP covers equities only — IWM must come from yfinance (the
    sanctioned FREE source for ETFs). yf Close is used for both 'close' and
    'closeadj' (hedge sleeve only needs returns); volume left NaN.
    """
    tk = sorted(set(tickers) - {HEDGE})
    parts = {f: sep_panel(tk, START, field=f) for f in ("close", "closeadj", "volume")}
    idx = parts["closeadj"].index
    try:
        iwm = yf_panel([HEDGE], START)
        iwm_close = iwm[HEDGE].reindex(idx).ffill()
    except Exception:
        iwm_close = pd.Series(np.nan, index=idx)
    for f in ("close", "closeadj"):
        parts[f] = parts[f].copy()
        parts[f][HEDGE] = iwm_close
    parts["volume"] = parts["volume"].copy()
    parts["volume"][HEDGE] = np.nan
    return pd.concat(parts, axis=1)  # MultiIndex columns: (field, ticker)


def load_data() -> pd.DataFrame:
    caps, sectors = SEARCH_CFG
    tickers, smap = _build_universe(caps, sectors)
    _SECTORS.update(smap)
    return _make_panel(tickers)


def load_gen_data(label) -> pd.DataFrame:
    caps, sectors = GEN_UNIVERSES[label]
    tickers, smap = _build_universe(caps, sectors)
    _SECTORS.update(smap)
    return _make_panel(tickers)


def _form_cohort(am_row, sz_row, px_row, bvol_row, prev_long, prev_short, p):
    """One tranche's formation: frozen within-tercile Amihud sort + hysteresis.
    ABLATION: bvol_row = 20d median $-volume per name; short candidates must clear
    p['borrow_floor'] (0 = OFF = byte-identical to deployed v3)."""
    valid = am_row.notna() & sz_row.notna() & px_row.notna() & (px_row > 1.0)
    names = am_row.index[valid]
    if len(names) < 90:
        return None, prev_long, prev_short

    sz = sz_row[names]
    terc = pd.Series(pd.qcut(sz.rank(method="first"), 3, labels=False), index=sz.index)

    w = {}
    long_set, short_set = set(), set()
    for t in range(3):
        members = terc.index[terc == t]
        if len(members) < 15:
            continue
        # ---- LONG: most-illiquid quintile (hysteresis band = 1.5x quintile) ----
        ill = am_row[members].sort_values(ascending=False)  # most illiquid first
        q = max(1, len(ill) // 5)
        band_l = set(ill.index[: int(np.ceil(q * p["long_band"]))])
        held_l = [n for n in ill.index if n in prev_long and n in band_l][:q]
        sel_long = held_l + [n for n in ill.index if n not in held_l][: q - len(held_l)]
        # ---- SHORT: top-N most-liquid, $10-$500 filter (band = 1.6x N) ----
        liq = am_row[members].sort_values(ascending=True)   # most liquid first
        elig = [n for n in liq.index
                if p["px_lo"] <= px_row[n] <= p["px_hi"] and n not in sel_long
                and (p["borrow_floor"] <= 0.0                       # ABLATION: borrowability floor
                     or bvol_row.get(n, 0.0) >= p["borrow_floor"])]  # NaN/missing -> excluded (conservative)
        ns = min(p["n_short"], len(elig))
        band_s = set(elig[: int(np.ceil(p["n_short"] * p["short_band"]))])
        held_s = [n for n in elig if n in prev_short and n in band_s][:ns]
        sel_short = held_s + [n for n in elig if n not in held_s][: ns - len(held_s)]

        if sel_long:
            wl = (1.0 / 3.0) / len(sel_long)
            for n in sel_long:
                w[n] = w.get(n, 0.0) + wl
            long_set.update(sel_long)
        if sel_short:
            ws = min((1.0 / 3.0) / len(sel_short), p["short_name_cap"])
            for n in sel_short:
                w[n] = w.get(n, 0.0) - ws
            short_set.update(sel_short)

    if not w:
        return None, prev_long, prev_short
    return pd.Series(w, dtype=float), long_set, short_set


def signal(panel, **params):
    p = dict(amihud_lb=63, size_lb=126, n_short=15, n_tranches=3,
             long_band=1.5, short_band=1.6,
             beta_cap=0.30, hedge_cap=0.35, beta_lb=60,
             cost_long_bps=30.0, cost_short_bps=7.5, cost_hedge_bps=2.0,
             borrow_rate=0.005, px_lo=10.0, px_hi=500.0, short_name_cap=0.10,
             borrow_floor=0.0)   # ABLATION: 0 = OFF (byte-identical to v3); >0 = $ floor on short candidates
    p.update(params)

    close, closeadj, volume = panel["close"], panel["closeadj"], panel["volume"]
    rets = closeadj.pct_change()
    stocks = [c for c in close.columns if c != HEDGE]
    dates = rets.index

    # Trailing-only Amihud illiquidity and dollar-volume size proxy.
    dvol = (close[stocks] * volume[stocks]).replace(0.0, np.nan)
    amihud = ((rets[stocks].abs() / dvol) * 1e6).rolling(
        p["amihud_lb"], min_periods=int(p["amihud_lb"] * 0.6)).mean()
    size = dvol.rolling(p["size_lb"], min_periods=60).median()
    bvol = dvol.rolling(20, min_periods=10).median()   # ABLATION: 20d median $-vol = borrowability proxy

    # Month-end trading dates.
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()

    nT = int(p["n_tranches"])
    tr_w = {k: pd.Series(dtype=float) for k in range(nT)}
    tr_long = {k: set() for k in range(nT)}
    tr_short = {k: set() for k in range(nT)}
    rows = {}
    for d in month_ends:
        k = (d.year * 12 + d.month) % nT  # calendar-stable tranche phase
        new_w, longs, shorts = _form_cohort(amihud.loc[d], size.loc[d],
                                            close[stocks].loc[d], bvol.loc[d],
                                            tr_long[k], tr_short[k], p)
        if new_w is not None:
            tr_w[k], tr_long[k], tr_short[k] = new_w, longs, shorts
        formed = [w for w in tr_w.values() if len(w)]
        if formed:
            # Book = sum of cohorts / n_tranches (JT overlapping average).
            rows[d] = pd.concat(formed, axis=1).fillna(0.0).sum(axis=1) / nT

    W = (pd.DataFrame(rows).T.reindex(dates).ffill().fillna(0.0)
         .reindex(columns=stocks, fill_value=0.0))

    # ---- Residual-only IWM beta trim (trailing 60d, acts only when |beta|>0.30).
    # Degrades to zero hedge if the IWM column is missing or empty.
    have_hedge = HEDGE in rets.columns and rets[HEDGE].notna().sum() > 252
    if have_hedge:
        pre = (W.shift(1) * rets[stocks]).sum(axis=1)  # pre-hedge book, already lagged
        iwm_r = rets[HEDGE]
        beta = (pre.rolling(p["beta_lb"], min_periods=40).cov(iwm_r)
                / iwm_r.rolling(p["beta_lb"], min_periods=40).var())
        exc = (beta.abs() - p["beta_cap"]).clip(lower=0.0) * np.sign(beta)
        hedge_raw = (-exc).clip(-p["hedge_cap"], p["hedge_cap"]).fillna(0.0)
        wk_last = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
        hedge = hedge_raw.where(dates.isin(wk_last.values)).ffill().fillna(0.0)
    else:
        hedge = pd.Series(0.0, index=dates)
    W[HEDGE] = hedge

    # ---- Net-of-cost returns: asymmetric per-leg costs + borrow haircut.
    # W is same-day weights -> shift(1) is OUR lag (no lookahead).
    Wl = W[stocks].clip(lower=0.0)
    Ws = W[stocks].clip(upper=0.0)
    Wh = W[[HEDGE]]
    r_long = net_of_cost(Wl.shift(1), rets, cost_bps=p["cost_long_bps"], name="long")
    r_short = net_of_cost(Ws.shift(1), rets, cost_bps=p["cost_short_bps"], name="short")
    r_hedge = net_of_cost(Wh.shift(1), rets.fillna(0.0)[[HEDGE]],
                          cost_bps=p["cost_hedge_bps"], name="hedge")
    short_gross = (Ws.abs().sum(axis=1) + Wh[HEDGE].clip(upper=0.0).abs()).shift(1)
    borrow = (short_gross * p["borrow_rate"] / 252.0).fillna(0.0)

    daily = (r_long.reindex(dates).fillna(0.0)
             + r_short.reindex(dates).fillna(0.0)
             + r_hedge.reindex(dates).fillna(0.0)
             - borrow)
    daily.name = "amihud_illiq_tranched_v3"

    sector_map = {t: _SECTORS.get(t, "Unknown") for t in W.columns}
    trades = trades_from_weights(W.shift(1), rets, sector_map)
    return daily, trades


SPEC = StrategySpec(
    id="amihud_v3_borrow_ablation",   # renamed from v3 so this can NEVER clobber the deployed registry/wiki
    family="illiquidity_premium",
    title=("Amihud illiquidity premium — deployable-short v3: long illiquid quintile / "
           "short top-15 liquid per size tercile + residual IWM trim, rebuilt as 3 "
           "overlapping monthly tranches"),
    markets=["US_smallmid_equity"],
    data_desc=("Sharadar SEP close/closeadj/volume (survivorship-clean, delisted incl.) "
               "via sector_universe small+mid, 5-sector search slice; IWM closes from "
               "yfinance (ETFs are not in SEP) for the declared residual hedge sleeve only."),
    pre_registration=(
        "KEPT FROZEN from elite parent: within-size-tercile Amihud sort; long = most-"
        "illiquid quintile EW; short = top-15 most-liquid per tercile, $10-$500 filter, "
        "10% name cap; residual-only IWM trim |beta|<0.3; costs 60bps RT long / 15bps RT "
        "+50bps/yr borrow short; hysteresis bands. MUTATION (only change): 3 overlapping "
        "monthly tranches (JT), book = cohort average, 1/3 reformed per month. PRE-"
        "REGISTERED EXPECTATIONS: (a) long-leg turnover -50-65% vs single-date parent "
        "(realized tranched turnover must be <=60% of 'untranched' grid variant or the "
        "cost-hardening mechanism failed); (b) tranche-phase dispersion: all 3 phases "
        "positive net sel-alpha, min phase >=50% of average — one lucky phase = FAIL; "
        "(c) short-leg sufficiency: N in {10,15,25} grid, graceful degradation, N=15 "
        "frozen primary; (d) Amihud-quintile monotonicity within terciles; sector "
        "breadth >=4/leg. Hard gates unchanged: write-once holdout 2022+ at full costs, "
        "MCPT within-size permutation null, |beta_to_universe|<0.3 AND "
        "selection_alpha_sharpe>0. Standalone first; trend overlay (<=25%) only after "
        "holdout+MCPT pass. Gen universes disjoint from search by sector or cap tier."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "short_n10": {"n_short": 10},
        "short_n25": {"n_short": 25},
        "untranched": {"n_tranches": 1},   # parent's single-date monthly path (diagnostic)
        "amihud_lb126": {"amihud_lb": 126},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=96,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
)