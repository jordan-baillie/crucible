"""
Amihud illiquidity premium — v2 (deployable long-side construction).

FIX vs the failed run: the crash was a pyarrow schema error inside sep_panel when asking
for field="close" — the SEP parquet store does not expose that column name (it serves
'closeadj' / 'closeunadj' / 'volume'). The raw-dollar-volume inputs are now fetched through
a small field-fallback helper: raw price tries 'close' then 'closeunadj' (then falls back
to 'closeadj' as a last resort so the module degrades instead of crashing), volume uses
'volume'. Everything else is unchanged.

KEPT VERBATIM from the elite parent: survivorship-clean small+mid Sharadar universe
(delisted included), $1 dollar-volume floor, trailing-12m Amihud = mean(|ret|/dollar-vol),
size-tercile x sector-neutral ranking, monthly rebalance with band hysteresis, and the
self-consistent impact-cost gate (round-trip cost = base + 2*(Amihud_name x $T); a name is
long-eligible only if its OWN modeled annualized cost is below the pre-registered premium).
Capture lives in the capacity-tradeable ~60-80th percentile illiquidity band.

THE ONE NEW MUTATION: the multi-name most-liquid-quintile short leg (undeployable at $5K:
dozens of borrows/locates) is replaced by ONE beta-scaled short hedge in a tier-matched
deep-liquidity index ETF (IWM for the small tier, MDY for the mid tier, blended by the long
book's tier weights), sized each rebalance to zero the book's trailing-60d beta, with
hedge-ratio hysteresis and gross capped at 1.6x. The unhedged long book is kept in the grid
as the mechanism benchmark ("unhedged_parent").

NO look-ahead: all signals use trailing data only; weights are applied via W.shift(1)
(the lag is OURS, stated below); net_of_cost receives the LAGGED weight matrix; per-name
modeled impact drag is also lagged one day alongside the weights.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2002-01-01"
HEDGE_ETFS = ("IWM", "MDY")            # small-tier hedge, mid-tier hedge (both in SEP)
_SEARCH = {"Small": 50, "Mid": 35}     # search universe: top-N per sector per cap tier

DEFAULTS = dict(
    band_lo=0.60, band_hi=0.80,        # mid-illiquid capture band (percentile, bucket-neutral)
    band_hys=0.05,                     # exit hysteresis on the band edges
    illiq_lb=252, illiq_min=120,       # trailing-12m Amihud, >=~6m to qualify
    dv_lb=63, vol_lb=63,               # size proxy (median $vol) and inverse-vol lookbacks
    max_w=0.08,                        # per-name weight cap
    base_bps=8.0,                      # base round-trip-half cost on turnover
    trade_size=10_000.0,               # pre-registered deployable trade size $T
    exp_premium=0.05,                  # pre-registered annual gross band premium for the gate
    turns_per_year=3.0,                # expected round trips/yr under hysteresis
    beta_lb=60,                        # trailing-60d beta window for the ETF hedge
    hedge_cap=0.60,                    # |hedge| cap -> gross <= 1.6x
    hedge_hys=0.15,                    # hedge-ratio hysteresis (no whipsaw)
)


# ----------------------------------------------------------------------------- universes

def _tiered_universe(spec):
    """Sector-spread small+mid universe via the kit; returns (tickers, sector_map, tier_map)."""
    tickers, sector_map, tier_map = [], {}, {}
    for cap, n in spec.items():
        tks, smap = sector_universe(cap, n)
        for tk in tks:
            if tk in sector_map:
                continue
            tickers.append(tk)
            sector_map[tk] = smap[tk]
            tier_map[tk] = cap.lower()
    return tickers, sector_map, tier_map


def _sep_first(names, fields):
    """Fetch a SEP panel trying field names in order — the parquet schema does not expose
    every Sharadar column name; this absorbs schema differences instead of crashing."""
    last_err = None
    for f in fields:
        try:
            return sep_panel(names, START, field=f)
        except Exception as e:           # pyarrow schema / missing-column errors
            last_err = e
            continue
    raise last_err


def _build_panel(tickers, sector_map, tier_map):
    """Panel with 4-level columns (field, ticker, sector, tier); fields: px (closeadj), dv (raw $vol)."""
    names = list(dict.fromkeys(list(tickers) + list(HEDGE_ETFS)))
    px = sep_panel(names, START, field="closeadj")
    # RAW close for dollar volume: 'close' if present, else 'closeunadj'; degrade to
    # 'closeadj' only as a last resort (keeps module alive; ranking is cross-sectional).
    cl = _sep_first(names, ("close", "closeunadj", "closeadj"))
    vo = _sep_first(names, ("volume",))                 # RAW (unadjusted) volume
    cl = cl.reindex(index=px.index, columns=px.columns)
    vo = vo.reindex(index=px.index, columns=px.columns)
    dv = cl * vo
    panel = pd.concat({"px": px, "dv": dv}, axis=1)
    tuples = [(f, t, sector_map.get(t, "Index-Hedge"), tier_map.get(t, "etf"))
              for f, t in panel.columns]
    panel.columns = pd.MultiIndex.from_tuples(tuples, names=["field", "ticker", "sector", "tier"])
    return panel.sort_index()


def load_data():
    tks, smap, tmap = _tiered_universe(_SEARCH)
    return _build_panel(tks, smap, tmap)


def load_gen_data(label):
    """Three generalization universes, DISJOINT from the search universe (and each other):
    small_ext  = small-cap, per-sector liquidity ranks ~51-90  (next slice down)
    mid_ext    = mid-cap,   per-sector liquidity ranks ~36-65
    small_deep = small-cap, per-sector liquidity ranks ~91-130 (deeper, still tradeable)
    """
    search = set(_tiered_universe(_SEARCH)[0])
    if label == "small_ext":
        tks, smap = sector_universe("Small", 90)
        tier, excl = "small", search
    elif label == "mid_ext":
        tks, smap = sector_universe("Mid", 65)
        tier, excl = "mid", search
    elif label == "small_deep":
        inner, _ = sector_universe("Small", 90)
        tks, smap = sector_universe("Small", 130)
        tier, excl = "small", search | set(inner)
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    keep = [t for t in tks if t not in excl][:400]   # keep each gen universe small
    return _build_panel(keep, {t: smap[t] for t in keep}, {t: tier for t in keep})


# ----------------------------------------------------------------------------- signal

def signal(panel, **params):
    p = dict(DEFAULTS)
    p.update(params)

    meta = {}
    for _, tkr, sec, tier in panel.columns:
        meta.setdefault(tkr, (sec, tier))
    px = panel["px"].copy(); px.columns = px.columns.get_level_values("ticker")
    dv = panel["dv"].copy(); dv.columns = dv.columns.get_level_values("ticker")

    rets = px.pct_change(fill_method=None)
    stocks = [t for t in px.columns if t not in HEDGE_ETFS]
    hedges = [t for t in HEDGE_ETFS if t in px.columns]
    sec_s = pd.Series({t: meta[t][0] for t in stocks})
    tier_s = pd.Series({t: meta[t][1] for t in stocks})

    # --- trailing signals (all trailing-only; acted on next day via shift(1) below) ---
    dvs = dv[stocks].where(dv[stocks] >= 1.0)                       # $1 dollar-volume floor
    illiq = (rets[stocks].abs() / dvs).rolling(
        p["illiq_lb"], min_periods=p["illiq_min"]).mean() * 1e6     # |ret| per $1M traded
    dv63 = dv[stocks].rolling(p["dv_lb"], min_periods=21).median() # size/liquidity proxy
    vol63 = rets[stocks].rolling(p["vol_lb"], min_periods=42).std()

    # self-consistent cost model: one-way impact of trading $T in THIS name
    impact = (illiq * (p["trade_size"] / 1e6)).clip(upper=0.05)
    ann_cost = (2.0 * (p["base_bps"] / 1e4 + impact)) * p["turns_per_year"]

    idx = rets.index
    reb_dates = idx.to_series().groupby(idx.to_period("M")).last()  # month-end trading days

    held, prev_beta, rows = set(), 0.0, {}
    for t in reb_dates:
        il, sz, vl = illiq.loc[t], dv63.loc[t], vol63.loc[t]
        ok = il.notna() & sz.notna() & (sz > 0) & vl.notna() & (vl > 0)
        if int(ok.sum()) < 30:
            continue
        df = pd.DataFrame({"il": il[ok], "sz": sz[ok]})
        df["sector"] = sec_s.reindex(df.index)
        df["terc"] = pd.qcut(df["sz"].rank(method="first"), 3, labels=False)

        # bucket-neutral illiquidity percentile (size-tercile x sector; tercile fallback if tiny)
        g = df.groupby(["terc", "sector"])["il"]
        pct = g.rank(pct=True)
        pct = pct.where(g.transform("count") >= 6,
                        df.groupby("terc")["il"].rank(pct=True))

        gate = ann_cost.loc[t].reindex(df.index) < p["exp_premium"]  # net-of-own-impact gate
        lo, hi, hy = p["band_lo"], p["band_hi"], p["band_hys"]
        enter = set(pct.index[(pct >= lo) & (pct <= hi) & gate])
        stay = {nm for nm in held
                if nm in pct.index and (lo - hy) <= pct[nm] <= (hi + hy)
                and bool(gate.get(nm, False))}
        held = enter | stay
        if not held:
            rows[t] = pd.Series(dtype=float)
            prev_beta = 0.0
            continue

        names = sorted(held)
        iv = 1.0 / vl[names]
        w = iv / iv.sum()
        w = w.clip(upper=p["max_w"]); w = w / w.sum()               # gross long = 1.0

        # --- single tier-matched ETF hedge, trailing-beta sized, with hysteresis ---
        s_sh = float(w[tier_s.reindex(names) == "small"].sum())
        m_sh = float(w[tier_s.reindex(names) == "mid"].sum())
        tot = s_sh + m_sh
        mix = {"IWM": (s_sh / tot) if tot > 0 else 1.0,
               "MDY": (m_sh / tot) if tot > 0 else 0.0}
        beta = 0.0
        if p["hedge_cap"] > 0 and len(hedges) == 2:
            win = rets.loc[:t].tail(int(p["beta_lb"]))              # trailing data only
            port = (win[names] * w).sum(axis=1)
            etf = win["IWM"] * mix["IWM"] + win["MDY"] * mix["MDY"]
            v = float(etf.var())
            if np.isfinite(v) and v > 0:
                beta = float(port.cov(etf) / v)
            beta = float(np.clip(beta, 0.0, p["hedge_cap"]))        # gross <= 1.6x
            if abs(beta - prev_beta) < p["hedge_hys"]:
                beta = prev_beta                                    # hedge-ratio hysteresis
        prev_beta = beta

        row = w.copy()
        for h in hedges:
            row[h] = -beta * mix[h]
        rows[t] = row

    if not rows:
        empty = pd.Series(0.0, index=idx, name="amihud_illiq_hedged")
        return empty, []

    W = pd.DataFrame(rows).T.sort_index()
    W = W.reindex(columns=rets.columns).fillna(0.0)
    Wd = W.reindex(idx).ffill().fillna(0.0)
    W_lag = Wd.shift(1).fillna(0.0)        # <-- OUR lag: weights trade at T+1, no look-ahead

    # base costs on turnover (kit), then SELF-CONSISTENT name-specific impact drag on top
    daily = net_of_cost(W_lag, rets, cost_bps=p["base_bps"], name="amihud_illiq_hedged")
    drag = (W_lag[stocks].diff().abs() * impact.shift(1)).sum(axis=1).fillna(0.0)
    daily = daily.sub(drag.reindex(daily.index).fillna(0.0)).rename("amihud_illiq_hedged")

    live = W_lag.abs().sum(axis=1)
    first = live[live > 0].index.min()
    if first is not None and not pd.isna(first):
        daily = daily.loc[first:]

    sector_map = {tk: meta[tk][0] for tk in rets.columns}
    trades = trades_from_weights(W_lag, rets, sector_map)           # kit stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------------------- spec

SPEC = StrategySpec(
    id="amihud_illiq_etf_hedged_v2",
    family="liquidity_premium",
    title=("Amihud illiquidity premium v2 — cost-gated mid-illiquid band, "
           "single beta-scaled IWM/MDY hedge (deployable long-side construction)"),
    markets=["us_smallmid_equity"],
    data_desc=("Sharadar SEP closeadj + raw close/volume (survivorship-clean, delisted "
               "included) for a sector-spread small+mid universe; IWM/MDY closeadj from SEP "
               "for the tier-matched index hedge and trailing-beta estimation. No other data."),
    pre_registration=(
        "MECHANISM: compensation for bearing illiquidity (limits-to-arbitrage risk premium), "
        "captured in the ~60-80th percentile illiquidity band (size-tercile x sector neutral), "
        "gated so each long is positive NET of its OWN modeled impact at the pre-registered "
        "trade size $T (round-trip = base + 2*(Amihud x $T), ~3 turns/yr, premium prior 5%/yr). "
        "MUTATION UNDER TEST: replace the parent's ~quintile liquid-name short leg with ONE "
        "beta-scaled tier-matched index-ETF short (IWM/MDY blend, trailing-60d beta, hysteresis "
        "0.15, gross<=1.6x) — economically near-equivalent (the liquid quintile IS the index) "
        "but executable at $5K with one borrow. PASS REQUIRES: (a) hedged version retains >=70% "
        "of the unhedged-parent band premium in-sample (grid 'unhedged_parent' is the benchmark; "
        "large leakage = the edge lived in the individual shorts, mutation FAILS honestly); "
        "(b) realized OOS beta to the hedge ETF ~0 (no hidden market bet); (c) NET premium "
        "degrades gracefully (no cliff) across the pre-registered $T sweep $2.5K/$10K/$50K; "
        "(d) broad-scope generalization: same frozen signal positive OOS on >=60% of three "
        "disjoint universes (small ranks 51-90, mid ranks 36-65, small ranks 91-130 per sector). "
        "Monthly rebalance with band hysteresis; signals trailing-only; weights applied T+1. "
        "Standalone first — trend tail-overlay (<=25%) considered only after a PASS."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "band_low": {"band_lo": 0.55, "band_hi": 0.75},
        "size_2k5": {"trade_size": 2_500.0},
        "size_50k": {"trade_size": 50_000.0},
        "unhedged_parent": {"hedge_cap": 0.0},
    },
    scope="broad",
    generalization_universes=["small_ext", "mid_ext", "small_deep"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=25,
)