"""
VARIANT v2.2 — Amihud illiquidity premium, self-consistent impact-cost gate,
RETAIL-DEPLOYABLE construction.

FIX vs v2.1 (DATA-ADAPTER CRASH): `sep_panel(..., field="close")` raised a
pyarrow schema error — the vendored SEP store does not expose a raw `close`
column (the `closeadj` call on the line above succeeded). The dollar-volume
build now requests only fields the store actually serves, with a guarded
fallback chain: closeunadj -> close -> closeadj (adjusted-price fallback is
acceptable because Amihud enters the signal as a cross-sectional RANK within
size terciles, so a uniform split-adjustment bias largely washes; the level-
based cost gate then only becomes conservative). `volume` is fetched the same
guarded way; if truly absent we fail LOUDLY (Amihud is undefined without it)
rather than silently degrade.

Carried fix from v2.1 (EXIT BUG): every rebalance row is a FULL weight vector
(explicit 0.0 for every non-held column) so band/hysteresis/cost-gate/max_names
exits are real exits and the long book sums to 1 at every rebalance.

Premium: illiquidity (compensation for bearing illiquidity — a limits-to-arbitrage
risk premium, NOT a forecast). LONG-ONLY mid-illiquid band (60-80th pct Amihud
within size tercile, sector-capped) in survivorship-clean small+mid caps, hedged
with a beta-matched SHORT in ONE liquid index instrument (IWM, fallback SPY).

Self-consistent cost gate: a name's modeled round-trip cost
    2*base_bps + 2 * Amihud_i * $T  (at pre-registered stress size $T)
must stay under cost_cap_bps, with the SAME Amihud estimate used as the signal.
Realized impact drag also charged on every trade at actual account sizing
(lambda * $traded), on top of 8 bps base via net_of_cost.

No lookahead: all signals (Amihud, ADV, size proxy, vols, beta) use data through
the rebalance close only; daily weight matrix is shift(1)-lagged before pricing.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2000-01-01"
HEDGE_TICKERS = ("IWM", "SPY")  # hedge instruments, NOT universe members

# Search universe: small+mid caps in 4 sectors. Gen universes: DISJOINT sector
# slices (no shared tickers with search; hedge instrument is shared by design —
# it is an instrument, not a candidate).
SEARCH_SECTORS = ("Technology", "Healthcare", "Industrials", "Consumer Cyclical")
GEN_UNIVERSES = {
    "fin_realestate": ("Financial Services", "Real Estate"),
    "defensive_comm": ("Consumer Defensive", "Utilities", "Communication Services"),
    "energy_materials": ("Energy", "Basic Materials"),
}

DEFAULTS = dict(
    band_lo=0.60, band_hi=0.80,      # mid-illiquid band (pct rank within size tercile)
    hyst=0.05,                        # membership hysteresis band (turnover control)
    amihud_lb=252, amihud_min_obs=120,
    adv_floor_usd=200_000.0,          # tradability floor: 63d median dollar volume
    px_floor=2.0,
    base_bps=8.0,                     # base cost on turnover (net_of_cost)
    gate_trade_usd=10_000.0,          # pre-registered $T stress size for the cost gate
    cost_cap_bps=80.0,                # max modeled round-trip cost vs ~monthly premium
    account_usd=25_000.0,             # realized-impact sizing for the drag term
    w_cap=0.05, max_names=55, sector_cap_frac=0.40,
    beta_lb=60,                       # trailing days for hedge beta
    hedge_cap=1.0,                    # |hedge| <= 1 -> gross <= 2x
    hedge_drift_tol=0.10,             # +/-10% beta drift band before re-sizing hedge
    hedge_pref=("IWM", "SPY"),
)


def _universe(sectors, top_n_per_sector=60):
    """Small+Mid tickers restricted to the given sectors, via the kit builder."""
    tickers, smap = [], {}
    for cap in ("Small", "Mid"):
        tk, sm = sector_universe(marketcap=cap, top_n_per_sector=top_n_per_sector)
        for t in tk:
            sec = sm.get(t)
            if sec in sectors and t not in smap:
                tickers.append(t)
                smap[t] = sec
    return tickers, smap


def _mi(df, field, secmap):
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(
        [(field, t, secmap.get(t, "HEDGE")) for t in df.columns],
        names=["field", "ticker", "sector"],
    )
    return df


def _try_sep(tickers, field_candidates):
    """Fetch the first SEP field the vendored store actually serves (schema-safe)."""
    for f in field_candidates:
        try:
            df = sep_panel(tickers, START, field=f)
            if df is not None and len(df.columns) > 0:
                return df
        except Exception:
            continue
    return None


def _build_panel(sectors, top_n_per_sector=60):
    tickers, smap = _universe(sectors, top_n_per_sector)
    px_u = sep_panel(tickers, START, field="closeadj")       # survivorship-clean

    # Raw dollar volume: guarded field chain (store may not vendor raw `close`).
    vol = _try_sep(tickers, ("volume",))
    if vol is None:
        raise RuntimeError("SEP 'volume' unavailable — Amihud illiquidity undefined; aborting loudly")
    raw = _try_sep(tickers, ("closeunadj", "close"))
    if raw is None:
        # Adjusted-price fallback: signal uses cross-sectional ranks within size
        # terciles (split bias washes); level-based cost gate becomes conservative.
        raw = px_u
    raw = raw.reindex(index=px_u.index, columns=px_u.columns)
    vol = vol.reindex(index=px_u.index, columns=px_u.columns)
    dvol = raw * vol

    px_h = sep_panel(list(HEDGE_TICKERS), START, field="closeadj")
    hmap = {h: "HEDGE" for h in px_h.columns}
    panel = pd.concat(
        [_mi(px_u, "px", smap), _mi(px_h, "px", hmap), _mi(dvol, "dvol", smap)],
        axis=1,
    ).sort_index(axis=1)
    return panel


def load_data():
    return _build_panel(SEARCH_SECTORS, top_n_per_sector=60)


def load_gen_data(label):
    return _build_panel(GEN_UNIVERSES[label], top_n_per_sector=60)


def signal(panel, **params):
    p = {**DEFAULTS, **params}

    # ---- unpack panel (sector map travels in the column MultiIndex) ----
    sector_map = {t: s for (f, t, s) in panel.columns if f == "px"}
    px = panel["px"].copy()
    px.columns = px.columns.get_level_values(0)
    dvol = panel["dvol"].copy()
    dvol.columns = dvol.columns.get_level_values(0)

    hedges = [h for h in p["hedge_pref"] if h in px.columns]
    univ = [t for t in px.columns if sector_map.get(t) != "HEDGE"]
    cols = univ + hedges

    rets = px.pct_change(fill_method=None)
    ru, dv = rets[univ], dvol[univ]

    # ---- trailing signals (all data-through-date only) ----
    amihud = (ru.abs() / dv.where(dv > 0)).rolling(
        p["amihud_lb"], min_periods=p["amihud_min_obs"]).mean()      # |ret| per $ traded
    adv = dv.rolling(63, min_periods=40).median()
    size_proxy = dv.rolling(252, min_periods=120).median()           # PIT size proxy
    vol63 = ru.rolling(63, min_periods=40).std()

    idx = px.index
    rebal = idx.to_series().groupby(idx.to_period("M")).last().tolist()  # month-ends

    rows = {}
    prev_hold: set = set()
    prev_hedge = (None, 0.0)
    band_mid = 0.5 * (p["band_lo"] + p["band_hi"])

    for d in rebal:
        am, a, sz = amihud.loc[d], adv.loc[d], size_proxy.loc[d]
        prc = px.loc[d, univ]
        elig = am.notna() & a.notna() & sz.notna() & (a >= p["adv_floor_usd"]) & (prc >= p["px_floor"])
        names = elig.index[elig]
        if len(names) < 30:
            continue  # skipped rebalance: previous full row carries via ffill

        # size-tercile-neutral illiquidity rank (higher = more illiquid)
        terc = pd.qcut(sz[names].rank(method="first"), 3, labels=False, duplicates="drop")
        rank = am[names].groupby(terc).rank(pct=True)

        # self-consistent cost gate: round-trip = 2*base + 2*lambda*$T (in bps)
        rt_bps = 2.0 * p["base_bps"] + 2.0e4 * am[names] * p["gate_trade_usd"]
        ok_cost = rt_bps <= p["cost_cap_bps"]

        in_band = (rank >= p["band_lo"]) & (rank <= p["band_hi"])
        held = pd.Series(rank.index.isin(prev_hold), index=rank.index)
        keep = held & (rank >= p["band_lo"] - p["hyst"]) & (rank <= p["band_hi"] + p["hyst"])
        sel = rank.index[(in_band | keep) & ok_cost]
        if len(sel) < 10:
            continue

        # sector cap + max_names: prefer names nearest the band centre
        order = (rank[sel] - band_mid).abs().sort_values().index
        cap = max(4, int(np.ceil(p["sector_cap_frac"] * min(len(sel), p["max_names"]))))
        chosen, per_sec = [], {}
        for t in order:
            s = sector_map.get(t, "?")
            if per_sec.get(s, 0) < cap and len(chosen) < p["max_names"]:
                per_sec[s] = per_sec.get(s, 0) + 1
                chosen.append(t)
        sel = chosen

        # inverse-vol long weights, capped, sum to 1
        v = vol63.loc[d, sel]
        w = (1.0 / v.where(v > 0)).fillna(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        w = w.clip(upper=p["w_cap"])
        w = w / w.sum()

        # beta-matched index hedge (trailing 60d, drift-tolerance band)
        win = ru.loc[:d, sel].tail(p["beta_lb"])
        rp = win.mul(w, axis=1).sum(axis=1)
        h_pick, beta = None, 0.0
        for h in hedges:
            hr = rets.loc[:d, h].tail(p["beta_lb"])
            pair = pd.concat([rp, hr], axis=1).dropna()
            if len(pair) >= int(0.8 * p["beta_lb"]):
                var = pair.iloc[:, 1].var()
                beta = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / var) if var > 0 else 1.0
                h_pick = h
                break
        beta = float(np.clip(beta, 0.0, p["hedge_cap"]))  # gross <= 2x
        if (h_pick is not None and h_pick == prev_hedge[0] and prev_hedge[1] > 0
                and abs(beta - prev_hedge[1]) <= p["hedge_drift_tol"] * prev_hedge[1]):
            beta = prev_hedge[1]  # within tolerance band: don't churn the hedge

        # FULL weight vector: explicit 0.0 for every non-held column, so names
        # dropped by the band/hysteresis/cost-gate/max_names logic ACTUALLY exit
        # at this rebalance (ffill between rebalances only carries true holdings).
        w_row = pd.Series(0.0, index=cols)
        w_row.loc[sel] = w.reindex(sel).values
        if h_pick is not None:
            w_row.loc[h_pick] = -beta
        rows[d] = w_row
        prev_hold = set(sel)
        prev_hedge = (h_pick, beta)

    W = pd.DataFrame(rows).T.reindex(columns=cols).sort_index()
    W = W.reindex(idx).ffill().fillna(0.0)   # hold between rebalances only
    W = W.where(px[cols].notna(), 0.0)       # force exit on delisting/missing price
    W_lag = W.shift(1).fillna(0.0)           # decided at close d -> held from d+1

    r_all = rets[cols].fillna(0.0)
    net = net_of_cost(W_lag, r_all, cost_bps=p["base_bps"], name="amihud_illiq_idxhedge_v2")

    # realized self-consistent impact drag at actual account size:
    # cost fraction of book per name = |dW| * (lambda * $T),  $T = account * |dW|
    dW = W_lag[univ].diff().abs().fillna(0.0)
    lam = amihud.shift(1).reindex(idx)
    lam = lam.T.fillna(lam.median(axis=1)).T.fillna(0.0)  # missing lambda -> per-date median
    drag = (lam[univ] * p["account_usd"] * dW.pow(2)).sum(axis=1)
    net = (net - drag.reindex(net.index).fillna(0.0)).rename("amihud_illiq_idxhedge_v2")

    live = W_lag.abs().sum(axis=1)
    if (live > 0).any():
        net = net.loc[live[live > 0].index.min():]

    trades = trades_from_weights(W_lag, r_all, sector_map)
    return net.dropna(), trades


SPEC = StrategySpec(
    id="amihud_illiq_idxhedge_v2",
    family="illiquidity",
    title=("Amihud illiquidity premium v2 — mid-illiquid band long, self-consistent "
           "impact-cost gate, beta-matched IWM/SPY index hedge (retail-deployable)"),
    markets=["US_smallmid_equity"],
    data_desc=("Sharadar SEP closeadj + raw-price (closeunadj/close, adjusted fallback) "
               "x volume dollar-volume, small+mid caps incl. delisted "
               "(survivorship-clean), 2000-present; IWM/SPY SEP series for the hedge "
               "leg. OWNED data only."),
    pre_registration=(
        "HYPOTHESIS: the Amihud illiquidity premium (2002) persists in small/mid US "
        "equities because the cost of harvesting it scales with the illiquidity that "
        "creates it (limits to arbitrage). PRIMARY (frozen): trailing-12m Amihud, "
        "60-80th pct band within size tercile (NOT the extreme tail), sector-capped, "
        "inverse-vol, monthly rebalance with 5pp hysteresis; include a name only if "
        "its OWN modeled round-trip cost (2*8bps + 2*Amihud*$10K) <= 80bps; realized "
        "impact additionally charged at $25K account sizing. Each rebalance emits a "
        "FULL weight vector (explicit zeros), so band/cost-gate exits are real exits "
        "and the long book sums to 1 at every rebalance. SINGLE MUTATION vs the "
        "prior long-short variant: the multi-name liquid short leg (infeasible at "
        "$5-25K: locates/borrow/minimums; ~zero premium by the mechanism's own logic) "
        "is replaced by ONE beta-matched index short (IWM, fallback SPY), trailing-60d "
        "beta, +/-10% drift band, gross <= 2x. DIAGNOSTIC (not a gate): the hedged "
        "long leg must source >=70% of the legacy long-short gross premium, else the "
        "'premium' lived in the short leg and the idea dies. EXPECTED: net Sharpe "
        "0.3-0.6 standalone; premium positive and monotonically weakening toward "
        "liquid names; pro-cyclical left tail (2008/2020) accepted standalone — trend "
        "tail-overlay (<=25%) considered only AFTER holdout+MCPT pass. "
        "GENERALIZATION: same frozen signal+params must be OOS-positive on disjoint "
        "sector slices (fin/RE, defensive/comm, energy/materials); hedge-choice "
        "(IWM vs SPY) must not flip conclusions. Lag discipline: all signals use "
        "data through rebalance close; W is shift(1)-lagged before pricing."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "band_55_75": {"band_lo": 0.55, "band_hi": 0.75},
        "strict_cost": {"gate_trade_usd": 25_000.0, "cost_cap_bps": 60.0},
        "hedge_spy": {"hedge_pref": ("SPY",)},
        "slow_hyst": {"hyst": 0.10},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=60,
)