"""
Liquidity-stress-gated Amihud illiquidity L/S.

Base book (Parent-1, frozen): within-size-tercile Amihud illiquidity sort on
survivorship-clean Sharadar small+mid US equities. Long = most-illiquid quintile
(EW) per size tercile; short = the most-LIQUID names per tercile (deployable short).
Stress gate (Parent-2, frozen flag, REPURPOSED): VVIX/VIX vol-of-vol divergence =
institutional tail-hedge demand spiking before a vol-regime shift, MATCHED to the
illiquidity premium's documented failure mode (flight-to-liquidity). When the flag
fires we de-gross the ENTIRE book (same legs, same relative weights, only overall
SCALE modulated) to GATE_GROSS for a fixed window, then revert.

This is risk-management on ONE premium (FDR family unchanged: illiquidity_premium),
NOT a second premium. No look-ahead: signal+gate are computed on close-of-t data and
held from t+1 via W.shift(1); full round-trip turnover costs hit every de/re-gross.
IWM is declared as a residual hedge SLEEVE on the spec (not in the alpha ledger).
All data owned/free ($0): Sharadar SEP/TICKERS, VVIX via yfinance ^VVIX, VIX via FRED.

COSTS (frozen v3 spec, asymmetric): 60bps round-trip long / 15bps round-trip short
+ 50bps/yr borrow on short notional. RT is modelled as 2x one-way turnover (cost per
one-way |dw| = RT/2), so a full enter->exit round trip pays the stated RT bps.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel, fred_series
from sdk.universe import sector_universe
from sdk.signal_kit import trades_from_weights, pit_panel
import numpy as np, pandas as pd

START = "2006-01-01"
HOLDOUT = "2022-01-01"

# search slice = Parent-1's 5-sector small+mid book
SEARCH_SECTORS = ["Technology", "Healthcare", "Industrials",
                  "Consumer Cyclical", "Financial Services"]

# 3 pre-declared DISJOINT generalization universes (Parent-1's set):
#   gen1/gen2 = different sectors (share NO tickers with search);
#   gen3 = different cap tier (Large; disjoint from Small+Mid).
GEN = {
    "smallmid_energy_materials_utils":
        dict(caps=["Small", "Mid"], sectors=["Energy", "Basic Materials", "Utilities"], top_n=60),
    "smallmid_defensive_comms_re":
        dict(caps=["Small", "Mid"], sectors=["Consumer Defensive", "Communication Services", "Real Estate"], top_n=60),
    "largecap_all_sectors":
        dict(caps=["Large"], sectors=None, top_n=30),
}

# frozen primary params; thresholds inherited from Parent-2, NOT re-searched here.
_DEFAULTS = dict(
    gate_gross=0.0, gate_window=10,            # primary risk-off action: sit flat 10d
    vvix_hi=110.0, vix_lo=18.0, ratio_hi=6.5,  # Parent-2's exact frozen flag
    amihud_lb=21, n_liq_short=15,
    price_min=10.0, price_max=500.0,
    name_cap=0.10, min_names=30,
    # frozen v3 cost model (asymmetric): RT bps per side + short borrow carry
    long_rt_bps=60.0, short_rt_bps=15.0, borrow_bps_yr=50.0,
)


# ---------------------------------------------------------------- universe + panel
def _universe(caps, sectors, top_n_per_sector):
    """Sector-spread, cap-tiered universe + {ticker: sector} via the mandatory kit."""
    tickers, smap = [], {}
    for cap in caps:
        t, s = sector_universe(marketcap=cap, top_n_per_sector=top_n_per_sector)
        for x in t:
            if x not in smap:
                tickers.append(x); smap[x] = s[x]
    if sectors is not None:
        tickers = [x for x in tickers if smap.get(x) in sectors]
        smap = {x: smap[x] for x in tickers}
    return tickers, smap


def _build_panel(caps, sectors, top_n_per_sector):
    tickers, smap = _universe(caps, sectors, top_n_per_sector)
    px = sep_panel(tickers, START, field="closeadj")        # survivorship-clean, delisted incl
    vol = sep_panel(tickers, START, field="volume")
    cols = px.columns.intersection(vol.columns)
    px, vol = px[cols], vol[cols]
    smap = {t: smap[t] for t in cols if t in smap}

    fund = sf1(list(cols), ["marketcap"])                   # datekey-based (no calendardate)
    mcap = pit_panel(fund, "marketcap", px.index, list(cols))

    vvix = yf_panel(["^VVIX"], START).iloc[:, 0].reindex(px.index).ffill()   # vol-of-vol (free)
    vix = fred_series({"VIXCLS": "VIX"}, START)["VIX"].reindex(px.index).ffill()  # spot VIX (free)

    panel = pd.concat({
        "px": px, "vol": vol, "mcap": mcap,
        "vvix": vvix.to_frame("VVIX"), "vix": vix.to_frame("VIX"),
    }, axis=1)
    panel.attrs["sector_map"] = smap   # kit's ledger needs this; carried on attrs
    return panel


def load_data():
    return _build_panel(["Small", "Mid"], SEARCH_SECTORS, top_n_per_sector=60)


def load_gen_data(label):
    cfg = GEN[label]
    return _build_panel(cfg["caps"], cfg["sectors"], cfg["top_n"])


# ---------------------------------------------------------------- signal helpers
def _xsec_weights(a, m, pr, p):
    """One rebalance date: within-size-tercile Amihud sort -> EW long illiquid / short liquid."""
    w = pd.Series(0.0, index=a.index)
    elig = a.notna() & m.notna() & (m > 0) & pr.between(p["price_min"], p["price_max"])
    a, m = a[elig], m[elig]
    if len(a) < p["min_names"]:
        return w
    try:
        terc = pd.qcut(m.rank(method="first"), 3, labels=False)
    except Exception:
        return w
    longs, shorts = [], []
    for t in (0, 1, 2):
        g = a[terc == t]
        if len(g) < 8:
            continue
        thr = g.quantile(0.80)                         # most-illiquid quintile (high Amihud)
        longs += list(g.index[g >= thr])
        shorts += list(g.nsmallest(int(p["n_liq_short"])).index)  # most-liquid (low Amihud)
    longs = list(dict.fromkeys(longs))
    shorts = [s for s in dict.fromkeys(shorts) if s not in longs]
    if longs:
        w.loc[longs] = 0.5 / len(longs)
    if shorts:
        w.loc[shorts] = -0.5 / len(shorts)
    return w.clip(-p["name_cap"], p["name_cap"])         # 10% single-name cap


def _flag_in_window(panel, window=None):
    """Primary stress gate -> bool Series: any flag fire within the last `window` days."""
    window = _DEFAULTS["gate_window"] if window is None else int(window)
    vvix = panel["vvix"]["VVIX"]; vix = panel["vix"]["VIX"]
    f = ((vvix > _DEFAULTS["vvix_hi"]) & (vix < _DEFAULTS["vix_lo"])) | \
        (((vvix / vix) > _DEFAULTS["ratio_hi"]) & (vix < _DEFAULTS["vix_lo"]))
    f = f.reindex(panel.index).fillna(False).astype(float)
    return f.rolling(window, min_periods=1).max().fillna(0.0) > 0.5


def _apply_costs(Wlag, rets, p):
    """Frozen v3 asymmetric cost model: 60bps RT long / 15bps RT short + 50bps/yr borrow.
    RT modelled as 2x one-way turnover (cost per one-way |dw| = RT/2). Borrow charged
    daily on short notional. Costs hit every de-gross / re-gross via the |dw| turnover."""
    gross = (Wlag * rets.reindex(Wlag.index).fillna(0.0)).sum(axis=1)
    wl = Wlag.clip(lower=0.0)   # long leg
    ws = Wlag.clip(upper=0.0)   # short leg (<=0)

    turn_l = wl.diff().abs()
    turn_s = ws.diff().abs()
    if len(wl):
        turn_l.iloc[0] = wl.iloc[0].abs()   # initial entry = full notional
        turn_s.iloc[0] = ws.iloc[0].abs()
    turn_l = turn_l.sum(axis=1)
    turn_s = turn_s.sum(axis=1)

    cost = (turn_l * (p["long_rt_bps"] / 2.0) / 1e4
            + turn_s * (p["short_rt_bps"] / 2.0) / 1e4)
    borrow = ws.abs().sum(axis=1) * (p["borrow_bps_yr"] / 1e4) / 252.0

    dr = (gross - cost - borrow)
    dr.name = "liqgate_amihud"
    return dr


# ---------------------------------------------------------------- signal
def signal(panel, **params):
    p = {**_DEFAULTS, **params}
    px = panel["px"].astype(float)
    vol = panel["vol"].astype(float)
    mcap = panel["mcap"].astype(float)
    vvix = panel["vvix"]["VVIX"]; vix = panel["vix"]["VIX"]
    smap = panel.attrs.get("sector_map", {})

    rets = px.pct_change()
    dollar_vol = (px * vol).replace(0.0, np.nan)          # dollar-volume proxy (adj close)
    amihud = (rets.abs() / dollar_vol).rolling(
        p["amihud_lb"], min_periods=max(5, p["amihud_lb"] // 2)).mean()

    # monthly single-date rebalance (last trading day of each month)
    rebal = pd.Series(px.index, index=px.index).groupby(
        [px.index.year, px.index.month]).last().values

    wmat = {}
    for dt in rebal:
        w = _xsec_weights(amihud.loc[dt], mcap.loc[dt], px.loc[dt], p)
        if w.abs().sum() > 0:
            wmat[pd.Timestamp(dt)] = w
    if not wmat:
        return pd.Series(0.0, index=px.index, name="liqgate_amihud"), []

    W = pd.DataFrame(wmat).T.sort_index().reindex(columns=px.columns).fillna(0.0)
    W = W.reindex(px.index, method="ffill").fillna(0.0)    # hold between rebalances

    # stress gate: scale ENTIRE book's gross to GATE_GROSS during flagged windows
    window = max(1, int(p["gate_window"]))
    f = ((vvix > p["vvix_hi"]) & (vix < p["vix_lo"])) | \
        (((vvix / vix) > p["ratio_hi"]) & (vix < p["vix_lo"]))
    f = f.reindex(px.index).fillna(False).astype(float)
    in_win = (f.rolling(window, min_periods=1).max().fillna(0.0) > 0.5)
    mult = pd.Series(np.where(in_win.values, p["gate_gross"], 1.0), index=px.index)
    W = W.mul(mult, axis=0)

    # ONE-day lag is OUR responsibility (W built from close-of-t data) -> shift(1)
    Wlag = W.shift(1).fillna(0.0)
    dr = _apply_costs(Wlag, rets, p)                       # frozen asymmetric costs + borrow
    trades = trades_from_weights(Wlag, rets, smap)         # kit stamps entry_regime
    return dr, trades


# ---------------------------------------------------------------- machine-checked expectations
def _exp_s1_loss(ctx):
    """S1: ungated book's mean during flagged windows is significantly negative (one-sided)."""
    try:
        u = ctx["grid"].get("ungated")
        if u is None or len(u) == 0:
            return {"pass": False, "observed": "no ungated grid"}
        inwin = _flag_in_window(ctx["panel"]).reindex(u.index).fillna(False)
        fl = u[inwin.values].dropna()
        if len(fl) < 10:
            return {"pass": False, "observed": f"flagged_days={len(fl)}"}
        sd = fl.std(ddof=1)
        t = float(fl.mean() / (sd / np.sqrt(len(fl)))) if sd > 0 else 0.0
        return {"pass": bool(t < -1.64), "observed": round(t, 2)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _exp_s2_placebo(ctx):
    """S2: actual flagged loss in worst 10% of count-matched random-window means (not seasonality)."""
    try:
        u = ctx["grid"].get("ungated")
        if u is None or len(u) == 0:
            return {"pass": False, "observed": "no ungated grid"}
        u = u.dropna()
        inwin = _flag_in_window(ctx["panel"]).reindex(u.index).fillna(False)
        n = int(inwin.sum())
        if n < 10:
            return {"pass": False, "observed": f"flagged_days={n}"}
        actual = float(u[inwin.values].mean())
        rng = np.random.default_rng(7)
        vals = u.values
        sims = np.array([vals[rng.choice(len(vals), size=n, replace=False)].mean()
                         for _ in range(1000)])
        pct = float((sims <= actual).mean())
        return {"pass": bool(pct < 0.10), "observed": round(pct, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _exp_s3_tail(ctx):
    """S3: gated Sharpe >= ungated (within 0.05) AND gated max-drawdown strictly shallower."""
    try:
        g, u = ctx["grid"].get("default"), ctx["grid"].get("ungated")
        if g is None or u is None:
            return {"pass": False, "observed": "missing grid"}
        sh = lambda r: float(r.dropna().mean() / r.dropna().std() * np.sqrt(252)) \
            if r.dropna().std() > 0 else 0.0
        dd = lambda r: float(((1 + r.fillna(0)).cumprod() /
                              (1 + r.fillna(0)).cumprod().cummax() - 1).min())
        sg, su, dg, du = sh(g), sh(u), dd(g), dd(u)
        ok = (sg >= su - 0.05) and (dg > du)
        return {"pass": bool(ok), "observed": f"Sh {sg:.2f}v{su:.2f}; MDD {dg:.1%}v{du:.1%}"}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _exp_s4_attribution(ctx):
    """S4: >=90% of squared gated-vs-ungated return difference lives inside flagged windows."""
    try:
        g, u = ctx["grid"].get("default"), ctx["grid"].get("ungated")
        if g is None or u is None:
            return {"pass": False, "observed": "missing grid"}
        idx = g.index.intersection(u.index)
        inwin = _flag_in_window(ctx["panel"]).reindex(idx).fillna(False).values
        d2 = ((g.reindex(idx) - u.reindex(idx)) ** 2).fillna(0.0).values
        tot = d2.sum()
        share = float(d2[inwin].sum() / tot) if tot > 0 else 1.0
        return {"pass": bool(share > 0.90), "observed": round(share, 3)}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="liqgate_amihud_vvix_v1",
    family="illiquidity_premium",   # gate is risk-mgmt on SAME premium -> no new FDR family
    title="Liquidity-stress-gated Amihud illiquidity L/S (VVIX/VIX vol-of-vol de-gross overlay)",
    markets=["US_equity_smallmid"],
    data_desc=("Sharadar SEP small+mid 5-sector L/S Amihud illiquidity book (long most-illiquid "
               "quintile / short most-liquid per size tercile); VVIX (yfinance ^VVIX) + spot VIX "
               "(FRED VIXCLS) vol-of-vol stress gate; IWM declared residual hedge sleeve."),
    pre_registration=(
        "Illiquidity (Amihud) cross-sectional L/S is compensation for bearing liquidity risk; its "
        "ONE documented failure mode is flight-to-liquidity during vol-regime shifts (long-illiquid "
        "craters faster than short-liquid in a stress sell-off). We OVERLAY a de-grossing gate "
        "MATCHED to that failure mode: flag = (VVIX>110 & VIX<18) OR (VVIX/VIX>6.5 & VIX<18) "
        "(institutional tail-hedge demand spiking before a vol-regime shift). On a flag we scale the "
        "ENTIRE book's gross to GATE_GROSS (primary 0.0 = flat) for a fixed 10-trading-day window, "
        "then revert -- SAME legs, SAME relative weights, only overall SCALE modulated. This is "
        "defensive risk management on ONE premium (FDR family unchanged: illiquidity_premium), NOT a "
        "second premium. Thresholds are inherited frozen from a prior standalone gate study and are "
        "NOT re-searched here. Costs are the frozen v3 model: 60bps RT long / 15bps RT short + "
        "50bps/yr borrow on short notional, charged on every de/re-gross turnover. No look-ahead: "
        "signal+gate computed on close-of-t data, held from t+1 via W.shift(1). Grid {gross "
        "0/0.3/0.5, window 5/10/15, + an explicit ungated reference} declared for honest "
        "effective-N and to power the synergy expectations S1-S4. HONEST CAVEAT (carried from the "
        "parent gate study): divergence episodes are rare and any benefit is concentrated in a few "
        "windows -- statistical power is the binding constraint. If <15 independent flagged episodes "
        "overlap the search window, or the gate never fires in the book's worst-drawdown months, "
        "treat as INCONCLUSIVE rather than a pass. VERDICT: prefer the gated construction for "
        "scale-up ONLY if all gates pass, S1-S4 hold, holdout Sharpe >= Parent-1 (1.273), AND holdout "
        "max-drawdown < the ungated book's; otherwise the ungated book remains deployable and the "
        "gate is banked as negative knowledge. IWM is a declared residual hedge sleeve (cap 0.35), "
        "not in the alpha ledger. Data: Sharadar SEP/TICKERS (owned, survivorship-clean), VVIX via "
        "yfinance ^VVIX, VIX via FRED VIXCLS -- all $0."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},                       # default = primary (gross 0.0, window 10)
    grid={
        "default": {},
        "ungated": {"gate_gross": 1.0},      # reference for S1-S4 (and honest search burden)
        "gross030": {"gate_gross": 0.30},
        "gross050": {"gate_gross": 0.50},
        "win5": {"gate_window": 5},
        "win15": {"gate_window": 15},
    },
    scope="broad",                           # universal liquidity-risk premium -> must generalise
    generalization_universes=list(GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=70,                 # ~25 illiquid longs + ~45 liquid shorts
    hedge_tickers=["IWM"],                   # residual beta trim judged on whitelist+cap, not the book
    hedge_cap=0.35,
    expectations=[
        {"name": "flagged_window_loss",
         "claim": "Ungated book mean daily net return during VVIX/VIX-flagged windows is "
                  "significantly negative (one-sided t<-1.64; daily-autocorrelation caveat) -- "
                  "the gate targets a real loss.",
         "check": _exp_s1_loss},
        {"name": "placebo_specificity",
         "claim": "Actual flagged-window mean loss sits in the worst 10% of count-matched random "
                  "windows -- the gate is not generic seasonality / time-of-month.",
         "check": _exp_s2_placebo},
        {"name": "tail_improved_sharpe_kept",
         "claim": "Gated (primary GATE_GROSS=0) search Sharpe >= ungated (within 0.05) AND gated "
                  "max-drawdown strictly shallower -- tail improved without hurting Sharpe.",
         "check": _exp_s3_tail},
        {"name": "difference_in_windows",
         "claim": ">=90% of squared gated-vs-ungated return difference occurs inside flagged "
                  "windows -- the entire effect is localized to the gate (attribution).",
         "check": _exp_s4_attribution},
    ],
)