# Grain inventory-surprise risk-transfer premium  (USDA Grain Stocks surprise x COT commercial confirmation)
# ---------------------------------------------------------------------------------------------------------
# Strategy module for the Crucible harness.  NO side effects (no writes / capital / config).
#
# ADAPTER NOTE (honest flag): besides the generic kit, this module depends on FOUR owned/free
# DOMAIN adapters that the proposal asserts are live-verified (gate0_data_check):
#     usda_nass, cot_positioning, fut_curve   (owned)   +   eia_series   (keyed 2026-06-16, energy gen leg).
# These are NOT in the generic "tested imports" line; they are the proposal-specific data layer.
# Their exact column names are not fully documented here, so every adapter read is shape-normalised
# defensively (see _first_col / _fut_returns / _comm / _eia_stocks / _usda_stocks).  If a signature
# differs, the operator adjusts the read in ONE place — the novel code (the signal) is unaffected.
#
# Mechanism: around a stock report, commercial hedgers shed scarcity/glut risk.  We compute a PIT
# inventory SURPRISE (reported stocks vs a MODELED seasonal+AR(1) expectation fit only on prior
# releases — no consensus feed is owned), build a market-neutral cross-section (long most-negative
# surprise = scarcer-than-priced, short most-positive = glut), and KEEP a leg only if the COT
# commercial net/OI sits on the OPPOSITE side (they are paying us to take the risk).  That gate is
# what turns raw underreaction into a risk-TRANSFER premium.  PIT throughout; enter the day AFTER
# release; fixed 20-day hold; within-contract futures returns; market-neutral -> absolute (MCPT) null.

import re
import numpy as np, pandas as pd

from sdk.harness import StrategySpec
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
from sdk.adapters import usda_nass, cot_positioning, fut_curve, eia_series  # owned/free domain adapters

SPEC_ID = "grain_inv_surprise_cot_v1"
_START = "2011-06-01"

# search (deployable) universe = grains; ZL/ZM inherit the SOYBEAN stocks surprise (no separate
# Grain Stocks series exists for products) but carry their OWN COT gate + returns + vol -> effective
# independent surprise SIGNALS = 3 (corn/wheat/soy), 5 distinct contracts dilute single-name share.
_GRAIN_STOCKS = {"ZC": "CORN", "ZS": "SOYBEANS", "ZW": "WHEAT", "ZL": "SOYBEANS", "ZM": "SOYBEANS"}

# energy generalization (breadth) universe = petroleum complex (EIA weekly ending stocks).
# Natural gas (NG) is EXCLUDED on purpose: it releases on a different day (Thu vs Wed) so it cannot
# join a same-date PIT cross-section.  The 3 gen slices below are all DISJOINT from the grain search
# universe (the binding requirement); they overlap each other because petroleum has only one true
# complex -> the decisive breadth arbiter is grains-vs-petroleum, the slices probe within-complex robustness.
_EIA_STOCKS = {"CL": "WCESTUS1", "HO": "WDISTUS1", "RB": "WGTSTUS1"}  # crude / distillate / gasoline
_GEN = {
    "energy_petroleum":          ["CL", "HO", "RB"],
    "energy_crude_distillate":   ["CL", "HO"],
    "energy_distillate_gasoline":["HO", "RB"],
}


# ----------------------------- defensive adapter reads -----------------------------
def _first_col(df, names, required=True):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if required:
        raise KeyError(f"none of {names} found in columns {list(df.columns)}")
    return None


def _call(fn, *a, **k):
    """Call adapter; if it rejects the start kwarg, retry without it (signature robustness)."""
    try:
        return fn(*a, **k)
    except TypeError:
        k.pop("start", None)
        return fn(*a, **k)


def _maybe_ret(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.pct_change() if (len(s) and s.abs().median() > 1.5) else s  # price -> return; else already a return


def _fut_returns(root, start=_START):
    """Daily WITHIN-CONTRACT return series for one futures root (never diff across a roll)."""
    obj = _call(fut_curve, root, start=start)
    if isinstance(obj, pd.Series):
        s = obj.copy(); s.index = pd.to_datetime(s.index)
        return _maybe_ret(s).sort_index()
    df = pd.DataFrame(obj); df.index = pd.to_datetime(df.index)
    rc = _first_col(df, ["ret", "return", "wc_ret", "ret_front", "c1_ret", "front_ret"], required=False)
    if rc:
        return pd.to_numeric(df[rc], errors="coerce").sort_index()
    pc = _first_col(df, ["close_1", "settle_1", "close", "settle", "price", "c1", "front"], required=True)
    px = pd.to_numeric(df[pc], errors="coerce")
    r = px.pct_change()
    cid = _first_col(df, ["contract", "contract_id", "symbol_1", "front_symbol", "expiry_1"], required=False)
    if cid:                                            # zero the artificial jump on a roll day
        r = r.mask(df[cid].astype(str) != df[cid].astype(str).shift(1), 0.0)
    else:
        r = r.clip(-0.25, 0.25)                        # crude guard vs roll-jump artifacts if no contract id
    return r.sort_index()


def _comm(root, start=_START):
    """Commercial net positioning / open interest, indexed by COT Friday RELEASE date (PIT)."""
    obj = _call(cot_positioning, root, start=start)
    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce"); s.index = pd.to_datetime(s.index)
        return s.sort_index()
    df = pd.DataFrame(obj); df.index = pd.to_datetime(df.index)
    c = _first_col(df, ["comm_net_oi", "commercial_net_oi", "comm_net_pct_oi", "comm_net_frac"], required=False)
    if c:
        return pd.to_numeric(df[c], errors="coerce").sort_index()
    nl = _first_col(df, ["comm_net", "commercial_net"], required=False)
    if nl:
        net = pd.to_numeric(df[nl], errors="coerce")
    else:
        L = _first_col(df, ["comm_long", "commercial_long", "comm_positions_long"], required=True)
        Sh = _first_col(df, ["comm_short", "commercial_short", "comm_positions_short"], required=True)
        net = pd.to_numeric(df[L], errors="coerce") - pd.to_numeric(df[Sh], errors="coerce")
    oic = _first_col(df, ["oi", "open_interest", "openinterest", "oi_all"], required=False)
    if oic:
        net = net / pd.to_numeric(df[oic], errors="coerce").replace(0, np.nan)
    return net.sort_index()


# ----------------------------- USDA release-date PROXY (QuickStats carries no publish date) ---------
_REF_MONTH = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _usda_release_date(year, ref_desc):
    """USDA NASS QuickStats does NOT carry the publication date — only the survey reference period
    (e.g. 'FIRST OF MAR') + calendar year.  Quarterly Grain Stocks releases land at the END of the
    reference month for Mar/Jun/Sep, and ~mid-January of the FOLLOWING year for the Dec-1 stocks.
    We map (year, ref) -> a CONSERVATIVE release-date proxy that is on-or-after the true release
    (never before) so the panel is strictly PIT-safe; entry is then the day AFTER this proxy.
    The \\b...\\b match also avoids false hits like 'MARKETING YEAR' (contains 'MAR')."""
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return pd.NaT
    m = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", str(ref_desc).upper())
    if not m:
        return pd.NaT
    mon = _REF_MONTH[m.group(1)]
    if mon == 12:                                    # Dec-1 stocks -> released ~Jan 10-12 of next year
        return pd.Timestamp(year=y + 1, month=1, day=15)
    return pd.Timestamp(year=y, month=mon, day=1) + pd.offsets.MonthEnd(0)   # end of reference month


def _usda_stocks(commodity):
    """National quarterly stocks (TOTAL = on-farm + off-farm) indexed by the PIT release-date PROXY
    derived from (year, reference_period_desc) — never the survey reference period as the as-of date.
    Total per period = MAX across the national rows (the total >= any on-farm / off-farm / by-class
    component, all reported in the same bushel unit), which avoids fragile short_desc string filtering."""
    df = pd.DataFrame(usda_nass(commodity, statisticcat_desc="STOCKS", agg_level_desc="NATIONAL"))
    if df.empty:
        return pd.Series(dtype=float)
    vcol = _first_col(df, ["Value", "value", "VALUE"])
    ycol = _first_col(df, ["year", "Year", "YEAR"])
    rcol = _first_col(df, ["reference_period_desc", "reference_period", "ref_period_desc"])
    val = pd.to_numeric(
        df[vcol].astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce").to_numpy()
    rel = pd.to_datetime(
        pd.Series([_usda_release_date(y, r) for y, r in zip(df[ycol], df[rcol])]),
        errors="coerce").to_numpy()
    tmp = pd.DataFrame({"rel": rel, "val": val}).dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    s = tmp.groupby("rel")["val"].max().sort_index()       # total stocks = max component per release
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")]


def _eia_stocks(series_id):
    """Weekly ending stocks indexed by REFERENCE date (publication lag added at placement time)."""
    try:
        obj = eia_series({series_id: "val"}, start="2009-06-01")
    except TypeError:
        try:
            obj = eia_series(series_id, start="2009-06-01")
        except TypeError:
            obj = eia_series(series_id)
    if isinstance(obj, pd.Series):
        s = obj
    else:
        df = pd.DataFrame(obj)
        s = df["val"] if "val" in df.columns else df[df.columns[-1]]
    s = pd.to_numeric(s, errors="coerce").dropna(); s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


# ----------------------------- surprise model (PIT, modeled expectation) -----------------------------
def _surprise_model(stocks, min_obs, std_min):
    """Standardized inventory surprise = log(stocks) - E[log stocks], E = monthly-seasonal + AR(1)
    fit ONLY on releases strictly before each release (no look-ahead); then z-scored by expanding
    PAST surprise mean/std.  Negative surprise => stocks below expectation => scarcer than priced."""
    s = stocks.dropna(); s = s[s > 0]
    if len(s) < min_obs + 2:
        return pd.Series(dtype=float)
    y = np.log(s.values.astype(float)); idx = s.index
    months = np.array([d.month for d in idx])
    raw = np.full(len(y), np.nan)
    for i in range(min_obs, len(y)):
        hy, hm, mi = y[:i], months[:i], months[i]
        seas = {m: hy[hm == m].mean() for m in np.unique(hm)}
        if mi not in seas:
            continue
        des = hy - np.array([seas[m] for m in hm])             # de-seasonalised history
        if len(des) >= 5 and np.std(des[:-1]) > 1e-9:
            b, a = np.polyfit(des[:-1], des[1:], 1)             # AR(1): x_t ~ a + b*x_{t-1}
        else:
            b, a = 0.0, float(np.mean(des))
        raw[i] = y[i] - ((a + b * des[-1]) + seas[mi])          # actual - expected
    z = np.full(len(y), np.nan)
    for i in range(len(y)):
        past = raw[:i][~np.isnan(raw[:i])]
        if len(past) >= std_min and np.std(past) > 1e-12:
            z[i] = (raw[i] - np.mean(past)) / np.std(past)
    return pd.Series(z, index=idx).dropna()


def _surprise_root(root, complex_):
    if complex_ == "grain":
        return _surprise_model(_usda_stocks(_GRAIN_STOCKS[root]), min_obs=8, std_min=5), 0
    return _surprise_model(_eia_stocks(_EIA_STOCKS[root]), min_obs=52, std_min=26), 5  # EIA ~5 bday pub lag


def _place_on_grid(rel_series, tindex, lag_bdays=0):
    """Stamp each release value on the first TRADING date >= (release date + publication lag)."""
    out = pd.Series(np.nan, index=tindex)
    for rd, v in rel_series.items():
        kd = (pd.Timestamp(rd) + pd.tseries.offsets.BDay(lag_bdays)) if lag_bdays else pd.Timestamp(rd)
        loc = tindex.searchsorted(kd, side="left")
        if loc < len(tindex):
            out.iloc[loc] = v
    return out


# ----------------------------- panel assembly -----------------------------
def _build_panel(roots, complex_, start=_START):
    rets = {r: _fut_returns(r, start) for r in roots}
    ret_df = pd.DataFrame(rets)
    ret_df = ret_df[~ret_df.index.duplicated(keep="last")].sort_index().dropna(how="all")
    ret_df = ret_df[ret_df.index >= pd.Timestamp(start)]
    tindex = ret_df.index
    surp, comm = {}, {}
    for r in roots:
        sser, lag = _surprise_root(r, complex_)
        surp[r] = _place_on_grid(sser, tindex, lag_bdays=lag)
        cser = _comm(r, start)
        comm[r] = cser.reindex(tindex.union(cser.index)).sort_index().ffill().reindex(tindex)
    surp_df = pd.DataFrame(surp).reindex(columns=roots)
    comm_df = pd.DataFrame(comm).reindex(columns=roots)
    panel = pd.concat({"ret": ret_df[roots], "surprise": surp_df, "comm": comm_df}, axis=1)
    return panel.sort_index()


def load_data():
    return _build_panel(["ZC", "ZS", "ZW", "ZL", "ZM"], "grain")


def load_gen_data(label):
    return _build_panel(_GEN[label], "energy")


# ----------------------------- the signal (the only novel code) -----------------------------
def _sector_map(roots):
    base = {"ZC": "grains", "ZW": "grains", "ZS": "oilseeds", "ZL": "oilseeds", "ZM": "oilseeds",
            "CL": "energy", "HO": "energy", "RB": "energy"}
    return {r: base.get(r, "commodity") for r in roots}


def _inv_vol_side(names, vt, sgn):
    iv = {r: 1.0 / vt.get(r, np.nan) for r in names if np.isfinite(vt.get(r, np.nan)) and vt.get(r, 0) > 0}
    tot = sum(iv.values())
    return {} if tot <= 0 else {r: sgn * (iv[r] / tot) for r in iv}


def signal(panel, **params):
    H         = int(params.get("hold_days", 20))
    use_gate  = bool(params.get("cot_gate", True))
    target_v  = float(params.get("target_vol", 0.10))
    cost_bps  = float(params.get("cost_bps", 8.0))     # ~3-5bps comm/fees + ~1 tick slippage on grains
    name      = str(params.get("name", SPEC_ID))

    rets = panel["ret"].astype(float)
    surp = panel["surprise"].astype(float)
    comm = panel["comm"].astype(float).ffill()
    roots, dates = list(rets.columns), rets.index
    posmap = {d: i for i, d in enumerate(dates)}

    vol = rets.rolling(60, min_periods=20).std().shift(1)     # trailing, lagged -> PIT inverse-vol sizing

    W = pd.DataFrame(0.0, index=dates, columns=roots)
    for t in surp.dropna(how="all").index:                    # one event per report release date
        s = surp.loc[t].dropna()
        if s.shape[0] < 2:
            continue
        z = xs_zscore(s.to_frame().T).iloc[0]                 # cross-sectional surprise z across roots
        ct = comm.loc[t]
        longs, shorts = [], []
        for r in s.index:
            zr = z.get(r, np.nan)
            if not np.isfinite(zr) or zr == 0:
                continue
            side = 1 if zr < 0 else -1                        # below-avg surprise (scarce) -> long
            if use_gate:                                      # RISK-TRANSFER confirmation gate
                cr = ct.get(r, np.nan)
                if not np.isfinite(cr):
                    continue
                if side == 1 and not (cr < 0):                # long requires commercials NET-SHORT
                    continue
                if side == -1 and not (cr > 0):               # short requires commercials NET-LONG
                    continue
            (longs if side == 1 else shorts).append(r)
        if not longs or not shorts:                           # market-neutral: need both legs
            continue
        vt = vol.loc[t]
        wl, ws = _inv_vol_side(longs, vt, +1.0), _inv_vol_side(shorts, vt, -1.0)
        if not wl or not ws:
            continue
        i0 = posmap[t] + 1                                    # ENTER the day AFTER release (no look-ahead)
        if i0 >= len(dates):
            continue
        hold = dates[i0:i0 + H]
        for r, wv in {**wl, **ws}.items():
            W.loc[hold, r] = W.loc[hold, r] + wv              # overlapping cohorts sum (energy weekly)

    if float(W.abs().to_numpy().sum()) == 0.0:
        return pd.Series(0.0, index=dates, name=name), []

    rets_f = rets.fillna(0.0)
    raw = (W.shift(1).fillna(0.0) * rets_f).sum(axis=1)       # pre-scale book return (for vol estimate)
    rv = raw.rolling(126, min_periods=30).std().shift(1)      # trailing, lagged -> PIT vol scaler
    scal = (target_v / np.sqrt(252) / rv).replace([np.inf, -np.inf], np.nan).clip(0.2, 5.0).fillna(1.0)
    Wt = W.mul(scal, axis=0)                                  # scale toward 10% annual vol target

    Wlag = Wt.shift(1).fillna(0.0)                            # weights built same-day -> lag is OURS
    daily = net_of_cost(Wlag, rets_f, cost_bps=cost_bps, name=name)
    trades = trades_from_weights(Wlag, rets_f, _sector_map(roots))   # kit stamps entry_regime
    return daily, trades


# ----------------------------- soft (machine-checkable) expectations -----------------------------
def _sharpe(s):
    if s is None:
        return 0.0
    s = pd.Series(s).dropna()
    sd = s.std()
    return 0.0 if (len(s) < 20 or not np.isfinite(sd) or sd == 0) else float(s.mean() / sd * np.sqrt(252))


def _chk_gate_adds_value(ctx):
    g = ctx.get("grid", {})
    sd, sn = _sharpe(g.get("default")), _sharpe(g.get("no_gate"))
    return {"pass": bool(sd >= sn - 1e-6), "observed": f"gated_sharpe={sd:.2f} ungated_sharpe={sn:.2f}"}


def _chk_vol_target(ctx):
    s = pd.Series(ctx.get("search")).dropna()
    av = float(s.std() * np.sqrt(252)) if len(s) > 20 else float("nan")
    return {"pass": bool(np.isfinite(av) and 0.04 <= av <= 0.20), "observed": round(av, 4)}


_PRE_REG = (
    "HYPOTHESIS: a fundamental inventory-surprise RISK-TRANSFER premium. At each USDA Grain Stocks "
    "release we measure a PIT inventory SURPRISE (reported national stocks vs a MODELED expectation), "
    "go LONG the most-negative surprise (scarcer than priced) and SHORT the most-positive (glut), and "
    "KEEP a leg ONLY when COT commercial net/OI sits on the OPPOSITE side (commercials net-short a scarce "
    "name = they pay us to take the long). The COT gate is what elevates underreaction to a risk-transfer "
    "premium; soft-check cot_gate_adds_value falsifies that gated Sharpe >= ungated. "
    "EXPECTATION CAVEAT: no analyst-consensus stocks feed is owned, so E[stocks] is MODELED = monthly "
    "seasonal mean + AR(1) on de-seasonalised log stocks, fit ONLY on releases strictly before each "
    "release; pre-registered and NEVER re-fit post-hoc. "
    "PIT DISCIPLINE: USDA QuickStats carries NO publish date, only the survey reference period + year, so "
    "we condition on a CONSERVATIVE release-date PROXY (end of reference month for Mar/Jun/Sep stocks; "
    "~Jan 15 of the next year for Dec-1 stocks) that is on-or-AFTER the true release (never before), and "
    "ENTER at the close the day AFTER that proxy; COT indexed by the FRIDAY release date (never the Tuesday "
    "data date), ffilled; returns computed WITHIN a single contract month via fut_curve (never diff "
    "close_1 across a roll); EIA energy stocks lagged ~5 business days for publication. "
    "SIZING: equal-risk legs (inverse 60d vol), market-neutral both sides, scaled toward 10% annual vol "
    "(trailing, lagged); ~8bps cost (commission + ~1 tick slippage); flat between events; fixed 20d hold. "
    "SCOPE = broad: inventory-surprise drift is a universal storage-theory mechanism, so it must "
    "GENERALISE. Deployable book = grains (ZC/ZS/ZW + small ZL/ZM). Breadth test = EIA petroleum "
    "(CL/HO/RB) on holdout; the decisive arbiter is grains-vs-petroleum (a one-complex pass is a "
    "non-generalising outlier, do not deploy). NG is excluded (separate release day breaks a same-date "
    "cross-section). ZL/ZM inherit the SOYBEAN stocks surprise (no separate Grain Stocks series exists "
    "for products) but carry their OWN COT gate, returns and vol, so they remain effectively independent "
    "legs while the deployable surprise SIGNALS reduce to 3 (corn/wheat/soy). No look-ahead, no consensus "
    "feed, no post-hoc re-fit; the harness runs all rails on frozen default params."
)


# ----------------------------- spec -----------------------------
_DEFAULT_PARAMS = {}
_GRID = {
    "default": {},                       # primary
    "no_gate": {"cot_gate": False},      # ungated comparator for cot_gate_adds_value soft-check
    "hold15":  {"hold_days": 15},        # search-burden variants (DSR effective-N)
    "hold30":  {"hold_days": 30},
}

SPEC = StrategySpec(
    id=SPEC_ID,
    family="commodity_inventory_surprise",
    title="Grain inventory-surprise risk-transfer premium (USDA Grain Stocks surprise x COT commercial confirmation)",
    markets=["grains_futures", "oilseed_futures", "energy_futures"],
    data_desc=("OWNED fut_curve within-contract futures returns (ZC/ZS/ZW/ZL/ZM grains; CL/HO/RB petroleum) "
               "+ usda_nass quarterly Grain Stocks (PIT release-date PROXY from reference period+year) + "
               "cot_positioning commercial net/OI (Friday release date); EIA petroleum weekly ending stocks "
               "(eia_series, +5bday pub lag) for the breadth battery. All point-in-time, modeled-expectation "
               "surprise (no consensus feed)."),
    pre_registration=_PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params=_DEFAULT_PARAMS,
    grid=_GRID,
    scope="broad",
    generalization_universes=list(_GEN.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=6,
    expectations=[
        {"name": "cot_gate_adds_value",
         "claim": "gated (default) Sharpe >= ungated (no_gate) Sharpe on the search window",
         "check": _chk_gate_adds_value},
        {"name": "vol_target_in_band",
         "claim": "realized annualized vol of search-window net returns within [4%, 20%]",
         "check": _chk_vol_target},
    ],
)