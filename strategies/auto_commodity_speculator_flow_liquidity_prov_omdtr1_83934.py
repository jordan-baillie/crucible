# Commodity speculator-flow liquidity-provision premium
# (COT positioning-FLOW reversal, weekly market-neutral cross-section)
#
# Mechanism (Kang-Rouwenhorst-Tang 2020 "second premium"): a SHORT-HORIZON
# liquidity-provision / demand-shock-absorption premium paid to take the other side
# of speculators' positioning *shocks* -- distinct from and orthogonal to the slow
# hedging-pressure LEVEL premium (which the prior cot_hedging_pressure_xs_ls book
# tested on comm_net LEVEL and FAILED). We FADE the recent speculator flow:
# specs just aggressively BOUGHT -> demand overshoot -> we SHORT (provide sell-side
# liquidity); specs just aggressively SOLD -> we LONG (provide buy-side liquidity).
#
# Only NOVEL code here is the signal (flow z + hysteresis state machine). All
# lookahead-sensitive steps use the kit: xs_zscore / net_of_cost / trades_from_weights.

from sdk.harness import StrategySpec
from sdk.adapters import cot_positioning, fut_curve
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------------
# Universe: owned 17-root Databento complex, 4 economic complexes.
# SEARCH on energy+metals (financialised, high speculator participation -> the
# cleanest substrate for a positioning-flow signal). GENERALISE (stage-2, disjoint,
# share NO tickers with search) to the agricultural complexes -> a real LP premium
# must show broadly across untouched complexes; an artifact lives in one lucky one.
# (Cross-ASSET FX/equity positioning generalisation is the longer-run test, pending
# non-gated positioning data -- noted, not faked.)
# ----------------------------------------------------------------------------------
COMPLEXES = {
    "ENERGY":    ["CL", "NG", "HO", "RB"],
    "METALS":    ["GC", "SI", "HG", "PL", "PA"],
    "GRAINS":    ["ZC", "ZS", "ZW", "ZL", "ZM"],
    "LIVESTOCK": ["LE", "HE", "GF"],
}

# Granular (economically real) sub-sectors for the trade ledger's sector-spread gate.
SECTOR_MAP = {
    "CL": "energy_crude", "NG": "energy_gas", "HO": "energy_products", "RB": "energy_products",
    "GC": "metals_precious", "SI": "metals_precious", "PL": "metals_precious", "PA": "metals_precious",
    "HG": "metals_base",
    "ZC": "grains", "ZW": "grains", "ZS": "oilseeds", "ZL": "oilseeds", "ZM": "oilseeds",
    "LE": "livestock", "HE": "livestock", "GF": "livestock",
}

SEARCH_ROOTS = COMPLEXES["ENERGY"] + COMPLEXES["METALS"]          # 9 roots, disjoint from gens
GEN_UNIVERSES = {                                                  # all disjoint from SEARCH_ROOTS
    "grains":    COMPLEXES["GRAINS"],                             # 5
    "livestock": COMPLEXES["LIVESTOCK"],                          # 3 (thin -> noisy, honest)
    "ags_all":   COMPLEXES["GRAINS"] + COMPLEXES["LIVESTOCK"],   # 8 (broader OOS cross-section)
}

NAME = "cot_spec_flow_lp_xs"


# ----------------------------------------------------------------------------------
# Data plumbing (no side effects).
# ----------------------------------------------------------------------------------
def _front_returns(root):
    """Front-month WITHIN-contract daily returns from fut_curve close_1.
    Never diff across a roll: null the return on any contract change (or, if the
    adapter exposes no contract id, conservatively null implausible roll-artifact jumps)."""
    fc = fut_curve(root)
    if isinstance(fc, pd.Series):
        fc = fc.to_frame("close_1")
    fc = fc.sort_index()
    cols = list(fc.columns)
    cclose = next((c for c in ("close_1", "close1", "front", "px_1", "settle_1", "close") if c in cols), cols[0])
    px = pd.to_numeric(fc[cclose], errors="coerce")
    rr = px.pct_change()
    cid = next((c for c in ("contract_1", "symbol_1", "expiry_1", "ticker_1", "contract", "symbol", "expiry")
                if c in cols), None)
    if cid is not None:
        idc = fc[cid].astype(str)
        rr = rr.mask(idc != idc.shift(1))          # within-contract only
    else:
        rr = rr.mask(rr.abs() > 0.5)               # roll-artifact guard (no contract id available)
    return rr.rename(root)


def _cot_cols(cot, r):
    nc = next((cot[c] for c in (f"{r}_noncomm_net", f"{r}_noncommercial_net", f"{r}_spec_net", f"{r}_noncomm")
               if c in cot.columns), None)
    oi = next((cot[c] for c in (f"{r}_oi", f"{r}_open_interest", f"{r}_openint", f"{r}_oint")
               if c in cot.columns), None)
    return nc, oi


def _build_panel(roots):
    """Panel with MultiIndex columns level0 in {ret, nc, oi}, level1 = root.
    COT (weekly, release-date PIT) ffilled onto the daily return index; the weekly
    W-FRI resampling done in signal() guarantees COT data is used no earlier than its
    Friday release regardless of how the adapter indexes it."""
    rets = {}
    for r in roots:
        try:
            s = _front_returns(r)
            if s.notna().sum() > 0:
                rets[r] = s
        except Exception:
            continue
    ret_df = pd.DataFrame(rets).sort_index()
    if ret_df.shape[1] == 0:
        empty = pd.DataFrame()
        return pd.concat({"ret": empty, "nc": empty, "oi": empty}, axis=1)
    idx = ret_df.index

    cot = cot_positioning(roots, start_year=2010)
    cot = cot.sort_index()
    nc_raw, oi_raw = {}, {}
    for r in roots:
        nc, oi = _cot_cols(cot, r)
        if nc is not None:
            nc_raw[r] = pd.to_numeric(nc, errors="coerce")
        if oi is not None:
            oi_raw[r] = pd.to_numeric(oi, errors="coerce")
    nc_df = pd.DataFrame(nc_raw).sort_index()
    oi_df = pd.DataFrame(oi_raw).sort_index()

    full = idx.union(nc_df.index).union(oi_df.index)
    nc_df = nc_df.reindex(full).ffill().reindex(idx)   # release-date carry-forward
    oi_df = oi_df.reindex(full).ffill().reindex(idx)

    cols = [r for r in roots if r in ret_df.columns and r in nc_df.columns and r in oi_df.columns]
    ret_df = ret_df[cols]
    nc_df = nc_df.reindex(columns=cols)
    oi_df = oi_df.reindex(columns=cols)
    return pd.concat({"ret": ret_df, "nc": nc_df, "oi": oi_df}, axis=1).sort_index()


def load_data() -> pd.DataFrame:
    return _build_panel(SEARCH_ROOTS)


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(GEN_UNIVERSES[label])


# ----------------------------------------------------------------------------------
# Signal helpers.
# ----------------------------------------------------------------------------------
def _flow_z_weekly(panel, window):
    """Weekly (W-FRI) cross-sectional z of the speculator positioning FLOW
    = trailing `window`-week change in noncomm_net normalised by current OI.
    W-FRI bucketing enforces release-date PIT (Tuesday data used no earlier than Friday)."""
    ncw = panel["nc"].resample("W-FRI").last()
    oiw = panel["oi"].resample("W-FRI").last()
    floww = ncw.diff(int(window)) / oiw.replace(0, np.nan)
    return xs_zscore(floww)


def _weekly_returns(panel):
    r = panel["ret"].fillna(0.0)
    return (1.0 + r).resample("W-FRI").prod() - 1.0


def _states(zw, enter_q, exit_q, min_hold):
    """Hysteresis + min-hold state machine: +1 LONG (bottom flow tercile),
    -1 SHORT (top flow tercile), 0 flat. Enter at the tercile band, exit at the
    median band (hysteresis) and only after min_hold weeks (caps turnover)."""
    cols = list(zw.columns)
    P = zw.rank(axis=1, pct=True).values
    T, N = P.shape
    out = np.zeros((T, N))
    prev = np.zeros(N, dtype=int)
    held = np.zeros(N, dtype=int)
    lo = 1.0 - enter_q
    lo_exit = 1.0 - exit_q
    for i in range(T):
        for j in range(N):
            p = P[i, j]
            s = prev[j]
            if np.isnan(p):
                ns = 0
            elif s == 0:
                ns = -1 if p >= enter_q else (1 if p <= lo else 0)
            elif held[j] < min_hold:
                ns = s
            elif s == -1:
                ns = 1 if p <= lo else (-1 if p >= exit_q else 0)
            else:  # s == 1
                ns = -1 if p >= enter_q else (1 if p <= lo_exit else 0)
            if ns != 0 and ns == s:
                held[j] += 1
            elif ns != 0:
                held[j] = 1
            else:
                held[j] = 0
            prev[j] = ns
            out[i, j] = ns
    return pd.DataFrame(out, index=zw.index, columns=cols)


def _leg_weights(iv_row, names, gross):
    w = iv_row.reindex(names).astype(float)
    med = np.nanmedian(w.values) if len(w) else np.nan
    w = w.fillna(med if np.isfinite(med) else 1.0).clip(lower=0.0)
    tot = w.sum()
    if not (tot > 0):
        w = pd.Series(1.0, index=names)
        tot = w.sum()
    return gross * w / tot


# ----------------------------------------------------------------------------------
# Signal: contrarian (FADE the flow), market-neutral, per-leg inverse-vol,
# book-level vol-targeted. Frozen across search and all generalisation universes.
# ----------------------------------------------------------------------------------
def signal(panel, **params):
    window   = int(params.get("flow_window_weeks", 4))
    enter_q  = float(params.get("enter_q", 0.667))   # tercile entry band
    exit_q   = float(params.get("exit_q", 0.50))     # median exit band (hysteresis)
    min_hold = int(params.get("min_hold", 1))        # weeks (default = weekly hold)
    vol_lb   = int(params.get("vol_lb", 63))
    tvol     = float(params.get("target_vol", 0.10))
    max_lev  = float(params.get("max_lev", 3.0))
    cost_bps = float(params.get("cost_bps", 8.0))    # conservative for liquid futures turnover

    ret = panel["ret"]
    cols = list(ret.columns)
    if len(cols) < 2:
        return pd.Series(dtype="float64", name=NAME), []

    zw = _flow_z_weekly(panel, window).reindex(columns=cols)
    if zw.dropna(how="all").empty:
        return pd.Series(dtype="float64", name=NAME), []

    state = _states(zw, enter_q, exit_q, min_hold)            # weekly {-1,0,+1}

    # per-leg inverse-vol weights (vol known at the Fri the weight is set -> no lookahead)
    vol_d = ret.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std()
    iv_w = (1.0 / vol_d.resample("W-FRI").last()).replace([np.inf, -np.inf], np.nan)
    iv_w = iv_w.reindex(index=state.index, columns=cols)

    W_w = pd.DataFrame(0.0, index=state.index, columns=cols)
    for dt in state.index:
        s = state.loc[dt]
        iv = iv_w.loc[dt]
        longs = s.index[s > 0]
        shorts = s.index[s < 0]
        if len(longs):
            W_w.loc[dt, longs] = _leg_weights(iv, longs, 0.5)
        if len(shorts):
            W_w.loc[dt, shorts] = _leg_weights(iv, shorts, -0.5)

    # a Friday weight is HELD the following week -> ffill, then book vol-target,
    # then shift(1): the 1-day execution lag is OUR responsibility (stated here).
    W_base = W_w.reindex(ret.index, method="ffill").fillna(0.0)

    base_bk = (W_base.shift(1) * ret).sum(axis=1)            # strictly-trailing book return
    bvol = base_bk.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std() * np.sqrt(252.0)
    scal = (tvol / bvol).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=max_lev)
    scal = scal.resample("W-FRI").last().reindex(ret.index, method="ffill").fillna(0.0)  # weekly only
    W_final = W_base.mul(scal, axis=0)

    W_held = W_final.shift(1).fillna(0.0)                    # <-- execution lag
    ret_f = ret.fillna(0.0)

    daily = net_of_cost(W_held, ret_f, cost_bps=cost_bps, name=NAME)
    trades = trades_from_weights(W_held, ret_f, SECTOR_MAP)  # kit stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------------------------
# Soft expectations (machine-checkable mechanism claims).
# ----------------------------------------------------------------------------------
def _sharpe(s):
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(252.0)) if (len(s) > 50 and s.std() > 0) else float("nan")


def _exp_contrarian_ic(ctx):
    """Mechanism: high flow predicts LOW next-week return -> in-sample cross-sectional
    IC(flow_z_t, ret_{t+1}) must be NEGATIVE. Falsifies a follow/momentum sign."""
    panel = ctx["panel"]
    hs = pd.Timestamp(ctx["holdout_start"])
    zw = _flow_z_weekly(panel, 4)
    fwd = _weekly_returns(panel).shift(-1)
    zw = zw[zw.index < hs]
    fwd = fwd.reindex(zw.index)
    ics = []
    for dt in zw.index:
        a, b = zw.loc[dt], fwd.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() >= 4:
            c = a[m].corr(b[m])
            if pd.notna(c):
                ics.append(c)
    if not ics:
        return {"pass": False, "observed": "n/a"}
    ic = float(np.mean(ics))
    return {"pass": ic < 0.0, "observed": round(ic, 4)}


def _exp_min_hold(ctx):
    """Hysteresis + min-hold cap churn: median trade hold >= 5 trading days (weekly floor)."""
    tr = ctx.get("trades") or []
    holds = [t.get("hold_days", 0) for t in tr if t.get("hold_days") is not None]
    if not holds:
        return {"pass": False, "observed": "n/a"}
    med = float(np.median(holds))
    return {"pass": med >= 5.0, "observed": med}


def _exp_robust_grid(ctx):
    """Not a cherry-picked flow window: every pre-declared grid variant is in-sample
    Sharpe-positive (the premium does not depend on one lucky parameter)."""
    grid = ctx.get("grid") or {}
    hs = pd.Timestamp(ctx["holdout_start"])
    shs = {}
    for lab, ser in grid.items():
        if ser is None:
            continue
        sh = _sharpe(ser[ser.index < hs])
        if np.isfinite(sh):
            shs[lab] = sh
    if not shs:
        return {"pass": False, "observed": "n/a"}
    mn = min(shs.values())
    return {"pass": mn > 0.0, "observed": round(mn, 3)}


# ----------------------------------------------------------------------------------
# Spec.
# ----------------------------------------------------------------------------------
SPEC = StrategySpec(
    id=NAME,
    family="liquidity_provision_positioning_flow",
    title="Commodity speculator-flow liquidity-provision premium (COT positioning-flow reversal, weekly XS)",
    markets=["commodity_futures"],
    data_desc=("Weekly CFTC COT noncommercial(speculator) net positioning + open interest "
               "(cot_positioning, release-date PIT, 2010+) and Databento front-month within-contract "
               "returns (fut_curve close_1, roll returns nulled) for the owned 17-root complex. "
               "Search universe = energy+metals (9 roots); generalisation = agricultural complexes."),
    pre_registration=(
        "PREMIUM: Kang-Rouwenhorst-Tang (2020) 'second premium' -- a short-horizon liquidity-"
        "provision / demand-shock-absorption premium paid to take the other side of speculators' "
        "positioning SHOCKS. Distinct from and orthogonal to the slow hedging-pressure LEVEL premium "
        "(the prior cot_hedging_pressure_xs_ls book tested comm_net LEVEL with a FOLLOW sign and FAILED). "
        "DIFFERENT axis per anti-pattern #3: different variable (Delta noncomm_net flow vs comm_net level), "
        "OPPOSITE sign (FADE vs follow), different horizon (weekly reversal vs slow level). "
        "Not carry/curve, basis-momentum, value, skew, storage or crush.\n"
        "SIGNAL (pre-registered PRIMARY -> default params, no grid cherry-picking): on the weekly COT "
        "release (W-FRI; Tuesday data used no earlier than its Friday release via W-FRI bucketing), per root "
        "flow = Delta_4w(noncomm_net)/OI, cross-sectionally z-scored across roots; CONTRARIAN market-neutral "
        "book -- SHORT the top tercile (specs just bought -> provide sell-side liquidity) and LONG the bottom "
        "tercile (specs just sold -> provide buy-side liquidity), tercile entry with a median-band hysteresis + "
        "min-hold to cap turnover, per-leg inverse-vol (equal-risk), book vol-targeted to 10%. Returns are "
        "front-month close_1 WITHIN-contract only (roll returns nulled). 1-day execution lag applied (W.shift(1)).\n"
        "COSTS: 8bps on turnover (conservative for liquid futures; real round-trip ~1-2bps).\n"
        "SCOPE/GENERALISATION (broad): the LP/positioning-flow reversal is a universal mechanism (documented "
        "in equity short-term reversal, FX and commodities). Search on the financialised energy+metals complex; "
        "the frozen signal+default params are run OOS on DISJOINT agricultural complexes (grains, livestock, "
        "ags_all) -- a real premium is broadly-positive across untouched complexes, an artifact lives in one. "
        "HONEST LIMITATION: commodities is a thin substrate (17 roots); livestock (3 roots) is a noisy 1-vs-1 "
        "cross-section, and the cross-ASSET (FX/equity positioning) generalisation is the longer-run test "
        "pending non-gated positioning data -- not faked here.\n"
        "STANDALONE FIRST: validated per the 2026-06-08 lesson; the boreas trend tail-overlay (LP is pro-"
        "cyclical -> opposite tail to trend) is DEFERRED to a separate sized overlay step, not blended 50/50 here.\n"
        "FALSIFIERS: in-sample contrarian IC not negative; grid not uniformly positive; sub-complex sign flips; "
        "MCPT/absolute-null shows a bid-ask/construction artifact -> reject."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"flow_window_weeks": 4, "enter_q": 0.667, "exit_q": 0.50,
                    "min_hold": 1, "vol_lb": 63, "target_vol": 0.10, "max_lev": 3.0, "cost_bps": 8.0},
    grid={
        "default": {},
        "flow_2w": {"flow_window_weeks": 2},
        "flow_6w": {"flow_window_weeks": 6},
        "min_hold_2w": {"min_hold": 2},
        "quartile": {"enter_q": 0.75},
    },
    scope="broad",
    generalization_universes=["grains", "livestock", "ags_all"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=12,
    expectations=[
        {"name": "contrarian_flow_ic",
         "claim": "in-sample cross-sectional IC(flow_z_t, next-week return) < 0 (we FADE flow)",
         "check": _exp_contrarian_ic},
        {"name": "min_hold_caps_churn",
         "claim": "median trade hold_days >= 5 (weekly floor + hysteresis cap turnover)",
         "check": _exp_min_hold},
        {"name": "robust_across_grid",
         "claim": "every declared grid variant is in-sample Sharpe-positive (no lucky window)",
         "check": _exp_robust_grid},
    ],
)