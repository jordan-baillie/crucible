import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights

# ============================================================================
# Energy inventory-SURPRISE storage-state risk premium  (petroleum complex).
#
# OWNED/FREE data only, NO external side effects:
#   * EIA weekly petroleum STOCKS via FRED mirrors (fred_series) -> the surprise
#   * front-month energy futures (yf_panel)                      -> tradable returns
#
# The ONLY novel code here is the PIT surprise model + the L/S construction;
# costs / regime-stamping / cross-section z are delegated to the kit.
# ============================================================================
STRAT_ID = "energy_inv_surprise_storage_v1"
START    = "2010-01-01"

# Tradable petroleum complex (crude -> products: a refining/storage chain that
# shares infrastructure -> a coherent cross-section).  Natural gas is NOT here:
# its EIA weekly working-gas storage series is not reachable through fred_series
# in this import surface (needs the eia_series adapter); grains/metals likewise
# need usda_nass -- see the scope note in pre_registration.
ROOTS      = ["CL", "RB", "HO"]
FUT        = {"CL": "CL=F", "RB": "RB=F", "HO": "HO=F"}              # yf front-month
FRED_MAP   = {"WCESTUS1": "CL", "WGTSTUS1": "RB", "WDISTUS1": "HO"}  # EIA weekly stocks on FRED
SECTOR_MAP = {"CL": "Crude", "RB": "Gasoline", "HO": "Distillate"}  # distinct -> sector spread

SURP_LB          = 52    # weeks of PAST surprises for the standardising sigma
RELEASE_LAG_DAYS = 6     # week-ending(Fri)+6 -> strictly AFTER Wed (or holiday-Thu) EIA release
SURP_CLIP        = 4.0   # cap standardised-surprise magnitude


# ---------------------------------------------------------------------------
def _surprise(level):
    """PIT standardised inventory-CHANGE surprise for one root.

    expected weekly change = expanding week-of-year SEASONAL mean
                           + expanding AR(1) on the detrended change,
    fit ONLY on weeks strictly BEFORE each release (no look-ahead).
    surprise = (actual change - expected) / rolling sigma of PAST surprises.
    Tighter-than-expected (draw) -> negative surprise; gluttier (build) -> positive.
    """
    lvl = pd.Series(level).dropna().sort_index()
    if lvl.empty:
        return pd.Series(dtype=float)
    lvl = lvl.resample("W-FRI").last().dropna()       # robust to daily-ffilled FRED output
    chg = lvl.diff()
    if chg.dropna().empty:
        return pd.Series(dtype=float)
    woy = chg.index.isocalendar().week.astype(int).values
    vals = chg.values
    n = len(chg)
    exp = np.full(n, np.nan)
    wsum, wcnt = {}, {}
    s_xy = s_xx = 0.0
    prev_d = np.nan
    run_sum = 0.0
    run_cnt = 0
    for i in range(n):
        w = woy[i]
        if wcnt.get(w, 0) >= 2:                         # seasonal mean from PAST same-week obs
            seas = wsum[w] / wcnt[w]
        else:
            seas = (run_sum / run_cnt) if run_cnt > 0 else 0.0
        phi = (s_xy / s_xx) if s_xx > 1e-9 else 0.0     # AR(1) from PAST detrended pairs
        phi = max(min(phi, 0.95), -0.95)
        ar = phi * prev_d if not np.isnan(prev_d) else 0.0
        exp[i] = seas + ar
        a = vals[i]                                     # reveal AFTER expectation is formed
        if not np.isnan(a):
            wsum[w] = wsum.get(w, 0.0) + a
            wcnt[w] = wcnt.get(w, 0) + 1
            run_sum += a
            run_cnt += 1
            d = a - seas
            if not np.isnan(prev_d):
                s_xy += prev_d * d
                s_xx += prev_d * prev_d
            prev_d = d
    surprise = chg - pd.Series(exp, index=chg.index)
    sig = surprise.shift(1).rolling(SURP_LB, min_periods=16).std()   # PAST sigma only
    return (surprise / sig).clip(-SURP_CLIP, SURP_CLIP).dropna()


# ---------------------------------------------------------------------------
def _fred_one(fred_id, col, start):
    """Fetch ONE FRED series, isolated so a single bad/unmirrored id can't 400 the
    whole batch (the original combined fred_series(FRED_MAP,...) request was the
    HTTP-400 failure point). Returns a Series or None."""
    try:
        df = fred_series({fred_id: col}, start)
    except Exception:
        return None
    if df is None or col not in getattr(df, "columns", []):
        return None
    s = df[col].dropna()
    return s if len(s) else None


# ---------------------------------------------------------------------------
def load_data():
    # tradable returns: continuous front-month futures
    px = yf_panel(list(FUT.values()), START)
    px = px.rename(columns={v: k for k, v in FUT.items()})
    px = px[[r for r in ROOTS if r in px.columns]]
    px = px.where(px > 0)                                 # drop Apr-2020 negative-crude artifact
    rets = px.pct_change().clip(-0.5, 0.5)                # kill zero-cross / roll-gap artifacts
    rets.index = pd.to_datetime(rets.index)

    # inventory surprise, aligned to the EIA RELEASE date (PIT), ffilled to daily.
    # Fetch each EIA/FRED stock series INDIVIDUALLY (the batched call 400'd).
    surp = {}
    for fid, r in FRED_MAP.items():
        if r not in rets.columns:
            continue
        ser = _fred_one(fid, r, START)
        if ser is None:
            continue
        sw = _surprise(ser).copy()
        if sw.empty:
            continue
        sw.index = pd.to_datetime(sw.index) + pd.Timedelta(days=RELEASE_LAG_DAYS)
        sw = sw[~sw.index.duplicated(keep="last")].sort_index()
        surp[r] = sw.reindex(rets.index, method="ffill")  # known only on/after release date

    # roots that have BOTH a tradable return panel and a reachable inventory surprise
    roots = [r for r in ROOTS if r in rets.columns and r in surp]
    if len(roots) < 2:                                   # degrade safely, never crash:
        roots = [r for r in ROOTS if r in rets.columns] or list(rets.columns)
    rets = rets.reindex(columns=roots)
    surp = pd.DataFrame(
        {r: surp.get(r, pd.Series(index=rets.index, dtype=float)) for r in roots}
    )[roots]

    return pd.concat({"ret": rets, "surp": surp}, axis=1)


# ---------------------------------------------------------------------------
def signal(panel, band=0.0, vol_lb=60, target_vol=0.10, max_lev=3.0, flip=False, **kw):
    rets = panel["ret"].copy()
    surp = panel["surp"].reindex(columns=rets.columns)

    sgn = 1.0 if flip else -1.0                           # PRIMARY = -s (tight->long, glut->short)
    tilt = sgn * surp
    if band and band > 0:
        tilt = tilt.where(surp.abs() >= band, 0.0)        # hysteresis dead-zone (throttle)
    tilt = tilt.sub(tilt.mean(axis=1), axis=0)            # dollar-neutral cross-section

    vol = rets.rolling(vol_lb, min_periods=20).std().shift(1)   # LAGGED -> no look-ahead
    pos = tilt / vol.replace(0.0, np.nan)                 # inverse-vol (equal-risk) weighting
    pos = pos.sub(pos.mean(axis=1), axis=0)               # restore neutrality after inv-vol
    gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
    w = pos.div(gross, axis=0).fillna(0.0)                # gross-normalised L/S weights

    # weekly rebalance: freeze the book on a weekly grid, hold until the next release
    w_wk = w.resample("W-WED").last().reindex(w.index, method="ffill").fillna(0.0)

    # vol-target ~target_vol ann. on TRAILING (lagged) realised vol of the unit book
    unit = (w_wk.shift(1) * rets).sum(axis=1)
    rvol = unit.rolling(vol_lb, min_periods=20).std().shift(1) * np.sqrt(252.0)
    lev = (target_vol / rvol).clip(lower=0.0, upper=max_lev).fillna(0.0)
    W = w_wk.mul(lev, axis=0)

    # execution lag: weights built on info<=t are applied to t+1 (my responsibility, stated here)
    Wlag = W.shift(1).fillna(0.0)
    daily = net_of_cost(Wlag, rets, cost_bps=8.0, name=STRAT_ID)   # ~8bps/turnover (conservative for futures)
    trades = trades_from_weights(Wlag, rets, SECTOR_MAP)           # regime-stamped by the kit
    return daily, trades


# ---------------------------------------------------------------------------
def load_gen_data(label):
    # scope='local' -> the harness does NOT run the stage-2 breadth battery.
    # A genuine BROAD test (grain stocks ZC/ZS/ZW via USDA quarterly, or natgas via
    # the EIA weekly working-gas storage series) requires usda_nass / eia_series
    # adapters that are NOT in this module's allowed import surface. Returning the
    # base panel keeps the symbol callable; it is unused for local scope.
    return load_data()


# ---------- soft-expectation checks (machine-checkable mechanism claims) ----
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(252.0))


def _chk_direction(ctx):
    # the storage DIRECTION (-s) must beat the sign-flipped (+s) placebo in-sample.
    base = _sharpe(ctx["search"])
    pl_ret, _ = signal(ctx["panel"], flip=True)                    # one extra signal() call
    pl = pl_ret[pl_ret.index < pd.Timestamp(ctx["holdout_start"])] # sliced to search window
    return {"pass": base > _sharpe(pl), "observed": round(base - _sharpe(pl), 3)}


def _chk_hold(ctx):
    hd = [t.get("hold_days") for t in (ctx.get("trades") or []) if t.get("hold_days") is not None]
    if not hd:
        return {"pass": False, "observed": "no_trades"}
    med = float(np.median(hd))
    return {"pass": 4.0 <= med <= 25.0, "observed": med}           # ~weekly, not daily churn


def _chk_halves(ctx):
    r = pd.Series(ctx["search"]).dropna()
    if len(r) < 80:
        return {"pass": False, "observed": "short"}
    mid = len(r) // 2
    s1, s2 = _sharpe(r.iloc[:mid]), _sharpe(r.iloc[mid:])
    return {"pass": (s1 > 0 and s2 > 0), "observed": (round(s1, 2), round(s2, 2))}


# ---------------------------------------------------------------------------
SPEC = StrategySpec(
    id=STRAT_ID,
    family="commodity-storage-premium",
    title="Energy inventory-surprise storage-state premium (EIA-weekly cross-sectional L/S, petroleum complex CL/RB/HO)",
    markets=["commodity-futures-energy"],
    data_desc=(
        "EIA weekly petroleum STOCKS via FRED mirrors -- crude WCESTUS1, total gasoline "
        "WGTSTUS1, distillate WDISTUS1 (fred_series, fetched per-series so an unmirrored id "
        "cannot 400 the batch) -- for the inventory-CHANGE surprise; front-month energy "
        "futures CL=F/RB=F/HO=F (yf_panel) for tradable returns, positive-price-filtered + "
        "clipped at +/-50%/day to remove the Apr-2020 negative-crude and continuous-futures "
        "roll-gap artifacts. Point-in-time throughout (2010->2026)."
    ),
    pre_registration=(
        "PREMISE: a slow-diffusion storage / convenience-yield risk premium -- the commodity "
        "analog of PEAD. When realized EIA inventory comes in TIGHTER than a seasonal/AR "
        "expectation (a draw surprise -> backwardation/stockout risk) go LONG that root; when it "
        "comes in GLUTTIER (build surprise -> contango) go SHORT. PRIMARY signal = -s where s is "
        "the standardised surprise; default params ARE the primary -- no grid cherry-pick.\n"
        "EXPECTATION MODEL (honest caveat): we do NOT own analyst CONSENSUS, so 'surprise' is "
        "proxied by our OWN expanding-window model of the weekly inventory CHANGE -- week-of-year "
        "seasonal mean + AR(1) on the detrended change, fit ONLY on weeks strictly before each "
        "release. This is a MODEL-surprise, not a survey-surprise (a true consensus test needs a "
        "paid forecast feed). Disclosed, legitimate, not look-ahead.\n"
        "PIT / NO LOOK-AHEAD: inventories are aligned to the EIA RELEASE, not the survey reference "
        "week. The week-ending(Fri) value is treated as known only +6 calendar days later (covers "
        "the Wed release and holiday-delayed Thu), forward-filled to the daily calendar; the weight "
        "matrix is then shift(1)-lagged for execution -- entry is the session AFTER the release and "
        "we never trade the release-day intraday move (we harvest the post-release DRIFT). The "
        "expectation model, the standardising sigma, and the inverse-vol all use PAST data only.\n"
        "CONSTRUCTION: dollar-neutral cross-section over {CL,RB,HO} (the crude->products "
        "refining/storage chain), inverse-vol (equal-risk) weighted, frozen on a weekly grid (held "
        "to the next release), vol-targeted ~10% ann. on trailing realised vol; 8bps/turnover "
        "(conservative for futures). Returns use the continuous front-month series with the "
        "positive-price filter + clip; a production deployment should measure WITHIN the front "
        "contract month via the Databento curve close_1 (not reachable in this import surface).\n"
        "SCOPE = LOCAL (deviates from the proposal's 'broad'): only the EIA PETROLEUM weekly stocks "
        "are reachable through fred_series here. The universal-storage breadth test (grain stocks "
        "ZC/ZS/ZW via USDA, or natgas via EIA weekly storage) requires the usda_nass / eia_series "
        "adapters NOT exposed in this module, and the proposal itself offers only ONE generalization "
        "complex (grains) -- short of the >=3 disjoint universes a broad stage-2 battery demands. We "
        "therefore register a defensibly universe-specific petroleum-complex edge and confirm it via "
        "OOS holdout (>=2022) + forward paper validation, per the contract's local path.\n"
        "WHY NOT A DUPLICATE: prior commodity-fundamentals tests used inventory LEVELS/deviations "
        "(inv_cond_convenience_yield_v1, commod_scarcity_2of3, commodity-xs-carry-spotbasis) or "
        "price-only basis (basis_momentum_bp2019). This is the inventory-SURPRISE -- actual minus a "
        "PIT seasonal/AR expectation, release-aligned -- a different AXIS (event/flow update of the "
        "storage state), not a re-parameterisation."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"band": 0.0, "vol_lb": 60, "target_vol": 0.10, "max_lev": 3.0, "flip": False},
    grid={
        "default":     {},
        "band_0p5":    {"band": 0.5},
        "vol_lb_90":   {"vol_lb": 90},
        "voltgt_0p08": {"target_vol": 0.08},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "storage_direction_beats_flip",
         "claim": "in-sample Sharpe of the storage-direction (-s) tilt exceeds the sign-flipped (+s) placebo",
         "check": _chk_direction},
        {"name": "weekly_holding",
         "claim": "median trade hold is ~one weekly rebalance (4-25 trading days), not daily churn",
         "check": _chk_hold},
        {"name": "robust_across_halves",
         "claim": "search-window edge is positive in BOTH halves (not a single episode)",
         "check": _chk_halves},
    ],
)