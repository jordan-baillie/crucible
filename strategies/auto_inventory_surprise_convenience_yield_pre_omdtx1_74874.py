"""
Inventory-surprise convenience-yield premium — energy complex (LOCAL build)
===========================================================================

THESIS (storage theory / convenience-yield risk premium, event-driven):
On each inventory-report release, a storage SURPRISE = actual inventory change
minus the change EXPECTED by a seasonal + AR(1) model fit ONLY on data prior to
that release (strict PIT). A negative surprise (drew harder than the seasonal
norm) = physical tightness = bullish; a positive surprise (built more than norm)
= glut = bearish. A naked surprise bet is just announcement drift; the thesis is
that requiring PRICE CONFIRMATION (trend agreement, see substitution note) turns
the event into a harvestable convenience-yield premium.  Long confirmed-tight,
short confirmed-glut, inverse-vol sized, vol-targeted, weekly rebalanced with
hysteresis, net of futures round-trip + roll costs.

HONEST DEVIATIONS FROM THE PROPOSAL (data reality under the allowed adapters):
  * `eia_series` / `usda_nass` / `fut_curve` are NOT in the tested adapter set
    (eia_series is KEY-PENDING per ops). So:
      - Inventory is sourced from FRED-hosted EIA *petroleum* stocks via the
        allowed `fred_series` adapter:  WCESTUS1 (crude ex-SPR), WGTSTUS1 (total
        gasoline), WDISTUS1 (distillate).  These are released ~Wed 10:30 ET for
        the prior-Friday reference week; FRED dates them to the reference week,
        so we apply a conservative RELEASE_LAG (business-day shift) to remove the
        publication look-ahead.  ==> roots = {CL, RB, HO}.
      - Natural-gas storage and USDA grain stocks have no clean PIT series via
        the allowed adapters, so NG and the grains sub-book are DEFERRED (not
        silently faked).  This is therefore an ENERGY-COMPLEX book.
      - TERM-STRUCTURE roll-yield (the proposal's confirmer) needs contract-month
        data (fut_curve) we cannot load; as a DECLARED PROXY we confirm with a
        trailing price-trend sign (backwardated markets tend to carry/trend up).
        This is a weaker confirmer than true roll-yield — prior stays LOW.
  * With only ~3 highly-correlated roots, cross-sectional TERCILES are degenerate,
    so we trade the confirmed SIGN directly (long tight / short glut / flat on
    disagreement) and let inverse-vol sizing spread risk.

FIX (vs the failed build, error: "Invalid comparison between dtype=int64 and str"):
  (1) signal() no longer returns an empty `pd.Series(dtype=float)`, whose DEFAULT
      RangeIndex (int64) made the harness date-split `returns.index < holdout_start`
      blow up — it now returns a zero Series carried on the panel's DatetimeIndex
      (an empty/degenerate DatetimeIndex compares cleanly with a date string).
  (2) load_data() DECOUPLES the tradable price block from the inventory block:
      a root is kept whenever its PRICE loads, and the (possibly missing) FRED
      inventory is attached as a NaN column.  A swallowed `fred_series` failure can
      therefore no longer collapse the panel to ZERO columns — the exact path that
      produced the empty RangeIndex Series.  Missing inventory => all-NaN surprise
      => flat for that root (degrades gracefully, never crashes).
  (3) every index is explicitly coerced to a NAMED DatetimeIndex so the FLAT panel
      survives the harness cache round-trip (the prior MultiIndex-column fix stands).

SCOPE = 'local' (override of proposal's 'broad').  The broad generalization
battery is equity-cross-section-shaped (>=3 disjoint 150-400 name universes); a
3-root energy complex cannot form those, and the grains sub-complex is
unavailable.  Per the proposal's OWN fallback ("if the edge lives only in energy,
downgrade to LOCAL"), we ship local and confirm via the frozen holdout /
forward-paper rather than promote on a single complex.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, inv_vol_position
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

SID = "inv_surprise_convyield_energy_v1"
START = "2009-01-01"
RELEASE_LAG = 5          # business-day shift to clear the EIA publication delay (Fri ref-week -> Wed release)

# root -> (yfinance future, FRED inventory id, ledger "sector")
ROOTS = {
    "CL": ("CL=F", "WCESTUS1", "Crude Oil"),        # crude ex-SPR ending stocks
    "RB": ("RB=F", "WGTSTUS1", "Gasoline"),         # total gasoline stocks
    "HO": ("HO=F", "WDISTUS1", "Distillate Fuel"),  # distillate fuel oil stocks
}
YF2ROOT = {v[0]: k for k, v in ROOTS.items()}
SECTOR_MAP = {k: v[2] for k, v in ROOTS.items()}

DEFAULTS = dict(
    min_train=40,     # min reports before the PIT expectation model is fit
    mom_lb=63,        # trailing-trend confirmer lookback (~3m), proxy for term-structure carry
    enter=0.5,        # tightness-z entry band (hysteresis)
    exit=0.1,         # tightness-z exit band (hysteresis < entry -> caps churn -> ~10d holds)
    target_vol=0.10,  # annualised vol target
    vol_lb=63,        # inverse-vol lookback
    max_pos=3,
    cost_bps=10.0,    # ~8bps base + ~2bps futures roll approximation, on turnover
    confirm=True,     # require price-trend confirmation (False == "naked surprise" baseline)
)


def _to_dt_index(obj):
    """Coerce an index/Series/DataFrame index to a named DatetimeIndex (round-trip safe)."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj = obj.copy()
        obj.index = pd.to_datetime(obj.index, errors="coerce")
    obj.index.name = "date"
    return obj


# ---------------------------------------------------------------------------
# Helpers (the ONLY novel code; everything else is kit)
# ---------------------------------------------------------------------------
def _pit_surprise(level_daily, min_train):
    """PIT inventory-SURPRISE z-score, carried forward daily.

    level_daily : daily, ffilled, ALREADY release-lagged inventory level.
    Surprise = actual weekly change - change predicted by a seasonal(2 harmonics)
    + AR(1) model fit on STRICTLY PRIOR reports only (expanding-window OLS).
    z-scored against the expanding distribution of past surprises.  No look-ahead:
    each prediction at report i uses rows [0, i) only; the level itself was lagged
    past the public release date in load_data().  All-NaN input -> all-NaN out.
    """
    out = pd.Series(np.nan, index=level_daily.index, dtype=float)
    s = level_daily.dropna()
    if s.empty:
        return out
    rep = s[s.ne(s.shift(1))]          # value at each new report (1st daily date it appears, lagged)
    dc = rep.diff().dropna()           # weekly inventory CHANGE
    if len(dc) < min_train + 5:
        return out
    doy = dc.index.dayofyear.values.astype(float)
    X = np.column_stack([
        np.ones(len(dc)),
        dc.shift(1).values,                       # AR(1)
        np.sin(2 * np.pi * doy / 365.25),         # annual cycle
        np.cos(2 * np.pi * doy / 365.25),
        np.sin(4 * np.pi * doy / 365.25),         # semi-annual cycle
        np.cos(4 * np.pi * doy / 365.25),
    ])
    y = dc.values
    ok = ~np.isnan(X).any(axis=1)                 # drop the first row (AR(1) lag NaN)
    X, y, idx = X[ok], y[ok], dc.index[ok]
    if len(idx) <= min_train:
        return out
    surprise = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(min_train, len(idx)):
        beta, *_ = np.linalg.lstsq(X[:i], y[:i], rcond=None)   # fit on prior reports ONLY
        surprise.iloc[i] = y[i] - X[i] @ beta                  # actual - expected
    mu = surprise.expanding(min_periods=20).mean()
    sd = surprise.expanding(min_periods=20).std()
    sz = (surprise - mu) / sd.replace(0.0, np.nan)
    return sz.reindex(level_daily.index).ffill()               # carry forward to next report


def _state_machine(tight_z, mom, enter, exit_, use_mom):
    """Hysteresis long/short/flat per root.
    tight_z > 0 = tighter-than-expected (bullish);  long needs confirmation mom>0,
    short needs mom<0 (proxy for backwardation/contango).  Exit on band cross or
    trend flip; this + weekly rebalance gives ~>=10-day holds and caps turnover.
    """
    tz = tight_z.values
    mm = mom.values if (use_mom and mom is not None) else np.zeros(len(tz))
    out = np.zeros(len(tz)); state = 0
    for i in range(len(tz)):
        t, m = tz[i], mm[i]
        if np.isnan(t) or (use_mom and np.isnan(m)):
            out[i] = state; continue
        long_ok = (not use_mom) or (m > 0)
        short_ok = (not use_mom) or (m < 0)
        if state == 0:
            if t > enter and long_ok: state = 1
            elif t < -enter and short_ok: state = -1
        elif state == 1:
            if t < exit_ or (use_mom and m < 0): state = 0
            if t < -enter and short_ok: state = -1
        else:  # state == -1
            if t > -exit_ or (use_mom and m > 0): state = 0
            if t > enter and long_ok: state = 1
        out[i] = state
    return pd.Series(out, index=tight_z.index)


# ---------------------------------------------------------------------------
# Data  (FLAT single-level panel + named DatetimeIndex -> survives round-trip)
# ---------------------------------------------------------------------------
def load_data():
    fut = yf_panel(list(YF2ROOT.keys()), START)
    fut = fut[[c for c in YF2ROOT if c in fut.columns]].rename(columns=YF2ROOT)
    fut = _to_dt_index(fut).sort_index()
    idx = fut.index

    inv_cols = {}
    for root, (_, fid, _) in ROOTS.items():
        if not fid:
            continue
        try:
            s = fred_series({fid: root}, START)[root]
            if s.dropna().shape[0] > 100:
                inv_cols[root] = s
        except Exception:
            pass

    if inv_cols and len(idx):
        inv_raw = _to_dt_index(pd.DataFrame(inv_cols))
        inv = (inv_raw.reindex(idx.union(inv_raw.index)).sort_index()
               .ffill().reindex(idx))
        inv = inv.shift(RELEASE_LAG)          # PIT: remove EIA publication look-ahead
    else:
        inv = pd.DataFrame(index=idx)

    # PRICE-driven universe: keep a root whenever its tradable price exists; attach
    # inventory if available, else a NaN column (flat signal, never collapses panel).
    roots = [r for r in ROOTS if r in fut.columns]
    out = pd.DataFrame(index=idx)
    for r in roots:
        out["P_" + r] = fut[r]                                  # price block
        out["I_" + r] = inv[r] if r in inv.columns else np.nan  # release-lagged inventory block
    out = _to_dt_index(out).sort_index()
    return out


def load_gen_data(label):
    # scope='local' -> the broad generalization battery is not applicable to a
    # tiny commodity complex; forward-paper / holdout confirms the book.
    raise NotImplementedError("scope='local': no generalization battery")


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def _empty_returns(panel):
    """Degenerate return: ALWAYS a DatetimeIndex (never the default RangeIndex)."""
    idx = panel.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex([])
    return pd.Series(0.0, index=idx, name=SID), []


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    if not isinstance(panel.index, pd.DatetimeIndex):     # defensive round-trip coercion
        panel = panel.copy()
        panel.index = pd.to_datetime(panel.index, errors="coerce")

    roots = [c[2:] for c in panel.columns if isinstance(c, str) and c.startswith("P_")]
    if not roots:
        return _empty_returns(panel)

    prices = panel[["P_" + r for r in roots]].copy(); prices.columns = roots
    inv = panel[["I_" + r for r in roots]].copy();    inv.columns = roots
    rets = prices.pct_change()

    sig = pd.DataFrame(0.0, index=prices.index, columns=roots)
    for r in roots:
        tight_z = -_pit_surprise(inv[r], p["min_train"])          # tightness = -surprise
        mom = prices[r].pct_change(p["mom_lb"])                    # trend = term-structure proxy
        sig[r] = _state_machine(tight_z, mom, p["enter"], p["exit"], p["confirm"])

    # inv_vol_position returns weekly-held, inverse-vol-sized, vol-targeted,
    # ALREADY-LAGGED positions -> do NOT shift again before net_of_cost.
    W = inv_vol_position(sig, rets, p["target_vol"], p["vol_lb"], p["max_pos"], "W")
    daily = net_of_cost(W, rets, cost_bps=p["cost_bps"], name=SID)
    trades = trades_from_weights(W, rets, SECTOR_MAP)
    return daily, trades


# ---------------------------------------------------------------------------
# Soft expectations (machine-checkable mechanism claims)
# ---------------------------------------------------------------------------
def _sr(r):
    if r is None:
        return float("nan")
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(np.sqrt(252) * r.mean() / r.std())


def chk_confirmation(ctx):
    """Thesis: price-confirmation turns the naked surprise bet into a premium ->
    confirmed search-window Sharpe >= naked (no-confirmation) Sharpe.  Compared
    via the pre-declared grid variants (no extra signal() call)."""
    g = ctx.get("grid", {}) or {}
    d, n = _sr(g.get("default")), _sr(g.get("naked"))
    ok = (d == d) and (n == n) and (d >= n - 1e-9)
    return {"pass": bool(ok), "observed": f"confirmed_SR={d:.2f} naked_SR={n:.2f}"}


def chk_minhold(ctx):
    """Claim: hysteresis + weekly rebalance give a ~>=10 trading-day median hold."""
    tr = ctx.get("trades", []) or []
    hd = [t.get("hold_days", 0) for t in tr if t.get("hold_days") is not None]
    med = float(np.median(hd)) if hd else 0.0
    return {"pass": bool(med >= 10), "observed": med}


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
SPEC = StrategySpec(
    id=SID,
    family="commodity_convenience_yield",
    title="Inventory-surprise convenience-yield premium — EIA-petroleum event-driven energy book (LOCAL)",
    markets=["commodity_futures"],
    data_desc=("FRED-hosted EIA weekly petroleum stocks (WCESTUS1 crude, WGTSTUS1 "
               "gasoline, WDISTUS1 distillate; release-lagged to remove publication "
               "look-ahead) + yfinance energy futures (CL=F/RB=F/HO=F). Surprise = "
               "actual weekly change minus a PIT seasonal+AR(1) expectation; price-trend "
               "confirmation proxies term-structure roll-yield (fut_curve unavailable)."),
    pre_registration=(
        "Long roots whose inventory drew harder than the PIT seasonal+AR(1) norm AND "
        "whose price trend confirms (proxy for backwardation); short roots that built "
        "more than norm AND trend-confirm contango; flat on disagreement. Inverse-vol, "
        "10% vol target, weekly rebalance, hysteresis (entry>exit) for ~>=10d holds, "
        "10bps turnover cost (8bps base + roll). PRIMARY = 'default'. SUBSTITUTIONS "
        "(prior LOW): (1) modeled surprise, not analyst consensus (unowned); (2) trend "
        "proxy instead of true roll-yield (fut_curve unavailable); (3) NG storage + USDA "
        "grains DEFERRED (no clean PIT series via allowed adapters) -> energy-complex "
        "only. SCOPE='local' (override of proposal 'broad'): a 3-root complex cannot "
        "form the >=3 disjoint 150-400-name generalization universes the broad battery "
        "needs; per the proposal's own fallback, energy-only => LOCAL, confirm on the "
        "frozen holdout / forward-paper. Expect a modest Sharpe; the confirmation gate "
        "must beat the naked-surprise baseline (soft-checked) or the convenience-yield "
        "story is falsified."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "naked": {"confirm": False},      # drops price confirmation -> tests the thesis
        "slow_mom": {"mom_lb": 120},
        "wide_band": {"enter": 1.0},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "confirmation_adds_value",
         "claim": "confirmed-book search Sharpe >= naked-surprise (no price confirmation) Sharpe",
         "check": chk_confirmation},
        {"name": "min_hold_10d",
         "claim": "hysteresis + weekly rebalance give a median hold of >= 10 trading days",
         "check": chk_minhold},
    ],
)