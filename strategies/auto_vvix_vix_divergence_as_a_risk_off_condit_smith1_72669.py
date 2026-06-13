"""
VVIX/VIX divergence as a risk-off conditioning gate on US-equity exposure.

THESIS (conditioning/timing, NOT a new short-vol premium): institutional tail-hedge
demand shows up as vol-OF-vol (VVIX) rising while headline VIX stays low — the
documented "Volmageddon" configuration. When sophisticated hedgers are paying up for
VIX convexity while spot vol is asleep, the left tail is being insured ahead of a
vol-regime shift. The harvested object is the risk-adjusted improvement of a
CONDITIONED long-equity stream (cut to cash on the flag) over the SAME book held
unconditionally (buy-and-hold long US-equity beta). No leverage, no short leg.

FAITHFUL BASE BOOK: per the proposal ('US equity (SPY/ES via yf_panel)'; deployment
proxy = a liquid broad-equity vehicle SPY/IVV/VTI), the base is a BUY-AND-HOLD broad
US-equity beta book (equal-weight SPY/IVV/VTI), NOT an inverse-vol weekly-rebalanced
stock book. This keeps the base weighting and cadence identical to the frozen 'long
SPY' design and avoids confounding the gate's drawdown contribution.

DATA (all $0 / OWNED-FREE):
  - US broad-equity beta book: SPY/IVV/VTI via yf_panel (adjusted close).
  - VVIX (vol-of-vol): yfinance "^VVIX" via yf_panel (index history ~2007+).
  - spot VIX: FRED VIXCLS via fred_series.

LAG / NO LOOK-AHEAD: the divergence FLAG is computed on each day-t close, folded into
same-day target weights, then the whole weight matrix is shift(1)'d before costing &
ledgering. So a flag observed at close t-1 first cuts exposure on day t — strict 1-day
lag, zero look-ahead (the lag lives in `W = weights.shift(1)`).

BINDING CONSTRAINT: LOW EVENT COUNT. Distinct divergence episodes 2006-2026 are few
(expected < ~15). Statistical power is the limiting factor; thresholds are FROZEN and
are NOT loosened to manufacture events. Any pass must carry the low-power caveat to the
human gate. Placebo (random count/season-matched gate dates must not improve the
stream) is pre-registered as harness MCPT.
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2006-01-01"
_VVIX_COL, _VIX_COL = "__VVIX__", "__VIX__"

# Broad US-equity beta book, BUY-AND-HOLD (frozen-faithful 'long SPY' base; deployment proxy).
_EQ = ["SPY", "IVV", "VTI"]
_SECTOR_MAP = {"SPY": "US-Equity-Beta", "IVV": "US-Equity-Beta", "VTI": "US-Equity-Beta"}

# Frozen primary parameters (grid declares the search burden as ROBUSTNESS, not selection).
_DEFAULTS = {"vvix_th": 110.0, "vix_th": 18.0, "ratio_th": 6.5, "window": 10}


def load_data() -> pd.DataFrame:
    prices = yf_panel(_EQ, START)               # adjusted-close broad US-equity beta book
    idx = prices.index

    vvix_df = yf_panel(["^VVIX"], START)
    vvix = vvix_df["^VVIX"] if isinstance(vvix_df, pd.DataFrame) else vvix_df
    vix = fred_series({"VIXCLS": _VIX_COL}, START)[_VIX_COL]

    panel = prices.copy()
    # join conditioning series onto the equity trading-day grid; bridge small gaps only
    panel[_VVIX_COL] = vvix.reindex(idx).ffill(limit=5)
    panel[_VIX_COL] = vix.reindex(idx).ffill(limit=5)
    return panel


def _split(panel):
    cond_cols = [c for c in (_VVIX_COL, _VIX_COL) if c in panel.columns]
    prices = panel.drop(columns=cond_cols)
    return prices, panel[_VVIX_COL], panel[_VIX_COL]


def signal(panel, **params):
    apply_gate = params.pop("apply_gate", True)
    p = {**_DEFAULTS, **params}
    prices, vvix, vix = _split(panel)
    idx = prices.index

    # ---- divergence FLAG (evaluated on day-t close; lagged 1 day below) ----
    flag = (((vvix > p["vvix_th"]) & (vix < p["vix_th"])) |
            ((vvix / vix > p["ratio_th"]) & (vix < p["vix_th"])))
    flag = flag.reindex(idx).fillna(False).astype(float)

    if apply_gate:
        # any flag in the trailing `window` days -> risk-off (rolling re-arm/extend)
        risk_off = flag.rolling(int(p["window"]), min_periods=1).max() > 0
        gate = (~risk_off).astype(float)        # 1 = hold beta book, 0 = cash
    else:
        gate = pd.Series(1.0, index=idx)        # ungated baseline (used by soft check)

    # ---- base book: BUY-AND-HOLD equal-weight broad equity beta (frozen 'long SPY') ----
    rets = prices.pct_change()
    n = prices.shape[1]
    base = pd.DataFrame(1.0 / n, index=idx, columns=prices.columns)   # constant long beta target

    # ---- apply conditioning gate, then LAG the whole matrix (lag = our responsibility) ----
    weights = base.mul(gate, axis=0).fillna(0.0)                     # same-day target weights
    W = weights.shift(1).fillna(0.0)                                 # held on day t, info <= t-1

    daily = net_of_cost(W, rets, cost_bps=8.0, name="vvix_vix_divergence_gate")
    trades = trades_from_weights(W, rets, _SECTOR_MAP)               # auto-stamps entry_regime
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': the broad-generalization stage-2 battery is NOT run by the harness.
    # Robustness here is via the declared grid (threshold/window sign-stability), the soft
    # expectations, and forward-validation on the holdout — not cross-universe transfer
    # (VVIX is a US-S&P-specific vol-of-vol object, not a universal cross-market factor).
    raise NotImplementedError("scope='local': no broad-generalization battery for this gate")


# ---------------------------- soft expectations ----------------------------
def _maxdd(r):
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _check_event_count(ctx):
    # Surfaces the BINDING constraint (low event count) and confirms a non-degenerate flag.
    _, vvix, vix = _split(ctx["panel"])
    hs = pd.Timestamp(ctx["holdout_start"])
    flag = (((vvix > _DEFAULTS["vvix_th"]) & (vix < _DEFAULTS["vix_th"])) |
            ((vvix / vix > _DEFAULTS["ratio_th"]) & (vix < _DEFAULTS["vix_th"])))
    flag = flag.reindex(vvix.index).fillna(False)
    flag = flag.loc[flag.index < hs]
    onsets = int((flag & ~flag.shift(1).fillna(False)).sum())
    # pass = flag is non-degenerate; the count itself is the headline caveat for the human gate.
    return {"pass": onsets >= 1, "observed": onsets}


def _check_gate_cuts_drawdown(ctx):
    # Core mechanism claim: conditioning REDUCES the left tail vs the ungated beta book.
    hs = pd.Timestamp(ctx["holdout_start"])
    cond = ctx["search"]
    ungated, _ = signal(ctx["panel"], apply_gate=False)   # the single allowed extra signal() call
    ungated = ungated.loc[ungated.index < hs]
    cond = cond.loc[cond.index < hs]
    dd_c, dd_u = _maxdd(cond), _maxdd(ungated)            # both <= 0
    return {"pass": dd_c >= dd_u - 1e-9,
            "observed": f"cond_maxdd={dd_c:.4f} vs ungated_maxdd={dd_u:.4f}"}


def _check_threshold_sign_robust(ctx):
    # Robustness, NOT selection: threshold/window perturbations must share the default's sign.
    grid = ctx.get("grid") or {}
    if "default" not in grid:
        return {"pass": True, "observed": "no grid"}
    sgn = lambda r: int(np.sign(r.fillna(0.0).mean()))
    base = sgn(grid["default"])
    frac = float(np.mean([1.0 if (sgn(r) == base and base != 0) else 0.0 for r in grid.values()]))
    return {"pass": frac >= 0.80, "observed": round(frac, 3)}


SPEC = StrategySpec(
    id="vvix_vix_divergence_gate",
    family="volatility_conditioning",
    title="VVIX/VIX divergence risk-off gate on US-equity beta (vol-of-vol tail-hedge timing)",
    markets=["US-equity-broad"],
    data_desc=("Buy-and-hold broad US-equity beta book (SPY/IVV/VTI via yf_panel, adjusted close) as the "
               "frozen 'long SPY' base; conditioned on the free CBOE vol complex: VVIX (yfinance ^VVIX, "
               "~2007+) and spot VIX (FRED VIXCLS), joined to the equity trading-day grid (ffill limit 5d). "
               "All $0/owned."),
    pre_registration=(
        "FROZEN SPEC. Hypothesis: VVIX (vol-of-vol) rising while VIX stays low = institutional "
        "tail-hedge demand preceding vol-regime shifts (Volmageddon config). CONDITIONING/timing, "
        "explicitly NOT a new short-vol premium. "
        "BASE = long buy-and-hold broad US-equity beta (SPY/IVV/VTI, equal-weight, no inverse-vol/no "
        "weekly rebalance) — the frozen 'long SPY' exposure; benchmark is the same book held unconditionally. "
        "FLAG = (VVIX>110 & VIX<18) OR (VVIX/VIX>6.5 & VIX<18), evaluated on day-t close and ACTED "
        "with a strict 1-day lag (weights.shift(1)) => a flag seen at close t-1 first cuts exposure "
        "day t; zero look-ahead. RULE: flag fires -> cut the ENTIRE long beta book to cash for 10 "
        "trading days (rolling re-arm/extend); else hold long. No leverage, no short leg. Costs 8bps "
        "on turnover (turnover fires on gate transitions). "
        "BINDING CONSTRAINT: LOW EVENT COUNT. Distinct divergence episodes 2006-2026 are few (expected "
        "<~15); statistical power is the limiting factor; thresholds are FROZEN and NOT loosened to "
        "manufacture events; any pass MUST carry the low-power caveat to the human gate. "
        "GRID (DSR effective-N / honest search burden, declared as ROBUSTNESS not selection): "
        "VVIX 105/110/115, ratio 6.0/6.5/7.0, window 5/10/15; perturbations must share sign. "
        "PRE-REGISTERED PLACEBO: a random count/season-matched set of gate dates must NOT improve the "
        "stream (run as harness MCPT, not a single cheap resample which would be statistically "
        "meaningless). GATE0 data checks: VVIX joins to VIXCLS on the business-day grid with no gaps "
        ">5d; episode count reported loudly; flag verified non-degenerate; strict-lag audit on the "
        "joined frame; equity panel covers 2006+. Holdout 2022-01-01; search < 2022."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},                       # "default" = the frozen _DEFAULTS, primary variant
    grid={
        "default": {},
        "vvix105": {"vvix_th": 105.0},
        "vvix115": {"vvix_th": 115.0},
        "ratio6.0": {"ratio_th": 6.0},
        "ratio7.0": {"ratio_th": 7.0},
        "win5": {"window": 5},
        "win15": {"window": 15},
    },
    scope="local",                           # US-vol-specific timing gate; forward-validation confirms
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "flag_non_degenerate_event_count",
         "claim": "Divergence flag fires at least once in the search window; episode count is the "
                  "binding low-power caveat reported to the human gate (expected < ~15 distinct events).",
         "check": _check_event_count},
        {"name": "gate_cuts_drawdown",
         "claim": "Conditioned (gated) book max drawdown <= ungated buy-and-hold beta book max drawdown "
                  "over the search window — the gate avoids the left tail, which is the harvested object.",
         "check": _check_gate_cuts_drawdown},
        {"name": "threshold_window_sign_robust",
         "claim": ">=80% of grid threshold/window perturbations share the sign of the default's "
                  "search-window mean return (robustness, not selection).",
         "check": _check_threshold_sign_robust},
    ],
)