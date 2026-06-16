"""
Storage-surprise convenience-yield premium — ENERGY petroleum complex.

PRIMARY (frozen): per-commodity standardized inventory SURPRISE, contrarian to BUILD.
For each energy commodity we forecast the just-released EIA weekly ending-stock level
with an expanding-window model (intercept + AR(1) level + 2 annual harmonics), fit
STRICTLY on data available before the release; surprise = actual - forecast, standardized
by the expanding std of past forecast errors. Target = -clip(z)/clip  (unexpected DRAW ->
long front future, unexpected BUILD -> short), with a +/-0.5sigma hysteresis band
(hold prior position when |z|<band) and hold-to-next-release.

NOTE ON SCOPE (honest downgrade from the proposal's 'broad'): the whitelisted adapter set
reaches EIA petroleum stocks via FRED but NOT USDA grain stocks (no usda_nass) nor a
reliable weekly natural-gas-storage FRED series, so the proposal's energy-vs-grains
cross-complex independence test and the >=3 disjoint 150-400-name stage-2 battery are
structurally impossible for a 3-commodity petroleum book. The three products are also
correlated -> the built-in independence robustness is weak. We therefore set scope='local'
and validate via the write-once 2022+ holdout (forward-validation) + MCPT, treating any
in-search pass with suspicion (the "surprise" is a MODEL residual, NOT an analyst-consensus
surprise -> prediction-edge tail risk). No look-ahead: see PIT handling below.

FIX (this revision): the previous "0 commodities loaded" abort was a PARTIAL DATA-FETCH
failure, not a logic bug — a single flaky/rate-limited futures ticker can zero out a
batched yf_panel call, dropping the whole book (the same transient signature seen on the
broad-universe crypto re-fetch). Hardening: (1) futures are loaded with a batched attempt
FIRST, then a PER-TICKER fallback fills any gaps, so one bad ticker no longer kills the
others; (2) series are assembled with an outer-join (pd.DataFrame(dict)) so the union of
trading dates is preserved; (3) prices are numerically coerced and floored at 200 obs to
reject empty/garbage frames; (4) the EIA series remain fetched ONE-PER fred_series CALL so
a single dead/renamed id cannot abort the load (no semantically-wrong fallback id is used —
e.g. incl-SPR crude would corrupt the frozen surprise model). The book proceeds on whatever
subset (>=2) of {price, inventory} pairs returns; a residual all-source outage still raises
(correct, transient — the paced nightly run recovers).
"""

from __future__ import annotations

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, inv_vol_position
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# code -> (roll-adjusted continuous front future on yfinance, FRED EIA weekly stock series, sector)
CONFIG = {
    "CL": {"fut": "CL=F", "fred": "WCESTUS1", "sector": "Energy-Crude"},       # crude ending stocks excl SPR
    "RB": {"fut": "RB=F", "fred": "WGTSTUS1", "sector": "Energy-Gasoline"},    # total gasoline stocks
    "HO": {"fut": "HO=F", "fred": "WDISTUS1", "sector": "Energy-Distillate"},  # distillate -> heating oil
}
START = "2007-01-01"
RELEASE_LAG_DAYS = 5  # EIA Weekly Petroleum Status Report: Friday reference period, published ~the next Wed.

SECTOR_MAP = {k: v["sector"] for k, v in CONFIG.items()}


# ----------------------------------------------------------------------------- data
def _align_to_release(weekly: pd.Series, trade_index: pd.DatetimeIndex) -> pd.Series:
    """Map each FRED week-ending observation to the first futures trading day on/after
    (reference_date + RELEASE_LAG_DAYS) -> the value is only ever VISIBLE at its release.
    PIT-conservative; inv_vol_position then adds the standard +1 trading-day execution lag."""
    rel = weekly.copy()
    rel.index = rel.index + pd.Timedelta(days=RELEASE_LAG_DAYS)
    pos = np.clip(trade_index.searchsorted(rel.index, side="left"), 0, len(trade_index) - 1)
    out = pd.Series(rel.values, index=trade_index[pos])
    return out[~out.index.duplicated(keep="last")]


def _fetch_fred_series(fred_id: str, col: str, start: str):
    """Fetch ONE FRED series in its own request. Returns a float Series or None on any
    failure (HTTP 400 from a bad/renamed id, empty frame, etc.) so the load never aborts."""
    try:
        raw = fred_series({fred_id: col}, start)
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, pd.Series):
        s = raw
    elif col in getattr(raw, "columns", []):
        s = raw[col]
    elif getattr(raw, "shape", (0, 0))[1] >= 1:
        s = raw.iloc[:, 0]
    else:
        return None
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s if len(s) else None


def _load_futures(cfg: dict, start: str) -> pd.DataFrame:
    """Robust front-future loader: batched attempt first, then per-ticker fallback for any
    gaps. Outer-joins (pd.DataFrame(dict)) to preserve the union of trading dates. A single
    flaky ticker can no longer zero out the whole book."""
    codes = list(cfg.keys())
    fut = {code: cfg[code]["fut"] for code in codes}
    series: dict = {}

    # 1) Batched attempt (one download call).
    try:
        raw = yf_panel(list(fut.values()), start)
    except Exception:
        raw = None
    if raw is not None:
        if isinstance(raw, pd.Series):
            raw = raw.to_frame()
        cols = list(getattr(raw, "columns", []))
        for i, code in enumerate(codes):
            tic = fut[code]
            col = None
            if tic in cols:
                col = tic
            elif tic.replace("=F", "") in cols:
                col = tic.replace("=F", "")
            elif len(cols) == len(codes):
                col = cols[i]              # positional fallback if adapter renamed columns
            if col is not None:
                s = pd.to_numeric(raw[col], errors="coerce").dropna()
                if len(s) >= 200:
                    series[code] = s

    # 2) Per-ticker fallback for anything still missing.
    for code in codes:
        if code in series:
            continue
        try:
            raw1 = yf_panel([fut[code]], start)
        except Exception:
            continue
        if raw1 is None:
            continue
        if isinstance(raw1, pd.Series):
            s = raw1
        elif getattr(raw1, "shape", (0, 0))[1] >= 1:
            s = raw1[fut[code]] if fut[code] in getattr(raw1, "columns", []) else raw1.iloc[:, 0]
        else:
            continue
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) >= 200:
            series[code] = s

    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def load_data() -> pd.DataFrame:
    cfg = CONFIG

    px = _load_futures(cfg, START).dropna(how="all")
    if px.shape[1] == 0:
        raise RuntimeError("storage_surprise: no energy futures loaded -> transient yfinance "
                           "fetch failure (paced nightly run should recover).")
    trade_index = px.index

    panel = pd.DataFrame(index=trade_index)
    kept = []
    for code, c in cfg.items():
        if code not in px.columns:
            continue
        # One request per EIA series: a single bad id (HTTP 400) cannot kill the whole load.
        s = _fetch_fred_series(c["fred"], code, START)
        if s is None:
            continue
        s = s[s != s.shift(1)]                 # collapse any ffilled duplicates -> genuine weekly obs
        if len(s) < 80:
            continue
        panel[f"px_{code}"] = px[code]
        panel[f"inv_{code}"] = _align_to_release(s, trade_index).reindex(trade_index)
        kept.append(code)

    if len(kept) < 2:
        raise RuntimeError(f"storage_surprise: only {len(kept)} commodities loaded -> insufficient "
                           f"(likely transient FRED/yfinance fetch failure; nightly run recovers).")
    return panel


def load_gen_data(label: str) -> pd.DataFrame:
    # scope='local' -> stage-2 battery is not run; defined for signature safety only.
    return load_data()


# ----------------------------------------------------------------------------- signal helpers
def _standardized_surprise(s: pd.Series, min_train: int = 78) -> pd.Series:
    """One-step-ahead forecast error of EIA stock level (intercept + AR(1) + 2 annual harmonics),
    fit on an EXPANDING window of observations STRICTLY before each release (no look-ahead),
    standardized by the expanding std of PAST forecast errors only."""
    s = s.astype(float)
    idx = s.index
    n = len(s)
    doy = idx.dayofyear.values.astype(float)
    ann = 2.0 * np.pi * doy / 365.25
    X = np.column_stack([
        np.ones(n),
        np.r_[np.nan, s.values[:-1]],          # AR(1) lag (known at release t)
        np.sin(ann), np.cos(ann),
        np.sin(2 * ann), np.cos(2 * ann),
    ])
    y = s.values
    resid = np.full(n, np.nan)
    for t in range(1, n):
        Xtr, ytr = X[1:t], y[1:t]              # rows strictly before t (row 0 has NaN lag)
        good = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        if good.sum() < min_train or not np.isfinite(X[t]).all():
            continue
        coef, *_ = np.linalg.lstsq(Xtr[good], ytr[good], rcond=None)
        resid[t] = y[t] - X[t] @ coef
    resid = pd.Series(resid, index=idx)
    roll_std = resid.expanding(min_periods=min_train // 2).std().shift(1)  # past residuals only
    return resid / roll_std


def _apply_hysteresis(z: pd.Series, band: float, clip: float) -> pd.Series:
    """Update target only when |z|>=band (hysteresis); otherwise hold prior position.
    Positions change only on release dates -> implicit min-hold to the next release."""
    pos = np.zeros(len(z))
    cur = 0.0
    vals = z.values
    for i, v in enumerate(vals):
        if np.isfinite(v) and abs(v) >= band:
            cur = -float(np.clip(v, -clip, clip)) / clip   # DRAW(neg surprise)->long, BUILD->short
        pos[i] = cur
    return pd.Series(pos, index=z.index)


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    band = float(params.get("hysteresis", 0.5))
    clip = float(params.get("clip", 3.0))

    codes = [c[3:] for c in panel.columns if c.startswith("px_")]
    prices = panel[[f"px_{c}" for c in codes]].copy()
    prices.columns = codes
    rets = prices.pct_change().fillna(0.0)     # roll-adjusted continuous front futures (no roll diffs)

    sig = {}
    for c in codes:
        s = panel[f"inv_{c}"].dropna()
        s = s[s != s.shift(1)]
        z = _standardized_surprise(s)
        target = _apply_hysteresis(z, band=band, clip=clip)        # on release-trade dates
        sig[c] = target.reindex(rets.index).ffill()               # hold-to-next-release
    signal_df = pd.DataFrame(sig).reindex(rets.index).fillna(0.0)

    # Inverse-vol sizing across sleeves, book vol-targeted ~10%, weekly rebalance.
    # inv_vol_position returns positions ALREADY lagged 1 trading day (per kit docstring):
    # we therefore pass W directly to net_of_cost/trades_from_weights (NO extra shift).
    W = inv_vol_position(signal_df, rets, target_vol=0.10, vol_lb=63, max_pos=0.5, rebalance="W")

    daily = net_of_cost(W, rets, cost_bps=5.0, name="storage_surprise_energy")  # ~5bps incl slippage
    daily = daily.dropna()
    daily.name = "storage_surprise_energy"

    trades = trades_from_weights(W, rets, SECTOR_MAP)              # kit stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------------------- soft expectation
def _exp_hysteresis_turnover(ctx):
    """Mechanism claim: the +/-0.5sigma hysteresis band lowers the number of position-change
    runs (turnover proxy) vs a no-band variant on the search window. One extra signal() call;
    everything sliced to dates < holdout_start."""
    try:
        panel = ctx["panel"]
        hs = str(ctx["holdout_start"])
        base = [t for t in ctx.get("trades", []) if str(t.get("entry_date", "")) < hs]
        _, nh = signal(panel, hysteresis=0.0)
        nh = [t for t in nh if str(t.get("entry_date", "")) < hs]
        bn, nn = len(base), len(nh)
        return {"pass": bn <= nn, "observed": f"default_runs={bn} vs no_band_runs={nn}"}
    except Exception as e:
        return {"pass": False, "observed": f"error:{e}"}


SPEC = StrategySpec(
    id="storage_surprise_convenience_yield_energy_v1",
    family="commodity_convenience_yield",
    title="Storage-surprise convenience-yield premium (EIA PIT, energy petroleum complex)",
    markets=["CL=F", "RB=F", "HO=F"],
    data_desc=("yfinance roll-adjusted continuous front futures (CL/RB/HO), loaded batched then "
               "per-ticker-fallback, + FRED EIA weekly ending stocks (WCESTUS1 crude, WGTSTUS1 "
               "gasoline, WDISTUS1 distillate) fetched one-per-request, release-date PIT "
               "(ref-date + ~5d -> next trade day)."),
    pre_registration=(
        "PRIMARY (frozen, no grid cherry-pick): per-commodity standardized inventory SURPRISE, "
        "contrarian to BUILD. Forecast the just-released EIA weekly ending-stock level with an "
        "expanding-window model (intercept + AR(1) level + 2 annual harmonics) fit STRICTLY on "
        "observations available before the release; surprise = actual - forecast, standardized by "
        "the expanding std of past forecast errors. Target = -clip(z,+/-3)/3 (unexpected DRAW->long "
        "front future, unexpected BUILD->short) with a +/-0.5sigma hysteresis band (hold prior "
        "position when |z|<0.5) and hold-to-next-release. MECHANISM: an unexpected draw reveals a "
        "binding convenience yield -> curve tilts to backwardation -> positive expected front return; "
        "a repricing of the storage premium at scheduled releases, NOT an underreaction scalp. "
        "PIT: EIA stocks are FRED-dated at the Friday reference period but PUBLISHED ~the next Wed; "
        "each obs is made visible only on the first futures trading day on/after ref_date+5d, and "
        "inv_vol_position adds the standard +1 trading-day execution lag. Continuous roll-adjusted "
        "futures preserve within-contract continuity (we never diff close_1 across a roll). "
        "ROBUST LOAD: futures are loaded batched then per-ticker so one flaky ticker cannot zero the "
        "book, and the three EIA series are fetched one-per-fred_series-call so a single dead/renamed "
        "id cannot abort the load; the book proceeds on whatever subset (>=2) returns (an all-source "
        "outage still raises -> transient, recovers on the paced nightly run). "
        "SCOPE=LOCAL (honest downgrade from 'broad'): the whitelisted adapters reach EIA petroleum "
        "stocks via FRED but NOT USDA grains (no usda_nass) nor a reliable weekly NG-storage FRED "
        "series, so the proposal's energy-vs-grains independence test and the >=3 disjoint "
        "150-400-name stage-2 battery are structurally impossible for a 3-commodity book; the three "
        "petroleum products are also correlated -> the independence robustness is weak. Validate via "
        "the write-once 2022+ holdout (forward-validation) + MCPT; TREAT ANY IN-SEARCH PASS WITH "
        "SUSPICION. WEAKNESSES: (1) 'surprise' is a MODEL residual, not analyst-consensus (we do not "
        "own consensus) -> weaker, prediction-edge tail risk. (2) Only 3 sleeves -> the deployment "
        "single-name-share / sector-spread gates are tight by construction, a real limitation of a "
        "small storable-energy book, not a bug. (3) STANDALONE test only; per the 2026-06-08 "
        "over-blend lesson the 21-market trend tail-overlay is a SEPARATE future combination, not part "
        "of this frozen design. COSTS: 5bps per unit turnover (liquid energy futures incl slippage); "
        "inverse-vol sizing, book vol-target ~10%, weekly rebalance, signals lagged. CHECKABLE: "
        "hysteresis is expected to lower the number of position-change runs vs a no-band variant "
        "(declared as a soft expectation)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "no_hyst": {"hysteresis": 0.0},
        "tight_band": {"hysteresis": 1.0},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=3,
    expectations=[
        {"name": "hysteresis_reduces_turnover",
         "claim": "default (+/-0.5sigma hysteresis) yields <= the no-band variant's position-change runs (search window)",
         "check": _exp_hysteresis_turnover},
    ],
)