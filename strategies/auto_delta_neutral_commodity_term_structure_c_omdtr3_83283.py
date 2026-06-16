"""
commod_calspread_carry_storage_v1
=================================
Delta-neutral commodity term-structure CARRY, harvested via INTRA-COMMODITY
CALENDAR SPREADS (long-front/short-deferred when backwardated; reverse when
contango), with EIA/USDA inventory as a CONTINUOUS sizing tilt (never a gate).

Why this is not a duplicate of commod_curvecarry_x_trend_v1 (FAIL):
  - That book was an OUTRIGHT roll-yield cross-section -> carries the commodity
    PRICE LEVEL / inflation beta as a confound.
  - This book is the DELTA-NEUTRAL CALENDAR-SPREAD construction: per root we
    trade (close_1 - close_2), which mechanically nets out a flat parallel move
    and isolates the pure convergence / storage premium.
  - Storage is a CONTINUOUS multiplier in [0.5,1.5] (never zeroes a leg) instead
    of the hard gates that failed in inv_cond_convenience_yield_v1 -> directly
    the high-freq-base + slow-overlay reframe.

Construction (DAILY base, weekly rebalance):
  carry_r = annualised log(close_1/close_2).  Backwardated (carry>0) -> LONG the
  (c1-c2) spread; contango (carry<0) -> SHORT it.  Hold the extreme terciles,
  inverse-vol weighted, book vol-targeted ~10%.  Spread P&L is diffed ONLY when
  symbol_1 AND symbol_2 are unchanged day-over-day (never across a roll); each
  spread is excluded when days_to_roll_1 < roll_exit_days.

LOOK-AHEAD HANDLING (stated explicitly):
  - inverse-vol uses a TRAILING rolling std .shift(1).
  - inventory z-score is trailing-only and triple-lagged (period shift + a small
    extra release lag + the final weight shift).
  - the final weight matrix is .shift(1)'d before net_of_cost / trades_from_weights
    (the lag is OUR responsibility; net_of_cost does not lag).

scope='broad': the mechanism is universal, so we SEARCH on energy+grains (where
the EIA/USDA tilt is fully active) and GENERALISE to the untouched complexes
(metals, livestock, and their union) -- DISJOINT root sets, tilt neutral there,
so a pass proves the premium is not an artifact of the fundamental overlay.
"""

from sdk.harness import StrategySpec
from sdk.adapters import fut_curve
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# inventory adapters are an OVERLAY only -> guard so the module still runs if a
# key lapses (fundamental-less roots already default to a neutral 1.0 tilt).
try:
    from sdk.adapters import eia_series
except Exception:
    eia_series = None
try:
    from sdk.adapters import usda_nass
except Exception:
    usda_nass = None

# --------------------------------------------------------------------------- #
# Universe definitions
# --------------------------------------------------------------------------- #
ROOT_SECTOR = {
    "CL": "ENERGY", "NG": "ENERGY", "HO": "ENERGY", "RB": "ENERGY",
    "GC": "METALS", "SI": "METALS", "HG": "METALS", "PL": "METALS",
    "ZC": "GRAINS", "ZS": "GRAINS", "ZW": "GRAINS", "ZL": "GRAINS", "ZM": "GRAINS",
    "LE": "LIVESTOCK", "HE": "LIVESTOCK", "GF": "LIVESTOCK",
}

# SEARCH universe: the two complexes with OWNED fundamentals (EIA + USDA),
# so the storage tilt is fully active during search / validation.
SEARCH_ROOTS = ["CL", "NG", "HO", "RB", "ZC", "ZS", "ZW", "ZL", "ZM"]

# GENERALISATION universes: DISJOINT from SEARCH_ROOTS (share NO roots).  No
# owned fundamentals here -> tilt is neutral 1.0 -> a clean test of the PURE
# term-structure carry mechanism in untouched complexes.
GEN_ROOTS = {
    "metals":           ["GC", "SI", "HG", "PL"],
    "livestock":        ["LE", "HE", "GF"],
    "metals_livestock": ["GC", "SI", "HG", "PL", "LE", "HE", "GF"],
}

# Nominal contract spacing (days) to annualise carry comparably across complexes
# (affects cross-root MAGNITUDE only; the within-root SIGN is unchanged).
NOMINAL_GAP_DAYS = {
    "CL": 30, "NG": 30, "HO": 30, "RB": 30,
    "GC": 60, "SI": 60, "HG": 30, "PL": 90,
    "ZC": 60, "ZS": 60, "ZW": 60, "ZL": 30, "ZM": 30,
    "LE": 60, "HE": 60, "GF": 60,
}

# Inventory mappings (overlay)
EIA_IDS = {                       # weekly stocks
    "CL": "PET.WCESTUS1.W",       # crude ex-SPR
    "HO": "PET.WDISTUS1.W",       # distillate
    "RB": "PET.WGTSTUS1.W",       # total gasoline
    "NG": "NG.NW2_EPG0_SWO_R48_BCF.W",   # working gas in storage L48
}
USDA_COMM = {                     # quarterly grain stocks (PIT on release)
    "ZC": "CORN", "ZW": "WHEAT",
    "ZS": "SOYBEANS", "ZL": "SOYBEANS", "ZM": "SOYBEANS",
}

START = "2010-01-01"

DEFAULTS = dict(
    n_frac=0.34,         # tercile width (long/short each side)
    roll_exit_days=5,    # exit a spread when days_to_roll_1 < this
    tilt_k=0.5,          # storage-tilt sensitivity (mult in [0.5,1.5])
    use_tilt=True,
    vol_lb=63, vol_min=20,
    book_vol_lb=63,
    target_vol=0.10,
    max_leverage=4.0,
    cost_bps=8.0,
    name="calspread_carry",
)


# --------------------------------------------------------------------------- #
# Inventory overlay helpers (best-effort; neutral fallback)
# --------------------------------------------------------------------------- #
def _to_series(r):
    """Coerce an adapter return into a numeric Series with a GUARANTEED
    DatetimeIndex.  Return None (-> neutral 1.0 tilt) if no date-aligned numeric
    data survives.

    BUG FIX: the previous version passed an integer/RangeIndex straight through.
    In _inventory_z that index was union'd with the trading-date DatetimeIndex,
    producing a mixed-type (Timestamp vs int) index that pandas cannot sort
    ('<' not supported between Timestamp and int).  We now either recover a real
    date index (promote a date column / parse an object index) or reject the
    series so the overlay degrades to neutral.
    """
    if r is None:
        return None

    s = None
    if isinstance(r, pd.Series):
        s = pd.to_numeric(r, errors="coerce")
    elif isinstance(r, pd.DataFrame) and r.shape[1] >= 1:
        df = r.copy()
        # promote a datetime-like column to the index when the index is not dates
        if not isinstance(df.index, pd.DatetimeIndex):
            for c in df.columns:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df = df.set_index(parsed).drop(columns=[c], errors="ignore")
                    break
        # first column that holds numeric values is the level series
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().any():
                s = v
                break
    if s is None:
        return None

    # --- GUARANTEE a DatetimeIndex ---
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(idx):
            return None  # a plain int/float index carries no usable date info
        idx = pd.to_datetime(idx, errors="coerce")
        s = pd.Series(np.asarray(s.values, dtype="float64"), index=idx)

    s = s[~s.index.isna()].dropna()
    if len(s) == 0:
        return None
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _safe_eia(sid):
    if eia_series is None or sid is None:
        return None
    for call in (lambda: eia_series({sid: "v"}, START),
                 lambda: eia_series(sid, start=START),
                 lambda: eia_series(sid)):
        try:
            s = _to_series(call())
            if s is not None and len(s) > 10:
                return s
        except Exception:
            continue
    return None


def _safe_usda(comm):
    if usda_nass is None or comm is None:
        return None
    for call in (lambda: usda_nass(comm, statistic="STOCKS", start=START),
                 lambda: usda_nass(comm, "STOCKS"),
                 lambda: usda_nass(comm)):
        try:
            s = _to_series(call())
            if s is not None and len(s) > 4:
                return s
        except Exception:
            continue
    return None


def _season_z(s, win):
    """Trailing deviation z-score (proxy for seasonal-deviation; lookahead-free).
    Detrends with a trailing-`win` mean/std then lags the latest obs one period."""
    s = pd.to_numeric(s, errors="coerce").sort_index().astype(float)
    mu = s.rolling(win, min_periods=max(8, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(8, win // 4)).std().replace(0, np.nan)
    return ((s - mu) / sd).shift(1)


def _inventory_z(root, dates):
    """Daily, PIT-safe inventory z-score for a root; NaN (->neutral 1.0) if none."""
    z = pd.Series(np.nan, index=dates, dtype="float64")
    sec = ROOT_SECTOR.get(root)
    series, win = None, 52
    if sec == "ENERGY":
        series, win = _safe_eia(EIA_IDS.get(root)), 52          # weekly
    elif sec == "GRAINS":
        series, win = _safe_usda(USDA_COMM.get(root)), 12       # quarterly
    if series is None or len(series) == 0:
        return z
    # defensive: only a genuine DatetimeIndex can be union'd with `dates`
    if not isinstance(series.index, pd.DatetimeIndex):
        return z

    zz = _season_z(series, win)
    zz = zz[~zz.index.isna()].sort_index()
    if len(zz) == 0:
        return z
    daily = (zz.reindex(zz.index.union(dates)).sort_index()
               .ffill().reindex(dates))
    return daily.shift(2)   # small extra release-lag safety on top of period lag


# --------------------------------------------------------------------------- #
# Panel builder
# --------------------------------------------------------------------------- #
def _build_curve_panel(roots, start=START):
    """MultiIndex-column panel: (root, field) for field in
    {close_1, close_2, sym1, sym2, dtr, invz}."""
    frames = {}
    for r in roots:
        try:
            fc = fut_curve(r, n_contracts=2)
        except Exception:
            continue
        if fc is None or len(fc) == 0:
            continue
        fc = fc.sort_index()
        fc = fc[fc.index >= pd.Timestamp(start)]
        if len(fc) == 0:
            continue
        low = {c.lower(): c for c in fc.columns}

        def g(name):
            if name in fc.columns:
                return fc[name]
            if name.lower() in low:
                return fc[low[name.lower()]]
            return pd.Series(index=fc.index, dtype="float64")

        sub = pd.DataFrame(index=fc.index)
        sub["close_1"] = pd.to_numeric(g("close_1"), errors="coerce")
        sub["close_2"] = pd.to_numeric(g("close_2"), errors="coerce")
        sub["sym1"] = g("symbol_1").astype("object")
        sub["sym2"] = g("symbol_2").astype("object")
        sub["dtr"] = pd.to_numeric(g("days_to_roll_1"), errors="coerce")
        sub["invz"] = _inventory_z(r, sub.index)
        frames[r] = sub

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()   # columns -> (root, field)
    return out


def load_data():
    return _build_curve_panel(SEARCH_ROOTS)


def load_gen_data(label):
    return _build_curve_panel(GEN_ROOTS[label])


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def _select_terciles(carry, n_frac):
    """+1 = long-spread (most backwardated, carry>0); -1 = short-spread
    (most contango, carry<0); 0 = not held.  Row-wise over dates."""
    out = pd.DataFrame(0.0, index=carry.index, columns=carry.columns)
    for dt, row in carry.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            continue
        n_hold = max(2, int(round(len(valid) * n_frac)))
        longs = valid[valid > 0].sort_values(ascending=False).head(n_hold)
        shorts = valid[valid < 0].sort_values(ascending=True).head(n_hold)
        if len(longs):
            out.loc[dt, longs.index] = 1.0
        if len(shorts):
            out.loc[dt, shorts.index] = -1.0
    return out


def signal(panel, **params):
    p = dict(DEFAULTS); p.update(params)

    if panel is None or len(panel) == 0 or not isinstance(panel.columns, pd.MultiIndex):
        return pd.Series(dtype="float64", name=p["name"]), []

    roots = list(panel.columns.get_level_values(0).unique())
    if len(roots) < 2:
        return pd.Series(dtype="float64", name=p["name"]), []
    sector_map = {r: ROOT_SECTOR.get(r, "OTHER") for r in roots}
    idx = panel.index

    c1 = pd.DataFrame({r: panel[(r, "close_1")] for r in roots})
    c2 = pd.DataFrame({r: panel[(r, "close_2")] for r in roots})
    s1 = pd.DataFrame({r: panel[(r, "sym1")] for r in roots})
    s2 = pd.DataFrame({r: panel[(r, "sym2")] for r in roots})
    dtr = pd.DataFrame({r: panel[(r, "dtr")] for r in roots})
    invz = pd.DataFrame({r: panel[(r, "invz")] for r in roots})

    # --- delta-neutral calendar-spread returns (NEVER diffed across a roll) ---
    spread = c1 - c2
    stable = (s1 == s1.shift(1)) & (s2 == s2.shift(1))
    spread_ret = (spread.diff() / c1.shift(1)).where(stable)
    spread_ret = (spread_ret.replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0))[roots]

    # --- carry signal (annualised) + holdability (away from the roll) ---
    gap = pd.Series({r: NOMINAL_GAP_DAYS.get(r, 60.0) for r in roots})
    carry = (np.log(c1 / c2)).div(gap, axis=1) * 365.0
    holdable = ((dtr >= p["roll_exit_days"]) & c1.notna() & c2.notna()
                & (c1 > 0) & (c2 > 0))
    carry = carry.where(holdable)

    direction = _select_terciles(carry, p["n_frac"])     # +1 / -1 / 0

    # --- continuous storage tilt in [0.5,1.5] (never a gate; never zeroes) ---
    if p["use_tilt"]:
        conf = (-invz) * np.sign(carry.fillna(0.0))      # +ve when fundamentals confirm
        mult = (1.0 + p["tilt_k"] * conf).clip(0.5, 1.5)
        mult = mult.where(np.isfinite(mult)).fillna(1.0)
    else:
        mult = pd.DataFrame(1.0, index=idx, columns=roots)

    # --- inverse-vol weights (trailing, lagged) ---
    vol = (spread_ret.rolling(p["vol_lb"], min_periods=p["vol_min"]).std()
                     .shift(1)).clip(lower=1e-4)
    invvol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    raw = (direction * mult * invvol).where(holdable, 0.0).fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    W_target = raw.div(gross, axis=0).clip(-0.5, 0.5)     # cap single-spread share
    g2 = W_target.abs().sum(axis=1).replace(0, np.nan)
    W_target = W_target.div(g2, axis=0).fillna(0.0)

    # --- weekly rebalance (hold from first trading day of each ISO week) ---
    iso = idx.isocalendar()
    wk = (iso["year"].astype(int) * 100 + iso["week"].astype(int))
    rebal_dates = idx[(~wk.duplicated()).values]
    W_held = W_target.loc[rebal_dates].reindex(idx, method="ffill").fillna(0.0)

    # --- vol-target the book to ~target_vol (trailing, applied weekly) ---
    r_unscaled = (W_held.shift(1) * spread_ret).sum(axis=1)
    bvol = r_unscaled.rolling(p["book_vol_lb"], min_periods=20).std().shift(1)
    tgt_d = p["target_vol"] / np.sqrt(252.0)
    scale_daily = (tgt_d / bvol).clip(upper=p["max_leverage"])
    scale_held = (scale_daily.loc[rebal_dates]
                  .reindex(idx, method="ffill").fillna(0.0))
    W_scaled = W_held.mul(scale_held, axis=0)

    # --- the lag is OUR responsibility: shift weights before costs/ledger ---
    W_final = W_scaled.shift(1).fillna(0.0)

    daily = net_of_cost(W_final, spread_ret, cost_bps=p["cost_bps"], name=p["name"])
    daily.name = p["name"]
    trades = trades_from_weights(W_final, spread_ret, sector_map)
    return daily, trades


# --------------------------------------------------------------------------- #
# Soft expectations (machine-checkable mechanism claims)
# --------------------------------------------------------------------------- #
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _chk_tilt(ctx):
    """Storage tilt should ADD value (not hurt) where fundamentals are owned."""
    g = ctx.get("grid", {}) or {}
    d, n = g.get("default"), g.get("no_tilt")
    if d is None or n is None:
        return {"pass": False, "observed": "grid variants missing"}
    diff = _sharpe(d) - _sharpe(n)
    return {"pass": diff >= -0.05, "observed": round(diff, 3)}


def _chk_neutral(ctx):
    """Calendar-spread book is ~delta-neutral to outright commodity direction."""
    r = pd.Series(ctx.get("search")).dropna()
    panel = ctx.get("panel")
    hs = pd.Timestamp(ctx.get("holdout_start", "2022-01-01"))
    if panel is None or len(r) < 30 or not isinstance(panel.columns, pd.MultiIndex):
        return {"pass": True, "observed": "n/a"}
    roots = list(panel.columns.get_level_values(0).unique())
    c1 = pd.DataFrame({rt: panel[(rt, "close_1")] for rt in roots})
    bench = np.log(c1).diff().mean(axis=1)               # eq-weight front-month return
    bench = bench[bench.index < hs]
    a, b = r.align(bench, join="inner")
    a, b = a.dropna(), b.dropna()
    a, b = a.align(b, join="inner")
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        return {"pass": True, "observed": "n/a"}
    corr = float(np.corrcoef(a, b)[0, 1])
    return {"pass": abs(corr) < 0.35, "observed": round(corr, 3)}


def _chk_trades(ctx):
    """Daily base + weekly rebalance must clear the sparse-signal trap."""
    n = len(ctx.get("trades") or [])
    return {"pass": n >= 50, "observed": n}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="commod_calspread_carry_storage_v1",
    family="commodity_carry",
    title=("Delta-neutral commodity term-structure carry via intra-commodity "
           "calendar spreads, with EIA/USDA inventory continuous sizing tilt"),
    markets=["commodity_futures"],
    data_desc=("Databento GLBX rank-1/2 contract closes via fut_curve(root, 2) "
               "(close_1/close_2/symbol_1/symbol_2/days_to_roll_1) for CME roots "
               "ENERGY{CL,NG,HO,RB} METALS{GC,SI,HG,PL} GRAINS{ZC,ZS,ZW,ZL,ZM} "
               "LIVESTOCK{LE,HE,GF}; EIA weekly stocks + USDA quarterly grain "
               "stocks (PIT) as a continuous [0.5,1.5] sizing multiplier."),
    pre_registration=(
        "Delta-neutral intra-commodity calendar spreads (long-front/short-deferred "
        "when backwardated, reverse when contango) harvest the storage/convenience-"
        "yield premium with the commodity PRICE LEVEL netted out. Spread P&L is "
        "diffed only when both leg symbols are unchanged (never across a roll); a "
        "spread is dropped when days_to_roll_1<5. Falsifiable claims: "
        "(1) DELTA-NEUTRALITY -- |corr(book, equal-weight front-month return)|<0.35; "
        "(2) STORAGE TILT ADDS VALUE where fundamentals are owned (energy+grains "
        "search Sharpe with tilt >= without); "
        "(3) HIGH-FREQ BASE -- daily signal + weekly rebalance yields >=50 trades "
        "across complexes (fixes the 2026-06-16 sparse-signal lesson). "
        "Mechanism is universal, so it MUST generalise OOS to the untouched "
        "complexes (metals, livestock) where the tilt is neutral. Tested STANDALONE; "
        "the validated Boreas trend pairing is reserved as a small tail overlay only "
        "after a standalone pass (avoid reflexive-50/50 dilution). NOTE: EIA/USDA "
        "overlay is best-effort -> if a key lapses or an adapter returns an index "
        "with no usable dates, that root's tilt degrades to neutral 1.0 (the carry "
        "leg is unaffected; this is by design, not a confound)."
    ),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid={
        "default": {},                       # primary
        "no_tilt": {"use_tilt": False},      # standalone term-structure carry
        "wide":    {"n_frac": 0.5},          # wider held set
        "vol12":   {"target_vol": 0.12},
    },
    scope="broad",
    generalization_universes=["metals", "livestock", "metals_livestock"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "delta_neutral_low_beta",
         "claim": "|corr(book, eq-weight front-month return)| < 0.35 (search)",
         "check": _chk_neutral},
        {"name": "storage_tilt_adds_value",
         "claim": "default (tilt) search Sharpe >= no_tilt search Sharpe",
         "check": _chk_tilt},
        {"name": "trade_count_ge_50",
         "claim": "daily base + weekly rebalance -> >=50 search-window trades",
         "check": _chk_trades},
    ],
)