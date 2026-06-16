"""
Commodity convenience-yield DYNAMICS — cross-sectional roll-yield MOMENTUM (carry-change).

Mechanism (pre-registered): the CHANGE in the front-curve slope (carry) is a forward-looking
physical-scarcity premium. Commodities whose curve is STEEPENING into backwardation (rising
convenience yield) out-earn those steepening into contango. This is the carry-*velocity* object,
distinct from the static carry LEVEL (commodity_xs_carry_spotbasis) and from the spread-of-
cumulative-returns basis_momentum (Boons-Prado).

The ONLY novel code here is the signal. Universe build, z-scoring, costs, the trade ledger and
its entry_regime stamping all come from the tested kit.
"""

from sdk.harness import StrategySpec
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# fut_curve is the OWNED Databento GLBX contract-month curve adapter referenced by the proposal /
# DATA_CATALOG and verified in gate0 (returns close_1, close_2, days_to_roll_1). It is not in the
# common adapter shortlist, so import it guardedly: if it is genuinely absent the module still
# imports and load_data() returns an empty panel (the harness reports "no data" instead of crashing).
try:
    from sdk.adapters import fut_curve
except Exception:  # pragma: no cover
    fut_curve = None

# ----------------------------------------------------------------------------------------------
# Universe (search) — the owned 16-root liquid complex (PA dropped: thin rank-2).
# ----------------------------------------------------------------------------------------------
COMPLEX = {
    "CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
    "ZC": "grains", "ZS": "grains", "ZW": "grains", "ZL": "grains", "ZM": "grains",
    "LE": "livestock", "HE": "livestock", "GF": "livestock",
}
ROOTS = list(COMPLEX.keys())
START = "2010-01-01"

# Generalization (scope='broad'): the universal carry-momentum premium, of which commodity
# convenience-yield dynamics is the physical instance, should also appear in OTHER term-structure
# complexes. These three baskets are UNTOUCHED and ticker-DISJOINT from the 16 commodity roots
# (a true OOS battery — within-commodity sub-complexes share search tickers and so are not valid
# stage-2 universes; with only 16 owned liquid roots no meaningful ticker-disjoint commodity
# sub-universe exists, so we generalize cross-asset on the SAME frozen signal). Loaded robustly:
# any root fut_curve cannot serve is skipped, so a thin/missing basket fails OOS rather than crashing.
GEN_ROOTS = {
    "fx_carry":     ["6E", "6J", "6B", "6A", "6C", "6S", "6N"],   # G10 FX futures
    "rates_carry":  ["ZT", "ZF", "ZN", "ZB", "UB"],               # US Treasury futures curve
    "equity_carry": ["ES", "NQ", "YM", "RTY", "EMD"],             # US equity-index futures
}
GEN_SECTOR = {r: u for u, rs in GEN_ROOTS.items() for r in rs}
SECTOR_MAP = {**COMPLEX, **GEN_SECTOR}

DEFAULT = dict(
    signal_mode="change",   # 'change' = PRIMARY carry-momentum ; 'level' = static-carry robustness read
    lookback=63,            # 63-trading-day change of the roll-yield (carry velocity)
    smooth=21,              # EWMA span on the change (denoise)
    carry_smooth=5,         # light EWMA on the raw carry level before differencing
    vol_lb=60,              # realized-vol lookback (inverse-vol leg sizing + book vol target)
    target_vol=0.10,        # 10% annualized book vol target
    max_leverage=3.0,
    tercile=1.0 / 3.0,      # long top tercile / short bottom tercile
    hysteresis=0.10,        # membership stickiness band (rank-fraction) to cap turnover
    rebalance="W-FRI",      # weekly rebalance
    cost_bps=8.0,           # conservative: liquid futures round-trip is ~2-3bps; 8bps overstates costs
    name="cvy_carry_mom",
)

GRID = {
    "default":       {},
    "lookback_42":   {"lookback": 42},
    "lookback_84":   {"lookback": 84},
    "smooth_10":     {"smooth": 10},
    "no_hysteresis": {"hysteresis": 0.0},
}

# ----------------------------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------------------------
def _col(df, names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def _curve_frame(root):
    """One root -> DataFrame[date] with columns ['carry','ret'].
    carry = log(front/second) front-curve slope (backwardation > 0); NaN where rank-2 absent.
    ret   = within-contract FRONT return (roll discontinuities removed; never diffed across a roll).
    Returns None on any failure so the panel build is robust per-root.
    """
    if fut_curve is None:
        return None
    try:
        df = fut_curve(root, n_contracts=2)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    c1 = _col(df, ["close_1", "close1", "c1", "px_1", "settle_1", "front"])
    c2 = _col(df, ["close_2", "close2", "c2", "px_2", "settle_2", "second"])
    if c1 is None or c2 is None:
        return None
    close1 = pd.to_numeric(df[c1], errors="coerce")
    close2 = pd.to_numeric(df[c2], errors="coerce")

    # carry = front-second log slope. NOT annualized by days_to_roll: dividing by time-to-front-
    # expiry injects a deterministic roll-cycle sawtooth (factor blows up as expiry nears) that
    # would dominate the velocity signal; the cross-sectional rank of the CHANGE is invariant to a
    # common positive scale, so the un-annualized slope is the clean object.
    carry = np.log(close1.where(close1 > 0)) - np.log(close2.where(close2 > 0))

    # within-contract front return -----------------------------------------------------------
    rcol = _col(df, ["ret_1", "return_1", "front_ret", "ret"])
    if rcol is not None:
        ret = pd.to_numeric(df[rcol], errors="coerce")
    else:
        ret = close1.pct_change()
        dtr_col = _col(df, ["days_to_roll_1", "days_to_roll", "dte_1", "days_to_expiry_1", "dte"])
        if dtr_col is not None:
            dtr = pd.to_numeric(df[dtr_col], errors="coerce")
            roll = dtr.diff() > 0.5                       # time-to-expiry jumped up => new front
            rstd = ret.rolling(20, min_periods=5).std()
            ret = ret.mask(roll & (ret.abs() > 5.0 * rstd), 0.0)   # zero only abnormal roll jumps
    ret = ret.where(ret.abs() <= 0.5, 0.0)               # global safety clip on any residual jump
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = pd.DataFrame({"carry": carry, "ret": ret})
    return out.dropna(how="all")


def _build_panel(roots):
    carry, ret = {}, {}
    for r in roots:
        f = _curve_frame(r)
        if f is None:
            continue
        carry[r] = f["carry"]
        ret[r] = f["ret"]
    if not carry:
        return pd.DataFrame()
    C = pd.DataFrame(carry).sort_index()
    R = pd.DataFrame(ret).reindex(C.index)
    panel = pd.concat({"carry": C, "ret": R}, axis=1)
    panel = panel.loc[panel.index >= pd.Timestamp(START)]
    panel.index.name = "date"
    return panel


def load_data() -> pd.DataFrame:
    return _build_panel(ROOTS)


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(GEN_ROOTS.get(label, []))


# ----------------------------------------------------------------------------------------------
# Signal helpers (the novel code)
# ----------------------------------------------------------------------------------------------
def _rebal_mask(idx, freq):
    per = pd.Series(pd.PeriodIndex(idx, freq=freq).astype(str), index=idx)
    m = per != per.shift(1)
    if len(m):
        m.iloc[0] = True
    return m


def _select(z, rebal_days, tercile, buffer):
    """Per-rebalance tercile selection with MEMBERSHIP HYSTERESIS (buffer in rank-fraction):
    a name stays in its leg until its cross-sectional rank decays 'buffer' past the entry band.
    Returns a {-1,0,+1} frame on rebal_days. Falls back to fresh count-terciles if a leg empties."""
    cols = list(z.columns)
    sel = pd.DataFrame(0.0, index=rebal_days, columns=cols)
    state = {c: 0 for c in cols}
    for dt in rebal_days:
        s = z.loc[dt].dropna()
        n = len(s)
        if n < 3:
            state = {c: 0 for c in cols}
            continue
        r = s.rank(pct=True)
        long_enter, long_stay = 1.0 - tercile, 1.0 - tercile - buffer
        short_enter, short_stay = tercile, tercile + buffer
        new = {c: 0 for c in cols}
        for c in r.index:
            f = r[c]
            cur = state.get(c, 0)
            if cur == 1:
                new[c] = 1 if f >= long_stay else 0
            elif cur == -1:
                new[c] = -1 if f <= short_stay else 0
            else:
                new[c] = 1 if f >= long_enter else (-1 if f <= short_enter else 0)
        longs = [c for c in new if new[c] == 1]
        shorts = [c for c in new if new[c] == -1]
        if not longs or not shorts:                      # never run a one-sided book
            k = max(1, min(int(round(n * tercile)), n // 2))
            new = {c: 0 for c in cols}
            for c in s.nlargest(k).index:
                new[c] = 1
            for c in s.nsmallest(k).index:
                new[c] = -1
        state = new
        sel.loc[dt] = pd.Series(new).reindex(cols).fillna(0.0).values
    return sel


def _weights(sel, iv, cols):
    Wr = pd.DataFrame(0.0, index=sel.index, columns=cols)
    for dt, row in sel.iterrows():
        longs = [c for c in cols if row[c] > 0]
        shorts = [c for c in cols if row[c] < 0]
        if not longs or not shorts:
            continue
        ivl = iv.loc[dt, longs].clip(lower=0).fillna(0.0)
        ivs = iv.loc[dt, shorts].clip(lower=0).fillna(0.0)
        wl = (ivl / ivl.sum()) if ivl.sum() > 0 else pd.Series(1.0 / len(longs), index=longs)
        ws = (ivs / ivs.sum()) if ivs.sum() > 0 else pd.Series(1.0 / len(shorts), index=shorts)
        Wr.loc[dt, longs] = 0.5 * wl.values              # each leg gross 0.5 -> dollar-neutral, gross 1
        Wr.loc[dt, shorts] = -0.5 * ws.values
    return Wr


def signal(panel, **params):
    p = dict(DEFAULT)
    p.update(params)
    name = p["name"]
    empty = pd.Series(dtype=float, name=name)
    empty.index = pd.DatetimeIndex([])

    if panel is None or len(panel) == 0 or not isinstance(panel.columns, pd.MultiIndex):
        return empty, []
    if "carry" not in panel.columns.get_level_values(0):
        return empty, []

    C = panel["carry"].astype(float)
    R = panel["ret"].astype(float).reindex(C.index).fillna(0.0)
    cols = list(C.columns)
    if len(cols) < 3:
        return empty, []
    sector_map = {c: SECTOR_MAP.get(c, "other") for c in cols}

    # --- carry-velocity signal -----------------------------------------------------------------
    carry_s = C.ewm(span=p["carry_smooth"], min_periods=1).mean()
    if p["signal_mode"] == "level":
        sig = carry_s                                    # static carry LEVEL (robustness comparator)
    else:
        sig = (carry_s - carry_s.shift(p["lookback"])).ewm(span=p["smooth"], min_periods=1).mean()
    z = xs_zscore(sig)                                   # cross-sectional, winsorized, NaN-preserving

    # --- weekly tercile L/S, inverse-vol legs --------------------------------------------------
    vol = R.rolling(p["vol_lb"], min_periods=20).std()
    iv = 1.0 / vol.replace(0.0, np.nan)
    idx = z.index
    rebal = _rebal_mask(idx, p["rebalance"])
    rebal_days = idx[rebal.values]

    sel = _select(z, rebal_days, p["tercile"], p["hysteresis"])
    Wr = _weights(sel, iv, cols)                         # unscaled weights on rebal days
    U = Wr.reindex(idx).ffill().fillna(0.0)              # weekly-constant unscaled book

    # --- book vol targeting (weekly-held, lagged) ----------------------------------------------
    book = (U.shift(1) * R).sum(axis=1)
    rv = book.rolling(p["vol_lb"], min_periods=20).std() * np.sqrt(252.0)
    scale = (p["target_vol"] / rv.replace(0.0, np.nan)).clip(upper=p["max_leverage"])
    scale_wk = scale.where(rebal).ffill().fillna(0.0).clip(upper=p["max_leverage"])
    target = U.mul(scale_wk, axis=0)

    # --- NO LOOK-AHEAD: target weights are decided from info through close t; the executed/held
    # weights earning return on day t are target.shift(1). This single shift is the mandated 1-day
    # lag and is OUR responsibility (net_of_cost receives the already-lagged matrix). The vol scale
    # is itself weekly-held from rv computed on U.shift(1)*R, so after the final shift it relies
    # strictly on information available before the holding day.
    held = target.shift(1).fillna(0.0)

    daily = net_of_cost(held, R, cost_bps=p["cost_bps"], name=name)
    trades = trades_from_weights(held, R, sector_map)    # kit stamps entry_regime (do not hand-roll)
    return daily, trades


# ----------------------------------------------------------------------------------------------
# Soft expectations (machine-checkable mechanism claims)
# ----------------------------------------------------------------------------------------------
def _mean_hold(trades, ho):
    h = [t["hold_days"] for t in trades
         if t.get("entry_date") and pd.Timestamp(t["entry_date"]) < ho]
    return float(np.mean(h)) if h else float("nan")


def _chk_distinct_from_level(ctx):
    """Carry-CHANGE is a different object than static carry LEVEL: search-window daily-return
    correlation between the two books should be modest (< 0.6)."""
    try:
        ho = pd.Timestamp(ctx["holdout_start"])
        base = ctx["search"].dropna()
        base = base[base.index < ho]
        lvl, _ = signal(ctx["panel"], signal_mode="level")     # <= one extra signal() call
        lvl = lvl.dropna()
        lvl = lvl[lvl.index < ho]
        a, b = base.align(lvl, join="inner")
        if len(a) < 60:
            return {"pass": False, "observed": "insufficient_overlap"}
        c = float(a.corr(b))
        return {"pass": bool(abs(c) < 0.6), "observed": round(c, 3)}
    except Exception as e:
        return {"pass": False, "observed": "err:%s" % type(e).__name__}


def _chk_hysteresis_cuts_turnover(ctx):
    """Membership hysteresis should LENGTHEN average holds (lower turnover) vs no-hysteresis.
    Falsifiable so a wrong turnover story is recorded, not shipped as prose (2026-06-12 lesson)."""
    try:
        ho = pd.Timestamp(ctx["holdout_start"])
        hyst_hold = _mean_hold(ctx["trades"], ho)              # default = hysteresis ON
        _, no_tr = signal(ctx["panel"], hysteresis=0.0)        # <= one extra signal() call
        noh_hold = _mean_hold(no_tr, ho)
        ok = np.isfinite(hyst_hold) and np.isfinite(noh_hold) and hyst_hold >= noh_hold
        return {"pass": bool(ok),
                "observed": "hold_days hyst=%.1f none=%.1f" % (hyst_hold, noh_hold)}
    except Exception as e:
        return {"pass": False, "observed": "err:%s" % type(e).__name__}


# ----------------------------------------------------------------------------------------------
PRE_REG = """
HYPOTHESIS. Convenience yield is the non-monetary benefit of holding the physical commodity; it
shows up in the front-curve slope (backwardation => high convenience yield => physical scarcity).
The STATIC level of that slope is the well-documented commodity carry premium. This strategy bets
instead on the CHANGE (velocity) of the slope: roots whose curve is STEEPENING toward backwardation
(convenience yield RISING -> market tightening) should out-earn roots steepening toward contango
(convenience yield FALLING -> market loosening). The change is forward-looking about scarcity in a
way the level is not, and is mechanically DISTINCT from (a) static carry LEVEL and (b) basis-
momentum (Boons-Prado), which is the spread of cumulative front/second RETURNS, not the change of
the slope itself.

SIGNAL. For each root: carry_t = log(close_1) - log(close_2), the front-second log slope, NOT
annualized by time-to-expiry (annualization injects a deterministic roll-cycle sawtooth; the
cross-sectional RANK of the change is invariant to a common positive scale, so the un-annualized
slope is the clean object). Lightly EWMA-smooth the level (span 5), take its 63-day change, then
EWMA-smooth the change (span 21). Cross-sectionally winsorized z-score. Weekly (W-FRI) tercile
long/short with membership HYSTERESIS (rank-fraction stickiness band) to cap turnover, inverse-vol
leg sizing, dollar-neutral (each leg gross 0.5), 10% annualized book-vol target capped at 3x
leverage, 8bps round-trip cost (liquid futures are ~2-3bps; 8 is deliberately conservative). All
signals lagged 1 day (held = target.shift(1)); the vol scale is weekly-held off the lagged book, so
nothing uses same-day return information. FRONT returns are within-contract (abnormal roll jumps
zeroed via days_to_roll_1), never differenced across a roll.

SCOPE = broad. Carry-momentum is posited as a UNIVERSAL term-structure premium; convenience-yield
dynamics is its physical-commodity instance. With only ~16 owned liquid commodity roots there is no
ticker-disjoint commodity sub-universe large enough for a valid stage-2 battery, so the SAME frozen
signal + default params generalizes CROSS-ASSET to three untouched, ticker-DISJOINT term-structure
complexes: G10 FX futures, US Treasury futures, and US equity-index futures. Per the rails, >=60% of
the generalization universes must be OOS-positive on their holdout or the edge is rejected as a
commodity-specific outlier rather than a real premium.

FALSIFIERS / EXPECTATIONS (machine-checked, recorded not blocking): (1) the carry-CHANGE book is a
DISTINCT object from a static carry-LEVEL book (search-window daily-return correlation < 0.6); a
high correlation would mean we merely re-discovered static carry. (2) Membership hysteresis
LENGTHENS mean hold_days vs the no-hysteresis variant (the turnover-control mechanism actually
works). Soft fails falsify the mechanism story instead of shipping it as prose (2026-06-12 lesson).

SEARCH BURDEN. PRIMARY = default params. Grid for honest DSR effective-N spans lookback {42,63,84},
smooth {10,21}, hysteresis {0,0.10}. The holdout from 2022-01-01 is untouched until a stage-1 pass.
"""

SPEC = StrategySpec(
    id="auto_commodity_convenience_yield_dynamics_cro_omdtr3_80629",
    family="commodity_carry",
    title="Commodity convenience-yield dynamics — cross-sectional roll-yield momentum (carry velocity)",
    markets=["commodity_futures", "fx_futures", "rates_futures", "equity_index_futures"],
    data_desc=("OWNED Databento GLBX front-curve (close_1, close_2, days_to_roll_1) for 16 liquid "
               "commodity roots, 2010+. Carry = log(close_1)-log(close_2); within-contract front "
               "returns with roll jumps removed. Generalization complexes (FX/rates/equity futures) "
               "loaded by the same adapter."),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={},
    grid=GRID,
    scope='broad',
    generalization_universes=list(GEN_ROOTS.keys()),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=12,
    expectations=[
        {"name": "distinct_from_level",
         "claim": "carry-change book daily-return |corr| with static-carry-level book < 0.6 (distinct object)",
         "check": _chk_distinct_from_level},
        {"name": "hysteresis_cuts_turnover",
         "claim": "membership hysteresis lengthens mean hold_days vs the no-hysteresis variant",
         "check": _chk_hysteresis_cuts_turnover},
    ],
)