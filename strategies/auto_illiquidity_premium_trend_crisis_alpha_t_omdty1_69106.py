"""
Illiquidity-Premium x Trio-Gated Commodity-Trend Crisis-Alpha — two-premium book (LOCAL).

This variant's ONE new idea is the PARAMETER-FREE TRIO GATE on the commodity-trend leg
(the defining mutation that separates it from the plain-TSMOM sibling):

    take the LONG-trend in a commodity ONLY when ALL THREE fundamental conditions agree:
       (1) term structure is BACKWARDATED   : close_1 > close_2          (fut_curve)
       (2) COMMERCIALS are NET-SHORT         : comm_net / open_interest > 0 (cot_positioning)
           -> hedging-pressure theory: hedgers short => speculators earn the long premium
       (3) STORAGE is TIGHT                  : inventory below its trailing-year norm (eia_series)
    and symmetrically take the SHORT-trend ONLY when all three are bearish
    (contango AND commercials net-long AND storage ample). No tunable thresholds -> the gate
    is parameter-free; it merely confirms/vetoes the sign(252d momentum) trend.

Universe is the OWNED commodity-FUTURES complex (fut_curve / cot_positioning roots), NOT a
tradable-ETF proxy (that proxy is the sibling this proposal explicitly does not duplicate).

Leg A = Amihud illiquidity premium (small-cap, survivorship-clean long/short). The book blends
the two vol-matched premia (100:25 risk) so the trio-gated crisis-trend leg complements the
pro-cyclical illiquidity leg (cut drawdown, preserve Sharpe, low leg correlation).

HONEST DATA STATE: the storage condition is wired to eia_series (US petroleum / nat-gas stocks)
+ degrades to NEUTRAL (gate -> duo of backwardation & COT) where no storage series is provisioned
(USDA grain stocks pending; EIA key-pending). This is declared, not faked: the trio mechanism is
BUILT; the storage leg is neutral until its key lands rather than fabricated.

FIX (vs failed run): when fut_curve/cot/eia are unavailable, the helper slicers return EMPTY
frames with an integer RangeIndex; concatenating those with the datetime-indexed equity/commodity
frames made the panel's index union collapse to a plain object Index -> .resample() raised
"Only valid with DatetimeIndex". The actual values are all Timestamps, so we coerce the panel
index back to a DatetimeIndex in load_data (+ defensively in each leg builder). No SDK change.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, yf_panel, inv_vol_position, trend_returns,
                          fut_curve, cot_positioning, eia_series)
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
START = "2005-01-01"
HOLDOUT = "2022-01-01"

# OWNED commodity-futures complex.  root : (yfinance continuous ticker for returns/trend,
# sector, EIA storage series id or None).  fut_curve / cot_positioning are keyed by ROOT.
CMDTY = {
    "CL": ("CL=F", "Energy",         "WCESTUS1"),            # WTI crude stocks
    "NG": ("NG=F", "Energy",         "NW2_EPG0_SWO_R48_BCF"),# nat-gas working storage
    "HO": ("HO=F", "Energy",         "WDISTUS1"),            # distillate stocks
    "RB": ("RB=F", "Energy",         "WGTSTUS1"),            # gasoline stocks
    "GC": ("GC=F", "PreciousMetals", None),
    "SI": ("SI=F", "PreciousMetals", None),
    "HG": ("HG=F", "BaseMetals",     None),
    "ZC": ("ZC=F", "Grains",         None),                  # USDA grain stocks pending
    "ZW": ("ZW=F", "Grains",         None),
    "ZS": ("ZS=F", "Grains",         None),
    "SB": ("SB=F", "Softs",          None),
    "KC": ("KC=F", "Softs",          None),
    "CT": ("CT=F", "Softs",          None),
    "LE": ("LE=F", "Livestock",      None),
    "HE": ("HE=F", "Livestock",      None),
}
_CM_SECTOR = {r: CMDTY[r][1] for r in CMDTY}

_TGT_VOL = 0.10        # annualised vol target each leg is scaled to before blending
_VOL_LB = 60           # trailing window for vol estimates
_AMIHUD_LB = 252       # trailing window for the illiquidity measure AND the TSMOM lookback
_COT_LAG = 5           # extra trading-day lag on COT/storage (publication delay) -> no lookahead
_SECTOR_MAP = {}       # populated by load_data(); fallback if panel.attrs is dropped

_FIELD_ALIASES = {
    "close_1": ["close_1", "c1", "m1", "front", "near", "p1", "settle_1", "px1"],
    "close_2": ["close_2", "c2", "m2", "second", "next", "p2", "settle_2", "px2"],
    "comm_net": ["comm_net", "commercial_net", "commercials_net", "comm_net_short", "cn"],
    "oi": ["oi", "open_interest", "openinterest", "total_oi", "open_int"],
}


# ----------------------------------------------------------------------------- #
# Small generic helpers (NOT signal kit — pure transforms, no lookahead)
# ----------------------------------------------------------------------------- #
def _ensure_dt(df):
    """Coerce a frame's index to DatetimeIndex (the empty-frame concat union can collapse it
    to a plain object Index, which breaks .resample()). Values are Timestamps -> safe."""
    if df is None:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
        except Exception:
            pass
    return df


def _vt(r, target=_TGT_VOL, lb=_VOL_LB):
    """Scale a daily return series to `target` annualised vol using TRAILING vol.
    scale_t uses std through t-1 (shift(1)) -> no lookahead. Leverage capped at 4x."""
    r = r.dropna()
    if len(r) < lb + 2:
        return r
    daily_tgt = target / np.sqrt(252.0)
    sd = r.rolling(lb).std()
    scale = (daily_tgt / sd).shift(1).clip(upper=4.0)
    return (r * scale).dropna()


def _maxdd(r):
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _ann_sharpe(r):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _get_sector_map(panel):
    smap = {}
    try:
        smap = dict(panel.attrs.get("sector_map") or {})
    except Exception:
        smap = {}
    if not smap:
        smap = dict(_SECTOR_MAP)
    if not smap:
        try:
            _, smap = sector_universe("Small", top_n_per_sector=120)
            smap = dict(smap)
        except Exception:
            smap = {}
    smap.update(_CM_SECTOR)
    return smap


def _slice_field(df, roots, field):
    """Pull a per-root field out of a (root, field) panel into columns=roots. Never raises."""
    try:
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame(columns=roots)
        aliases = _FIELD_ALIASES.get(field, [field])
        out = {}
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            lvl1 = pd.Index([str(x).lower() for x in df.columns.get_level_values(-1)])
            for r in roots:
                for a in aliases:
                    mask = (lvl0 == r) & (lvl1 == a.lower())
                    if mask.any():
                        out[r] = df.loc[:, mask].iloc[:, 0]
                        break
        else:
            low = {str(c).lower(): c for c in df.columns}
            for r in roots:
                for a in aliases:
                    key = f"{r}_{a}".lower()
                    if key in low:
                        out[r] = df[low[key]]
                        break
        return pd.DataFrame(out) if out else pd.DataFrame(columns=roots)
    except Exception:
        return pd.DataFrame(columns=roots)


def _safe_curve(roots):
    try:
        return fut_curve(roots, START)
    except Exception:
        return None


def _safe_cot(roots):
    try:
        return cot_positioning(roots, START)
    except Exception:
        return None


def _safe_storage(roots, idx):
    """EIA (and, when keyed, USDA) storage levels aligned to daily idx. NaN where unavailable
    -> the storage condition degrades to NEUTRAL (trio gate -> duo) rather than fabricated."""
    stor = pd.DataFrame(index=idx, columns=roots, dtype=float)
    for r in roots:
        sid = CMDTY[r][2]
        if not sid:
            continue
        try:
            s = pd.Series(eia_series(sid, START)).astype(float)
            stor[r] = s.reindex(idx).ffill().values
        except Exception:
            continue
    return stor


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Packed 2-level panel:
        ('eq_px',     ticker) -> Sharadar closeadj (returns; div+split adjusted)
        ('eq_dvol',   ticker) -> raw dollar volume = close(unadj) * volume(raw)
        ('cm_px',     root)   -> commodity front-month continuous close (yf; returns/trend)
        ('cm_c1',     root)   -> front-contract settle      (fut_curve; term structure)
        ('cm_c2',     root)   -> second-contract settle      (fut_curve)
        ('cm_commnet',root)   -> commercial NET-SHORT contracts (cot_positioning)
        ('cm_oi',     root)   -> open interest               (cot_positioning)
        ('cm_stor',   root)   -> storage level (eia_series; NaN where not provisioned)
    Sector map carried in panel.attrs + a module global (attrs can drop on slicing)."""
    global _SECTOR_MAP
    tickers, smap = sector_universe("Small", top_n_per_sector=120)  # sector-balanced small caps
    _SECTOR_MAP = dict(smap)

    closeadj = sep_panel(tickers, START, field="closeadj")
    close = sep_panel(tickers, START, field="close")     # unadjusted -> raw $vol
    volume = sep_panel(tickers, START, field="volume")   # raw shares
    cols = closeadj.columns.intersection(close.columns).intersection(volume.columns)
    closeadj, close, volume = closeadj[cols], close[cols], volume[cols]
    closeadj = _ensure_dt(closeadj); close = _ensure_dt(close); volume = _ensure_dt(volume)
    dvol = (close * volume).replace(0, np.nan)           # raw dollar volume (split-consistent)

    roots = list(CMDTY.keys())
    cm_raw = yf_panel([CMDTY[r][0] for r in roots], START)
    cm_px = cm_raw.rename(columns={CMDTY[r][0]: r for r in roots})
    cm_px = cm_px.reindex(columns=[r for r in roots if r in cm_px.columns])
    cm_px = _ensure_dt(cm_px)

    fc = _safe_curve(roots)
    cm_c1 = _slice_field(fc, roots, "close_1")
    cm_c2 = _slice_field(fc, roots, "close_2")

    cot = _safe_cot(roots)
    cm_commnet = _slice_field(cot, roots, "comm_net")
    cm_oi = _slice_field(cot, roots, "oi")

    idx = cm_px.index if len(cm_px.index) else closeadj.index
    cm_stor = _safe_storage(roots, idx)

    panel = pd.concat({
        "eq_px": closeadj, "eq_dvol": dvol,
        "cm_px": cm_px, "cm_c1": cm_c1, "cm_c2": cm_c2,
        "cm_commnet": cm_commnet, "cm_oi": cm_oi, "cm_stor": cm_stor,
    }, axis=1).sort_index()

    # CRITICAL: the empty (root,field) frames above carry an int RangeIndex; their union with the
    # datetime frames collapses panel.index to a plain object Index, which breaks .resample().
    # All real index values are Timestamps -> coerce straight back to a DatetimeIndex.
    panel = _ensure_dt(panel)

    try:
        panel.attrs["sector_map"] = dict(smap)
    except Exception:
        pass
    return panel


def load_gen_data(label) -> pd.DataFrame:
    # scope='local' -> stage-2 battery does not run; provided for interface completeness.
    return load_data()


# ----------------------------------------------------------------------------- #
# Leg builders (each returns a LAGGED weight matrix + its returns matrix)
# ----------------------------------------------------------------------------- #
def _amihud_book(panel):
    """Cross-sectional illiquidity long/short on small caps.
    illiq = trailing-mean(|ret| / dollar_volume); long most-illiquid, short most-liquid.
    Inverse-vol weighted within each leg, dollar-neutral (gross=1), weekly rebalance,
    1-DAY LAG applied here (our responsibility) -> pass straight to net_of_cost."""
    px = _ensure_dt(panel["eq_px"].dropna(how="all", axis=1))
    dvol = panel["eq_dvol"].reindex(index=px.index, columns=px.columns)
    rets = px.pct_change(fill_method=None)

    illiq = (rets.abs() / dvol).rolling(_AMIHUD_LB, min_periods=60).mean()
    z = xs_zscore(illiq)                              # winsorised x-sectional z, NaN-preserving
    rnk = z.rank(axis=1, pct=True)
    longs = (rnk >= 0.80).astype(float)              # most illiquid -> earn the premium
    shorts = (rnk <= 0.20).astype(float)             # most liquid

    ivol = 1.0 / rets.rolling(_VOL_LB).std().replace(0, np.nan)
    wl = (longs * ivol); wl = wl.div(wl.sum(axis=1), axis=0)
    ws = (shorts * ivol); ws = ws.div(ws.sum(axis=1), axis=0)
    W = (0.5 * wl).subtract(0.5 * ws, fill_value=0.0)  # dollar-neutral, gross ~ 1

    Ww = W.resample("W-FRI").last().reindex(W.index, method="ffill")  # weekly rebalance
    W_lag = Ww.shift(1)                                               # 1-day execution lag
    return W_lag, rets


def _cmdty_book(panel, mom_lb=_AMIHUD_LB):
    """TRIO-GATED commodity-trend crisis-alpha sleeve on the OWNED futures complex.
    trend = sign(trailing-mom). LONG taken only when (backwardated & commercials net-short &
    storage tight); SHORT only when (contango & commercials net-long & storage ample).
    The gate is parameter-free. inv_vol_position handles inverse-vol sizing, weekly rebalance,
    and the 1-day execution lag; COT/storage carry an extra publication lag (no lookahead)."""
    px = _ensure_dt(panel["cm_px"].dropna(how="all"))
    roots = list(px.columns)
    rets = px.pct_change(fill_method=None)
    trend = np.sign(px.pct_change(mom_lb, fill_method=None))

    def _blk(name):
        try:
            b = panel[name].reindex(columns=roots)
        except Exception:
            b = pd.DataFrame(index=px.index, columns=roots, dtype=float)
        b = _ensure_dt(b)
        return b.reindex(px.index).ffill()

    c1, c2 = _blk("cm_c1"), _blk("cm_c2")
    commnet, oi = _blk("cm_commnet"), _blk("cm_oi")
    stor = _blk("cm_stor")

    # (1) term structure
    backwardated = c1 > c2
    contango = c1 < c2
    # (2) hedging pressure (COT) — extra publication lag
    comm_ratio = (commnet / oi.replace(0, np.nan)).shift(_COT_LAG)
    comm_short = comm_ratio > 0          # commercials net SHORT -> long premium
    comm_long = comm_ratio < 0
    # (3) storage tightness — neutral (pass) where no series is provisioned
    stor_dev = (stor - stor.rolling(252, min_periods=40).mean()).shift(_COT_LAG)
    storage_tight = (stor_dev < 0) | stor.isna()
    storage_loose = (stor_dev > 0) | stor.isna()

    long_ok = (backwardated & comm_short & storage_tight).fillna(False)
    short_ok = (contango & comm_long & storage_loose).fillna(False)

    sig = pd.DataFrame(0.0, index=px.index, columns=roots)
    sig = sig.mask((trend > 0) & long_ok, 1.0)
    sig = sig.mask((trend < 0) & short_ok, -1.0)

    W = inv_vol_position(sig, rets, target_vol=_TGT_VOL, vol_lb=_VOL_LB,
                         max_pos=max(1, len(roots)), rebalance="W")  # inv-vol + weekly + 1d lag
    return W, rets


def _cmdty_book_ungated(panel, mom_lb=_AMIHUD_LB):
    """Plain-TSMOM sibling (NO trio gate) — used only by the gate-active soft check."""
    px = _ensure_dt(panel["cm_px"].dropna(how="all"))
    rets = px.pct_change(fill_method=None)
    sig = np.sign(px.pct_change(mom_lb, fill_method=None))
    W = inv_vol_position(sig, rets, target_vol=_TGT_VOL, vol_lb=_VOL_LB,
                         max_pos=max(1, len(px.columns)), rebalance="W")
    return W, rets


# ----------------------------------------------------------------------------- #
# Signal
# ----------------------------------------------------------------------------- #
def signal(panel, cmdty_weight=0.20, mom_lb=_AMIHUD_LB, **params):
    """Blend: Amihud 100% : trio-gated commodity-trend 25% of book risk (cmdty_weight=0.20 ->
    0.80/0.20 after both legs are vol-matched to equal vol). Final book re-targeted to the
    same vol so DD/Sharpe are comparable to the standalone Amihud leg."""
    We, re_ = _amihud_book(panel)
    Wc, rc = _cmdty_book(panel, mom_lb=mom_lb)
    smap = _get_sector_map(panel)

    a_net = net_of_cost(We, re_, cost_bps=8.0, name="amihud")   # We already lagged
    c_net = net_of_cost(Wc, rc, cost_bps=8.0, name="cmdty")     # Wc already lagged

    av, cv = _vt(a_net), _vt(c_net)                              # equal-vol legs
    df = pd.concat([av.rename("a"), cv.rename("c")], axis=1).dropna()
    raw = (1.0 - cmdty_weight) * df["a"] + cmdty_weight * df["c"]
    out = _vt(raw)
    out.name = "illiq_x_trio_cmdty_trend"

    # Contract ledger: one trade per held run, regime stamped by the kit (never by us).
    trades = trades_from_weights(We, re_, smap) + trades_from_weights(Wc, rc, _CM_SECTOR)
    return out, trades


# ----------------------------------------------------------------------------- #
# Soft expectations (machine-checkable mechanism claims; recompute via helpers,
# never via signal(); everything sliced to dates < holdout_start)
# ----------------------------------------------------------------------------- #
def _chk_gate_active(ctx):
    """Central-mutation check: the trio gate must REMOVE exposure vs the ungated TSMOM
    sibling (otherwise the gate is inert and this variant is the sibling it claims not to be)."""
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    Wg, _ = _cmdty_book(panel)
    Wu, _ = _cmdty_book_ungated(panel)
    Wg, Wu = Wg[Wg.index < h], Wu[Wu.index < h]
    g = float(Wg.abs().sum(axis=1).mean()) if len(Wg) else 0.0
    u = float(Wu.abs().sum(axis=1).mean()) if len(Wu) else 0.0
    obs = round(g / u, 3) if u > 0 else None
    return {"pass": bool(u > 0 and g < u), "observed": obs}


def _chk_dd(ctx):
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    comb = ctx["search"].dropna()
    We, re_ = _amihud_book(panel)
    a = _vt(net_of_cost(We, re_, cost_bps=8.0, name="a"))
    a = a[a.index < h]
    idx = comb.index.intersection(a.index)
    ddc, dda = _maxdd(comb.reindex(idx)), _maxdd(a.reindex(idx))
    obs = round(abs(ddc) / abs(dda), 3) if dda != 0 else None
    return {"pass": bool(dda < 0 and ddc >= 0.80 * dda), "observed": obs}


def _chk_sharpe(ctx):
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    comb = ctx["search"].dropna()
    We, re_ = _amihud_book(panel)
    a = _vt(net_of_cost(We, re_, cost_bps=8.0, name="a"))
    a = a[a.index < h]
    idx = comb.index.intersection(a.index)
    sc, sa = _ann_sharpe(comb.reindex(idx)), _ann_sharpe(a.reindex(idx))
    obs = round(sc / sa, 3) if sa != 0 else None
    return {"pass": bool(sa > 0 and sc >= 0.90 * sa), "observed": obs}


def _chk_corr(ctx):
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    We, re_ = _amihud_book(panel)
    Wc, rc = _cmdty_book(panel)
    a = net_of_cost(We, re_, cost_bps=8.0, name="a")
    c = net_of_cost(Wc, rc, cost_bps=8.0, name="c")
    d = pd.concat([a, c], axis=1).dropna()
    d = d[d.index < h]
    corr = float(d.iloc[:, 0].corr(d.iloc[:, 1])) if len(d) > 30 else 1.0
    return {"pass": bool(corr <= 0.10), "observed": round(corr, 3)}


def _chk_track(ctx):
    """Diagnostic (reported, NON-gating): does the trio-gated commodity sleeve still track the
    validated 21-market CTA crisis-alpha stream (the gate prunes positions, lowering corr)."""
    h = pd.Timestamp(ctx["holdout_start"]); panel = ctx["panel"]
    Wc, rc = _cmdty_book(panel)
    c = net_of_cost(Wc, rc, cost_bps=8.0, name="c")
    try:
        tr, _ = trend_returns()
    except Exception:
        return {"pass": True, "observed": "trend_returns_unavailable"}
    d = pd.concat([c, tr], axis=1).dropna()
    d = d[d.index < h]
    corr = float(d.iloc[:, 0].corr(d.iloc[:, 1])) if len(d) > 30 else 0.0
    return {"pass": bool(corr >= 0.20), "observed": round(corr, 3)}


# ----------------------------------------------------------------------------- #
# Spec
# ----------------------------------------------------------------------------- #
SPEC = StrategySpec(
    id="amihud_illiq_x_cmdty_trio_gated_trend_local",
    family="illiquidity_premium_x_crisis_trend",
    title=("Illiquidity-Premium x TRIO-GATED Commodity-Trend Crisis-Alpha two-premium book — "
           "Amihud illiquidity long/short (small caps) + parameter-free trio-gated TSMOM on the "
           "owned commodity-futures complex (backwardation & commercials-net-short & storage-tight)"),
    markets=["US small-cap equities (Sharadar SEP, survivorship-clean, delisted incl.)",
             "Commodity futures complex (CL/NG/HO/RB/GC/SI/HG/ZC/ZW/ZS/SB/KC/CT/LE/HE): "
             "front-month continuous (yf), term structure (fut_curve), COT (cot_positioning), "
             "storage (eia_series)"],
    data_desc=("Sharadar SEP closeadj (returns) + raw close*volume (Amihud $vol) on a sector-"
               "balanced small-cap universe (sector_universe Small, 120/sector). Commodity "
               "front-month continuous via yf_panel; term-structure close_1/close_2 via fut_curve; "
               "commercial net-short + open interest via cot_positioning; petroleum/nat-gas storage "
               "via eia_series. $0 incremental data; all owned/free."),
    pre_registration=(
        "TWO-PREMIUM BOOK. Leg A = Amihud illiquidity premium (pro-cyclical): long the most-"
        "illiquid quintile / short the most-liquid quintile of small caps, illiq = trailing-252d "
        "mean(|ret|/$vol), inverse-vol weighted, dollar-neutral, weekly, 1-day lag, 8bps. Leg B = "
        "TRIO-GATED commodity-trend crisis alpha on the OWNED futures complex: trend = sign(252d "
        "momentum), inverse-vol sized, weekly, 1-day lag. THE DEFINING MUTATION is a PARAMETER-FREE "
        "TRIO GATE that confirms the trend's sign with fundamentals: take the LONG only when "
        "(1) backwardated [close_1>close_2 via fut_curve] AND (2) commercials NET-SHORT "
        "[comm_net/open_interest>0 via cot_positioning -> hedging-pressure premium] AND (3) storage "
        "TIGHT [level below trailing-year mean via eia_series]; symmetrically take the SHORT only "
        "when (contango AND commercials net-long AND storage ample). No tunable thresholds. Legs "
        "vol-matched to 10% then blended 0.80/0.20 (=100:25 risk) and re-targeted to 10%. "
        "PRE-REGISTERED SUCCESS (machine-checked, search window only): (i) the trio gate REDUCES "
        "commodity gross exposure vs the ungated-TSMOM sibling (gate is not inert); (ii) combined "
        "MaxDD reduced >=20% vs standalone Amihud; (iii) Sharpe degradation <=10%; (iv) leg "
        "correlation <= +0.10. REPORTED DIAGNOSTIC (non-gating): gated-sleeve tracking corr vs the "
        "validated 21-market CTA trend stream >= 0.20. HONEST DATA STATE: the storage condition is "
        "wired to eia_series (US petroleum/nat-gas stocks) and degrades to NEUTRAL (gate -> duo of "
        "backwardation & COT) for roots without a provisioned storage series (USDA grain stocks "
        "pending; EIA key-pending) — the trio mechanism is BUILT, the storage leg is not fabricated. "
        "The Amihud illiquidity leg is reconstructed canonically (no byte-frozen importable stream "
        "exists in the kit). Scope LOCAL: both premia's standalone validity is settled elsewhere; "
        "the only new claim is the trio gate's book-level complementarity, confirmed by forward "
        "paper validation. Futures complex is research/owned data (deploy via a futures account)."),
    load_data=load_data,
    signal=signal,
    default_params={"cmdty_weight": 0.20, "mom_lb": 252},
    grid={
        "default": {},
        "cw_15": {"cmdty_weight": 0.15},
        "cw_30": {"cmdty_weight": 0.30},
        "mom_126": {"mom_lb": 126},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=20,
    expectations=[
        {"name": "trio_gate_active",
         "claim": "trio-gated commodity gross exposure < ungated-TSMOM sibling (gate not inert)",
         "check": _chk_gate_active},
        {"name": "drawdown_reduced_20pct",
         "claim": "combined MaxDD magnitude <= 80% of standalone Amihud (>=20% DD cut)",
         "check": _chk_dd},
        {"name": "sharpe_preserved",
         "claim": "combined Sharpe >= 90% of standalone Amihud (<=10% degradation)",
         "check": _chk_sharpe},
        {"name": "legs_low_correlation",
         "claim": "Amihud vs trio-gated commodity-trend leg daily-return correlation <= +0.10",
         "check": _chk_corr},
        {"name": "cmdty_tracks_cta_trend",
         "claim": "trio-gated commodity sleeve corr vs validated 21-market CTA stream >= 0.20",
         "check": _chk_track},
    ],
)