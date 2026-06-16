# =============================================================================
# Storage-confirmed commodity CURVE-CARRY with a CONTINUOUS inventory-deviation
# size tilt (storable complexes only).  scope='broad'.   *** CORRECTED ***
#
# Fixes vs the prior build, to match the FROZEN signal construction:
#  (1) CARRY = annualized log(close_1/close_2) from fut_curve(root,n_contracts=2)
#      -- the WITHIN-CONTRACT curve slope (never diffed across a roll), NOT a
#      trailing realized drift_fut-drift_spot proxy.  Falls back to the realized
#      roll yield ONLY if fut_curve is genuinely unprovisioned.
#  (2) UNIVERSE restored to the proposal: energy {CL,NG,HO,RB} + grains
#      {ZC,ZS,ZW,ZL,ZM} (no Brent substitution, HO/RB kept).
#  (3) STORAGE tilt sourced from eia_series (crude/products/natgas stocks) +
#      usda_nass (grain stocks) across roots, deseasonalised+z-scored, PIT,
#      with a FRED crude-mirror / identity fallback only where an adapter is
#      truly missing.
#  (4) light hysteresis deadband on the carry z (turnover control) per spec.
# The tilt NEVER gates (m in [0.5,1.5] > 0): long/short SELECTION is carry-only,
# so the full trade count is preserved.
# =============================================================================
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, inv_vol_position
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights

# Prefer the faithful adapters; degrade gracefully if a given one is unprovisioned.
try:
    from sdk.adapters import fut_curve
except Exception:
    fut_curve = None
try:
    from sdk.adapters import eia_series
except Exception:
    eia_series = None
try:
    from sdk.adapters import usda_nass
except Exception:
    usda_nass = None

START         = "2007-01-01"
HOLDOUT_START = "2022-01-01"
SLOPE_SMOOTH  = 5         # light smoothing (1wk) of the curve-slope carry
ROLLYLD_LB    = 63        # window for the realized-roll-yield FALLBACK only
VOL_LB        = 63
TARGET_VOL    = 0.10
N_SIDE        = 3         # long top-3 / short bottom-3 (rotation -> run-length trades)
DEFAULT_K     = 0.5       # single pre-registered storage-tilt constant
HYST_BAND     = 0.25      # |carry z| deadband (turnover control / hysteresis)
COST_BPS      = 8.0

# root -> meta: fut=yf front symbol, spot=FRED spot (fallback carry only),
# eia=EIA stock series id, usda=(commodity,statisticcat) for USDA grain stocks, cx=complex
UNIVERSES = {
    # SEARCH universe (the proposal's primary): storable ENERGY + GRAINS
    "search": {
        "CL": dict(fut="CL=F", spot="DCOILWTICO", eia="PET.WCESTUS1.W",            cx="energy"),
        "NG": dict(fut="NG=F", spot="DHHNGSP",    eia="NG.NW2_EPG0_SWO_R48_BCF.W", cx="energy"),
        "HO": dict(fut="HO=F", spot="DDFUELNYH",  eia="PET.WDISTUS1.W",            cx="energy"),
        "RB": dict(fut="RB=F", spot=None,         eia="PET.WGTSTUS1.W",            cx="energy"),
        "ZC": dict(fut="ZC=F", spot="PMAIZMTUSDM", usda=("CORN","GRAIN STOCKS"),     cx="grains"),
        "ZS": dict(fut="ZS=F", spot="PSOYBUSDM",   usda=("SOYBEANS","GRAIN STOCKS"), cx="grains"),
        "ZW": dict(fut="ZW=F", spot="PWHEAMTUSDM", usda=("WHEAT","GRAIN STOCKS"),    cx="grains"),
        "ZL": dict(fut="ZL=F", spot="PSOILUSDM",                                    cx="grains"),
        "ZM": dict(fut="ZM=F", spot="PSMEAUSDM",                                    cx="grains"),
    },
    # GEN universes (DISJOINT from search; share NO tickers) -- all STORABLE, so the
    # convenience-yield mechanism MUST generalise (>=60% OOS-positive on holdout).
    # No public stock series mapped here -> tilt is identity (tests BASE carry universality).
    "softs": {
        "KC": dict(fut="KC=F", spot="PCOFFOTMUSDM", cx="softs"),
        "SB": dict(fut="SB=F", spot="PSUGAISAUSDM", cx="softs"),
        "CC": dict(fut="CC=F", spot="PCOCOUSDM",    cx="softs"),
        "CT": dict(fut="CT=F", spot="PCOTTINDUSDM", cx="softs"),
    },
    "metals": {
        "GC":  dict(fut="GC=F",  spot="GOLDAMGBD228NLBM", cx="metals"),
        "HG":  dict(fut="HG=F",  spot="PCOPPUSDM",        cx="metals"),
        "ALI": dict(fut="ALI=F", spot="PALUMUSDM",        cx="metals"),
    },
    "grains_ext": {  # other grains, disjoint from the search book
        "ZR": dict(fut="ZR=F", spot="PRICENPQUSDM", cx="grains_ext"),
        "ZO": dict(fut="ZO=F", spot=None,           cx="grains_ext"),
    },
}
GEN_UNIVERSES = ["softs", "metals", "grains_ext"]
ROOT2CX = {r: m["cx"] for tbl in UNIVERSES.values() for r, m in tbl.items()}


# ----------------------------------------------------------------------------- helpers
def _naive(s):
    """tz-strip + normalise to midnight so yfinance aligns with FRED/EIA dates."""
    s = s.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s


def _safe_fred(fid):
    if not fid:
        return None
    try:
        s = fred_series({fid: "v"}, start=START)["v"].dropna()
        return _naive(s) if not s.empty else None
    except Exception:
        return None


def _deseason_z(inv, lag_days=7):
    """Deseasonalised inventory z-score, fully PIT.  z>0 = glut, z<0 = tight.
      * +lag_days availability lag (reports publish after the reference period),
      * subtract the TRAILING 5y same-week-of-year mean (prior years only -> no lookahead),
      * scale by the trailing same-week std."""
    s = _naive(inv).dropna().sort_index()
    if s.empty:
        return s
    s.index = s.index + pd.Timedelta(days=int(lag_days))
    df = pd.DataFrame({"v": s.values}, index=s.index)
    df["woy"] = pd.DatetimeIndex(df.index).isocalendar().week.astype(int).values
    g = df.groupby("woy")["v"]
    mean = g.transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    std  = g.transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    z = (df["v"] - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _root_carry(root, fcol, sp):
    """FROZEN carry: annualized log(close_1/close_2) from the 2-contract curve
    (WITHIN a contract, never across a roll).  Fallback (only if fut_curve is
    genuinely unprovisioned): trailing realized roll yield drift_fut - drift_spot."""
    if fut_curve is not None:
        try:
            cv = fut_curve(root, n_contracts=2)
            if cv is not None and len(cv) > 0:
                dfc = pd.DataFrame(cv).astype(float)
                if dfc.shape[1] >= 2:
                    c1, c2 = dfc.iloc[:, 0], dfc.iloc[:, 1]
                    slope = np.log(c1 / c2).replace([np.inf, -np.inf], np.nan).dropna()
                    if not slope.empty:
                        # ~monthly contract spacing -> annualise (x12); sign/rank invariant.
                        return _naive(slope) * 12.0
        except Exception:
            pass
    # ---- fallback realized roll yield (needs a spot reference) ----
    if sp is None or sp.empty:
        return None
    f = _naive(fcol); s = _naive(sp)
    u = f.index.union(s.index)
    f = f.reindex(u).ffill(); s = s.reindex(u).ffill()
    rj = (np.log(f).diff() - np.log(s).diff())
    return (rj.rolling(ROLLYLD_LB, min_periods=max(20, ROLLYLD_LB // 2)).mean() * 252.0).dropna()


def _root_inventory(meta):
    """Deseasonalised inventory z from EIA (energy) or USDA (grains); FRED crude
    mirror / identity fallback where an adapter is missing."""
    eid = meta.get("eia")
    if eid and eia_series is not None:
        try:
            s = pd.Series(eia_series(eid)).dropna()
            if not s.empty:
                return _deseason_z(s, lag_days=7)
        except Exception:
            pass
    ud = meta.get("usda")
    if ud and usda_nass is not None:
        try:
            s = pd.Series(usda_nass(commodity=ud[0], statisticcat=ud[1])).dropna()
            if not s.empty:
                return _deseason_z(s, lag_days=21)   # USDA stocks reports publish weeks later
        except Exception:
            pass
    if str(meta.get("eia", "")).startswith("PET.WCESTUS1"):   # crude FRED mirror fallback
        iv = _safe_fred("WCESTUS1")
        if iv is not None and not iv.empty:
            return _deseason_z(iv, lag_days=7)
    return None


def _build_panel(table):
    """(field, root) panel: fut=front future (PnL/vol), carry=annualised curve slope,
    inv=deseasonalised inventory z (0 where none).  Robust to any single feed failing."""
    fut_syms = [m["fut"] for m in table.values()]
    try:
        fut_px = _naive(yf_panel(fut_syms, start=START))
    except Exception:
        return pd.DataFrame()

    futs, carries, invs, smap = {}, {}, {}, {}
    for root, meta in table.items():
        fsym = meta["fut"]
        if fsym not in getattr(fut_px, "columns", []):
            continue
        fcol = fut_px[fsym].dropna()
        if fcol.empty:
            continue
        sp = _safe_fred(meta.get("spot")) if meta.get("spot") else None
        carry = _root_carry(root, fcol, sp)
        if carry is None or carry.empty:
            continue
        futs[root], carries[root], smap[root] = fcol, carry, meta["cx"]
        invz = _root_inventory(meta)
        if invz is not None and not invz.empty:
            invs[root] = invz

    if len(futs) < 2:
        return pd.DataFrame()

    fut_df = pd.DataFrame(futs).sort_index()
    idx = fut_df.index
    carry_df, inv_df = pd.DataFrame(index=idx), pd.DataFrame(index=idx)
    for r in futs:
        u = idx.union(carries[r].index)
        carry_df[r] = carries[r].reindex(u).ffill().reindex(idx)
        if r in invs:
            u2 = idx.union(invs[r].index)
            inv_df[r] = invs[r].reindex(u2).ffill().reindex(idx).fillna(0.0)
        else:
            inv_df[r] = 0.0

    panel = pd.concat({"fut": fut_df, "carry": carry_df, "inv": inv_df}, axis=1)
    panel.attrs["sector_map"] = smap
    return panel


def _cross_ls(cz, n_side, band=HYST_BAND):
    """Per-date LONG top-n_side / SHORT bottom-n_side of the carry z (middle flat),
    with a |z| deadband for turnover control (names inside the band are skipped)."""
    sig = pd.DataFrame(0.0, index=cz.index, columns=cz.columns)
    for dt, row in cz.iterrows():
        row = row.dropna()
        if len(row) < 2:
            continue
        ns = min(n_side, len(row) // 2)
        if ns < 1:
            continue
        longs  = row.nlargest(ns)
        shorts = row.nsmallest(ns)
        longs  = longs[longs > band]
        shorts = shorts[shorts < -band]
        if len(longs):
            sig.loc[dt, longs.index] = 1.0
        if len(shorts):
            sig.loc[dt, shorts.index] = -1.0
    return sig


def _dollar_neutral(pos):
    """Row-wise (no-lookahead) rescale of the short leg so net exposure ~= 0 each day."""
    pos = pos.fillna(0.0)
    longs, shorts = pos.clip(lower=0.0), pos.clip(upper=0.0)
    scale = (longs.sum(axis=1) / (-shorts).sum(axis=1).replace(0.0, np.nan)).fillna(1.0)
    return longs + shorts.mul(scale, axis=0)


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    return _build_panel(UNIVERSES["search"])


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(UNIVERSES.get(label, {}))


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    name = "curve_carry_storage"
    k      = float(params.get("k", DEFAULT_K))
    n_side = int(params.get("n_side", N_SIDE))
    band   = float(params.get("band", HYST_BAND))

    lv = panel.columns.get_level_values(0) if (panel is not None and hasattr(panel, "columns")) else []
    if panel is None or len(panel) == 0 or "carry" not in lv or "fut" not in lv:
        return pd.Series(dtype=float, name=name), []

    fut   = panel["fut"].astype(float)
    carry = panel["carry"].astype(float)
    carry = carry.rolling(SLOPE_SMOOTH, min_periods=1).mean()      # light smoothing of the slope
    invz  = (panel["inv"].astype(float).fillna(0.0)
             if "inv" in panel.columns.get_level_values(0) else carry * 0.0)
    if fut.shape[1] < 2:
        return pd.Series(dtype=float, name=name), []

    rets = fut.pct_change()

    cz  = xs_zscore(carry)                       # cross-sectional, winsorized, NaN-preserving
    sig = _cross_ls(cz, n_side, band)            # long/short SELECTION from carry ALONE

    # CONTINUOUS storage confirmation multiplier (never gates: m in [0.5,1.5]).
    # m>1 when inventory CONFIRMS carry sign (tight+backwardation, or glut+contango).
    mult = (1.0 + k * (-invz * np.sign(carry))).clip(0.5, 1.5).fillna(1.0)
    sig_tilt = sig * mult                         # rescales magnitude; zeros & signs unchanged

    # inverse-vol + ~10% vol-target + weekly hold + 1-day execution lag handled by the kit.
    pos = inv_vol_position(sig_tilt, rets, target_vol=TARGET_VOL, vol_lb=VOL_LB,
                           max_pos=2 * n_side, rebalance="W")
    pos = _dollar_neutral(pos)

    smap = panel.attrs.get("sector_map") or {c: ROOT2CX.get(c, "commodity") for c in fut.columns}
    daily = net_of_cost(pos, rets, cost_bps=COST_BPS, name=name)
    trades = trades_from_weights(pos, rets, smap)
    return daily, trades


# ----------------------------------------------------------------------------- soft expectations
def _sharpe(r):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _chk_trade_count(ctx):
    """Core claim: the tilt only SIZES, never gates -> trade count identical to k=0."""
    hs = str(ctx.get("holdout_start", HOLDOUT_START))
    try:
        _, tr0 = signal(ctx["panel"], k=0.0)
        n0 = len([t for t in tr0 if str(t.get("entry_date", "")) < hs])
        n1 = len(ctx.get("trades", []))
        return {"pass": n0 == n1, "observed": f"tilt={n1} no_tilt={n0}"}
    except Exception as e:
        return {"pass": False, "observed": f"err:{e}"}


def _chk_tilt_active(ctx):
    """Storage tilt must change the book vs no-tilt (else inventory never fed through)."""
    nt, base = ctx.get("grid", {}).get("no_tilt"), ctx.get("search")
    if nt is None or base is None:
        return {"pass": False, "observed": "missing series"}
    a, b = base.align(nt, join="inner")
    L1 = float((a - b).abs().sum())
    return {"pass": L1 > 1e-9, "observed": f"L1diff={L1:.6f}"}


def _chk_no_dilution(ctx):
    """Confirmation refinement should not dilute base carry Sharpe (>= no_tilt - 0.10)."""
    nt, base = ctx.get("grid", {}).get("no_tilt"), ctx.get("search")
    if nt is None or base is None:
        return {"pass": False, "observed": "missing series"}
    st, sn = _sharpe(base), _sharpe(nt)
    return {"pass": st >= sn - 0.10, "observed": f"sharpe_tilt={st:.2f} sharpe_notilt={sn:.2f}"}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="commod_curvecarry_storage_tilt_v1",
    family="commodity_carry",
    title="Storage-confirmed commodity curve-carry (continuous inventory-deviation size tilt)",
    markets=["commodities"],
    data_desc=("Front-month commodity futures (yf_panel '=F') for PnL/vol; carry = annualised "
               "log(close_1/close_2) from fut_curve(root,n_contracts=2) (within-contract curve "
               "slope, never across a roll); storage tilt from eia_series (crude WCESTUS1 / distillate "
               "WDISTUS1 / gasoline WGTSTUS1 / nat-gas weekly stocks) + usda_nass (corn/soy/wheat "
               "grain stocks), deseasonalised+z-scored, PIT; FRED crude mirror / identity fallback "
               "only where an adapter is unprovisioned."),
    pre_registration=(
        "H: convenience-yield/storage premium -> backwardated storables out-earn contangoed ones; "
        "fundamental inventory CONFIRMS the carry sign and should add when it agrees. "
        "Construction (frozen): carry per root = annualised log(F1/F2) from the 2-contract curve "
        "(within-contract, never across a roll), cross-sectional z with a |z|>0.25 deadband, long "
        "top-3 / short bottom-3, inverse-vol + ~10% vol-target + weekly hold + 1-day lag via "
        "inv_vol_position, ~dollar-neutral, 8bps costs. Storage tilt m=clip(1 + k*(-z*sign(carry)),"
        "0.5,1.5), k=0.5, deseasonalised (trailing-5y same-week) z, availability lag (EIA +7d / USDA "
        "+21d), ffilled. The tilt CONTINUOUSLY SIZES and NEVER gates/triggers -> selection is "
        "carry-only -> full trade count preserved (explicit reframe of the 0/13 gated/triggered "
        "substrate). Universe = storable ENERGY {CL,NG,HO,RB} + GRAINS {ZC,ZS,ZW,ZL,ZM}. scope=broad: "
        "mechanism is universal across STORABLE complexes, so it must generalise OOS to softs / metals "
        "/ extended grains (>=60%). On gen universes no public stock series is mapped, so the tilt is "
        "identity there (tests BASE carry universality); the storage marginal is tested on the search "
        "book. Falsifiers: gen battery <60% OOS-positive; tilt inactive (L1=0); tilt dilutes base "
        "Sharpe; MCPT. Trend overlay deferred until standalone survives (no reflexive 50/50)."),
    load_data=load_data,
    signal=signal,
    default_params={"k": DEFAULT_K, "n_side": N_SIDE, "band": HYST_BAND},
    grid={
        "default":  {},                 # primary
        "no_tilt":  {"k": 0.0},         # base carry, for the soft-expectation comparisons
        "k_strong": {"k": 1.0},
        "n_wide":   {"n_side": 2},
    },
    scope="broad",
    generalization_universes=GEN_UNIVERSES,
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=6,
    expectations=[
        {"name": "trade_count_preserved",
         "claim": "continuous tilt never gates -> trade count equals the k=0 (no-tilt) book",
         "check": _chk_trade_count},
        {"name": "tilt_active",
         "claim": "storage tilt changes the book vs no-tilt (inventory data fed through)",
         "check": _chk_tilt_active},
        {"name": "tilt_no_dilution",
         "claim": "tilt Sharpe >= no-tilt Sharpe - 0.10 (confirmation does not dilute base carry)",
         "check": _chk_no_dilution},
    ],
)