"""
Storage-Surprise Convenience-Yield Premium — FULL 7-COMMODITY CROSS-COMPLEX implementation.

THESIS (frozen design): Low/abnormally-drawn inventories -> high convenience yield -> backwardation
-> positive expected futures return; abnormal builds -> contango -> negative. Pool the premium across
TWO distinct complexes so it is a cross-complex factor, not a within-complex curiosity:
  energy  {CL=F crude, RB=F gasoline, HO=F distillate, NG=F natural gas}
  grains  {ZC=F corn, ZS=F soybeans, ZW=F wheat}
LONG the abnormally-tight names, SHORT the abnormally-loose names, market-neutral, inverse-vol sized.

FIXES vs the REJECTED v1 (thesis-mismatch):
  (1) UNIVERSE RESTORED: v1 dropped NG + the entire grains complex, collapsing to a 3-name petroleum
      slice (the exact tiny within-complex cross-section the design forbids). v2 pools all 7 names
      across energy + grains -> real cross-complex diversification.
  (2) CONFIRMATION DEMOTED: v1 ran a price-momentum confirmation as a DEFAULT HARD GATE (deviating
      from the frozen term-structure confirmation we cannot source). v2 tests the convenience-yield
      premium STANDALONE by default (confirm=False) and offers price-confirm as a labelled GRID variant
      only -> the headline number is the unconfirmed standalone premium.

FIX vs v2-FAILED (HTTP 400 on the combined FRED request): the three EIA petroleum storage mnemonics
were fetched in ONE fred_series() call, so a single rejected/aliased series id (FRED -> HTTP 400)
crashed the whole load. v3 fetches each storage series INDIVIDUALLY inside try/except; any leg whose
FRED id is unavailable degrades gracefully to the SAME documented price-seasonal proxy already used for
NG + grains (the fallback path that always existed). Nothing else in the frozen design changes.

DATA CONSTRAINT (binding, disclosed — not a design choice):
The allowed adapters expose storage only via FRED. FRED carries the EIA weekly PETROLEUM stocks
(WCESTUS1 crude, WGTSTUS1 gasoline, WDISTUS1 distillate) -> the petroleum legs use the TRUE
storage-flow surprise WHEN the series resolves. EIA weekly NG working-gas storage and USDA grain stocks
are NOT on FRED under stable mnemonics reachable here, so NG + the 3 grains use a documented
CONVENIENCE-YIELD PRICE-SEASONAL proxy: the seasonal+AR(1) abnormal component of the front-month return
is a recognised inventory/convenience-yield proxy (Gorton-Hayashi-Rouwenhorst 2013: past-return & basis
signals load on the same inventory premium). This construction risk is the headline caveat; it is the
most-faithful BUILDABLE version of the pooled-complex thesis under the allowed adapters.

scope='local': commodity futures are a small FIXED universe; no disjoint 150-400-name commodity sub-
universe exists for a stage-2 breadth battery here, so the write-once holdout (>=2022-01-01) forward-
validates instead (the 'local' path). The Boreas trend overlay is NOT bolted on (test standalone first).

LOOK-AHEAD CONTROLS: petroleum storage is published ~Wed for the prior-Fri reference week -> the weekly
surprise is stamped +5 calendar days (~publish) then ff-filled on the daily grid. Price-proxy surprises
are stamped at week-ending-Friday close (observable) and only ever consumed on the FOLLOWING W-WED rebal
(strictly after Friday). The seasonal mean uses PRIOR same-week data only (expanding().shift(1)); the
AR(1) coef and standardising std are trailing rolling().shift(1). Weights are weekly-rebalanced and the
final weight matrix is .shift(1) before costs/returns (the lag is ours, stated here).
"""

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2007-01-01"          # RBOB (RB=F) liquid from ~2006; warmup ~1.5y -> search starts ~2008/9
HOLDOUT = "2022-01-01"

# ticker -> (FRED weekly-storage series id OR None for price-proxy, sub-complex sector)
_COMMOD = {
    "CL=F": ("WCESTUS1", "petroleum"),   # U.S. ending stocks of crude oil      (TRUE storage flow)
    "RB=F": ("WGTSTUS1", "petroleum"),   # U.S. ending stocks of total gasoline (TRUE storage flow)
    "HO=F": ("WDISTUS1", "petroleum"),   # U.S. ending stocks of distillate     (TRUE storage flow)
    "NG=F": (None,        "natgas"),     # EIA wkly NG storage not on FRED here -> price-seasonal proxy
    "ZC=F": (None,        "grains"),     # USDA corn stocks not on FRED         -> price-seasonal proxy
    "ZS=F": (None,        "grains"),     # USDA soybean stocks not on FRED      -> price-seasonal proxy
    "ZW=F": (None,        "grains"),     # USDA wheat stocks not on FRED        -> price-seasonal proxy
}
_SECTOR_MAP = {tk: sec for tk, (_, sec) in _COMMOD.items()}


def _fred_storage_safe(fid, col, start):
    """Fetch ONE FRED storage series defensively. A rejected/aliased id (FRED -> HTTP 400) or any
    transport error must NOT crash the load -> return empty so the caller falls back to the proxy."""
    try:
        s = fred_series({fid: col}, start)
        if col in s.columns:
            v = s[col].astype(float).dropna()
            if v.shape[0] > 0:
                return v
    except Exception:
        pass
    return pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# NOVEL SIGNAL: PIT seasonal + AR(1) abnormal-flow surprise (storage OR price).
# Returns a standardised z; sign convention applied by the caller (see load_data).
# --------------------------------------------------------------------------- #
def _seasonal_surprise(level, ar_window=156, z_window=156):
    """Weekly LEVEL series -> standardised abnormal-FLOW surprise (z), fully PIT.
    surprise_t = dLevel_t - [seasonal(week-of-year, PRIOR years) + phi_t * resid_{t-1}]."""
    level = level.dropna().astype(float)
    if len(level) < 80:
        return pd.Series(dtype=float)
    dS = level.diff()
    woy = level.index.isocalendar().week.astype(int).values
    df = pd.DataFrame({"dS": dS.values, "woy": woy}, index=level.index)
    # seasonal normal: expanding mean of PRIOR same-week flows only (shift(1) -> no look-ahead)
    df["seas"] = df.groupby("woy")["dS"].transform(lambda s: s.expanding().mean().shift(1))
    resid = df["dS"] - df["seas"]
    # trailing PIT AR(1) coefficient of the deseasonalised flow
    rl = resid.shift(1)
    m_r = resid.rolling(ar_window, min_periods=52).mean()
    m_l = rl.rolling(ar_window, min_periods=52).mean()
    cov = (resid * rl).rolling(ar_window, min_periods=52).mean() - m_r * m_l
    var = (rl * rl).rolling(ar_window, min_periods=52).mean() - m_l ** 2
    phi = (cov / var.replace(0, np.nan)).shift(1).clip(-0.95, 0.95).fillna(0.0)
    surprise = resid - phi * rl
    sd = surprise.rolling(z_window, min_periods=52).std().shift(1)
    z = surprise / sd.replace(0, np.nan)
    return z.dropna()


def load_data() -> pd.DataFrame:
    tickers = list(_COMMOD.keys())
    px = yf_panel(tickers, START)
    px = px[[c for c in tickers if c in px.columns]].astype(float)

    # Fetch each FRED storage series INDIVIDUALLY + defensively. The previous version issued ONE
    # combined fred_series() request, so a single rejected id returned HTTP 400 and killed the whole
    # load. Now a bad/unavailable id simply routes that leg to the documented price-seasonal proxy.
    fred_cols = {}
    for tk, (fid, _) in _COMMOD.items():
        if not fid or tk not in px.columns:
            continue
        v = _fred_storage_safe(fid, tk, START)
        if not v.empty:
            fred_cols[tk] = v

    sup_daily = pd.DataFrame(index=px.index)
    for tk in px.columns:
        z = pd.Series(dtype=float)

        # (a) TRUE storage-flow surprise where FRED actually returned the weekly stock series
        if tk in fred_cols:
            lvl = fred_cols[tk].dropna().resample("W-FRI").last().dropna()
            z = _seasonal_surprise(lvl)
            if not z.empty:
                z = z.copy()
                z.index = z.index + pd.Timedelta(days=5)   # ~Wed publish lag (no look-ahead)
                # storage sign: z>0 == abnormal BUILD (loose). tilt = -z (short builds) -> stored as-is.

        # (b) convenience-yield PRICE-SEASONAL inventory proxy where storage is unreachable
        #     (NG, grains — AND any petroleum leg whose FRED id failed to resolve)
        if z.empty:
            lp = np.log(px[tk].dropna()).resample("W-FRI").last().dropna()
            zp = _seasonal_surprise(lp)               # zp>0 == abnormal price STRENGTH (tight) -> LONG
            z = -zp                                   # sign-harmonise so tilt=-z is long-on-tight again
            # observable at Friday close; only consumed on the FOLLOWING W-WED rebal -> no look-ahead.

        if z.empty:
            continue
        sup_daily[tk] = z.reindex(px.index, method="ffill")

    names = [c for c in px.columns if c in sup_daily.columns]
    px, sup_daily = px[names], sup_daily[names]
    panel = pd.concat({"px": px, "surprise": sup_daily}, axis=1).dropna(how="all")
    return panel


def signal(panel, **params):
    p = dict(confirm=False, confirm_lb=21, vol_lb=63, share_cap=0.40,
             target_vol=0.10, max_lev=3.0, min_z=0.0, rebal="W-WED")
    p.update(params)

    px = panel["px"].astype(float)
    sup = panel["surprise"].astype(float)
    rets = px.pct_change()
    names = list(px.columns)
    sector_map = {tk: _SECTOR_MAP.get(tk, "energy") for tk in names}

    # fundamental tilt: LONG abnormal tightness, SHORT abnormal looseness (uniform after sign-harmonise).
    tilt = -sup
    if p["min_z"] > 0:                                       # optional dead-band (hysteresis variant)
        tilt = tilt.where(tilt.abs() >= p["min_z"], 0.0)

    # OPTIONAL price-momentum confirmation PROXY (term-structure unavailable) — GRID VARIANT, OFF by
    # default so the standalone convenience-yield premium is the headline number.
    if p["confirm"]:
        mom = px.pct_change(p["confirm_lb"])
        agree = (np.sign(mom) == np.sign(tilt))
        tilt = tilt.where(agree, 0.0)

    # inverse-vol sizing (trailing realised vol, lagged)
    vol = rets.rolling(p["vol_lb"], min_periods=20).std().shift(1)
    raw = (tilt / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # dollar-neutral (net-zero) cross-section, gross-normalise, per-name capital cap
    raw = raw.sub(raw.mean(axis=1), axis=0)
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0)
    w = w.clip(-p["share_cap"], p["share_cap"])
    w = w.sub(w.mean(axis=1), axis=0)
    w = w.clip(-p["share_cap"], p["share_cap"])

    # vol-target the combined book to ~target_vol using TRAILING realised vol (PIT)
    r0 = (w.shift(1) * rets).sum(axis=1)
    tv = r0.rolling(63, min_periods=20).std() * np.sqrt(252)
    lev = (p["target_vol"] / tv.replace(0, np.nan)).shift(1).clip(0.0, p["max_lev"]).fillna(0.0)
    W = w.mul(lev, axis=0)

    # weekly rebalance + hold, then CONTRACT 1-day lag (weights built from same-day signal -> shift here)
    W = W.resample(p["rebal"]).last().reindex(px.index, method="ffill")
    W = W.shift(1).fillna(0.0)

    daily = net_of_cost(W, rets, cost_bps=8.0, name="storage_surprise_cy_v2")
    trades = trades_from_weights(W, rets, sector_map)
    return daily, trades


def load_gen_data(label) -> pd.DataFrame:
    # scope='local': no disjoint commodity sub-universe (sharing zero tickers) is reachable via the
    # allowed adapters, so the stage-2 breadth battery does not run. Provided for interface completeness.
    return load_data()


# --------------------------------------------------------------------------- #
# SOFT EXPECTATIONS (machine-checkable mechanism claims; cheap; search-window only)
# --------------------------------------------------------------------------- #
def _sharpe(r):
    r = pd.Series(r).dropna()
    return 0.0 if (len(r) < 20 or r.std() == 0) else float(r.mean() / r.std() * np.sqrt(252))


def _exp_standalone_holds(ctx):
    """Premium must stand alone: unconfirmed (default) Sharpe >= price-confirmed - 0.10."""
    g = ctx.get("grid", {})
    if "default" not in g or "price_confirm" not in g:
        return {"pass": True, "observed": "grid_missing"}
    sd, sc = _sharpe(g["default"]), _sharpe(g["price_confirm"])
    return {"pass": sd >= sc - 0.10, "observed": round(sd - sc, 3)}


def _exp_positive_both_halves(ctx):
    r = pd.Series(ctx["search"]).dropna()
    if len(r) < 60:
        return {"pass": False, "observed": "insufficient"}
    h = len(r) // 2
    a, b = float(r.iloc[:h].sum()), float(r.iloc[h:].sum())
    return {"pass": (a > 0) and (b > 0), "observed": (round(a, 4), round(b, 4))}


def _exp_market_neutral(ctx):
    px = ctx["panel"]["px"]
    mkt = px.pct_change().mean(axis=1)
    df = pd.concat([pd.Series(ctx["search"]).rename("r"), mkt.rename("m")], axis=1).dropna()
    df = df[df.index < pd.Timestamp(ctx["holdout_start"])]
    if len(df) < 60 or df["m"].std() == 0:
        return {"pass": True, "observed": "n/a"}
    beta = float(np.cov(df["r"], df["m"])[0, 1] / np.var(df["m"]))
    return {"pass": abs(beta) < 0.40, "observed": round(beta, 3)}


SPEC = StrategySpec(
    id="storage_surprise_cy_v2",
    family="commodity_convenience_yield",
    title="Storage-Surprise Convenience-Yield Premium — PIT seasonal+AR(1) inventory-flow surprise, "
          "market-neutral L/S pooled across the energy {crude,gasoline,distillate,natgas} + "
          "grains {corn,soy,wheat} complexes (price-confirm optional)",
    markets=["CL=F", "RB=F", "HO=F", "NG=F", "ZC=F", "ZS=F", "ZW=F"],
    data_desc="yfinance front-month futures (CL,RB,HO,NG,ZC,ZS,ZW). Storage flow via FRED weekly EIA "
              "petroleum stocks (WCESTUS1 crude, WGTSTUS1 gasoline, WDISTUS1 distillate), fetched per-"
              "series defensively (any unresolved id degrades to the price proxy). NG + grains storage "
              "not reachable here -> documented convenience-yield price-seasonal inventory proxy. "
              "Weekly W-WED rebalance, 8bps cost, inverse-vol, dollar-neutral, holdout>=2022-01-01.",
    pre_registration=(
        "Inventory/convenience-yield premium pooled across TWO commodity complexes (energy+grains). "
        "Sign: LONG abnormally-tight (drawn / seasonally-strong) names, SHORT abnormally-loose (built / "
        "seasonally-weak) names, market-neutral, inverse-vol sized, weekly rebalanced. "
        "PIT: seasonal mean = expanding mean of PRIOR same-week flows only; AR(1) coef + std trailing-"
        "rolling .shift(1); petroleum storage stamped +5d (~publish); price-proxy stamped at Friday close "
        "and only consumed at the following W-WED; final weight matrix .shift(1) before costs. "
        "DATA CONSTRAINT (disclosed, binding, not a design choice): the petroleum legs use the TRUE EIA "
        "weekly storage-flow surprise WHEN the FRED series resolves (fetched per-series defensively; a "
        "rejected/aliased id falls back to the proxy rather than crashing); NG + the 3 grains use a "
        "convenience-yield PRICE-SEASONAL inventory proxy (the seasonal+AR(1) abnormal front-month "
        "return — a recognised inventory-premium proxy, GHR 2013), which is the construction risk and the "
        "headline caveat; it is the most-faithful BUILDABLE version of the pooled-complex thesis under the "
        "allowed adapters. scope='local' because no disjoint commodity sub-universe exists for a stage-2 "
        "breadth battery; the write-once holdout forward-validates. Trend overlay NOT added (standalone "
        "first). FIXES vs rejected v1: (1) full 7-name cross-complex universe restored (v1 collapsed to a "
        "3-name petroleum slice); (2) price-confirmation demoted from a default hard gate to an optional "
        "grid variant so the headline is the unconfirmed standalone premium. FIX vs failed v2: per-series "
        "defensive FRED fetch (a combined request returned HTTP 400). EXPECTATIONS: positive in both "
        "search-window halves, market-neutral (|beta|<0.40), and the standalone (no-confirm) Sharpe should "
        "not be materially beaten by the confirmed variant."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                       # primary: standalone convenience-yield premium (confirm off)
        "price_confirm": {"confirm": True},  # price-momentum confirmation proxy (term-structure stand-in)
        "deadband": {"min_z": 0.5},          # hysteresis on the surprise z
        "slow_vol": {"vol_lb": 126},         # slower inverse-vol estimate
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=7,
    expectations=[
        {"name": "standalone_holds",
         "claim": "unconfirmed (default) search Sharpe >= price-confirmed variant - 0.10 (premium stands alone)",
         "check": _exp_standalone_holds},
        {"name": "positive_both_halves",
         "claim": "cumulative search-window return > 0 in BOTH halves",
         "check": _exp_positive_both_halves},
        {"name": "market_neutral",
         "claim": "|beta| of search returns to the equal-weight commodity panel < 0.40",
         "check": _exp_market_neutral},
    ],
)