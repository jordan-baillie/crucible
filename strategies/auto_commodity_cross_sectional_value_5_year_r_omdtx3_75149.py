"""
Commodity cross-sectional VALUE — 5-year real-price mean-reversion (Asness-Moskowitz-Pedersen,
"Value and Momentum Everywhere", 2013).  STANDALONE value leg.

MECHANISM (frozen, AMP form): monthly cross-section of the 16 owned commodity roots (PA dropped).
    value_t = -log(P_t / P_{t-60m})   on each root's Yahoo back-adjusted continuous price.
Cheap (price well below its 5y level) -> high value -> LONG; expensive -> SHORT.  Rank across roots,
equal-RISK (inverse-vol) LONG top tercile / SHORT bottom tercile, dollar-neutral, vol-target the whole
book to ~10% annualized, gross <= 2x.  Monthly signal ffilled to daily, 8bps cost on turnover, signals
lagged 1 trading day (the shift(1) is explicit, my responsibility — see signal()).  No look-ahead.

INVENTORY VETO (the proposal's fundamentals value-trap overlay) is DEFERRED, ON PURPOSE.  It needs
eia_series (energy stocks) + usda_nass (grain/oilseed stocks); NEITHER is in this harness's tested-import
whitelist (eia_series is KEY-PENDING per the 2026-06-16 data audit; there is no tested usda_nass adapter).
Hand-rolling those raw loaders would be a fresh look-ahead surface the rails cannot see, and the proposal
itself says "test value STANDALONE first".  So THIS module registers the clean value premium only; the
veto is bolted on (re-run) once those adapters are provisioned + live-verified.

SCOPE = broad.  Value / long-horizon reversal is claimed UNIVERSAL (AMP).  A 16-root cross-section is far
too small to host the contract's ~150-400-name generalization universes, so the SAME frozen signal+params
is stress-tested where the mechanism's disjoint cousin lives: equity LONG-TERM REVERSAL (DeBondt-Thaler =
the equity analog of the identical 5y price-reversal signal), on three DISJOINT mid-cap GICS-sector slices
that share no tickers with each other or with the futures.  >=60% OOS-positive on the holdouts or the
commodity result is an overfit/lucky-complex outlier and is rejected.  The inventory veto is storage-native
and is NOT expected to generalize (and is not part of this standalone book anyway).
"""
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

# ---- commodity cross-section: 16 owned roots via Yahoo continuous futures (PA dropped) ----
_COMMODITY = {
    "CL=F": "ENERGY", "NG=F": "ENERGY", "HO=F": "ENERGY", "RB=F": "ENERGY",
    "GC=F": "METALS", "SI=F": "METALS", "HG=F": "METALS", "PL=F": "METALS",
    "ZC=F": "GRAINS", "ZS=F": "GRAINS", "ZW=F": "GRAINS", "ZL=F": "GRAINS", "ZM=F": "GRAINS",
    "LE=F": "LIVESTOCK", "HE=F": "LIVESTOCK", "GF=F": "LIVESTOCK",
}
_START = "2001-01-01"

# equity generalization slices (DISJOINT GICS sectors; long-term reversal = equity "value")
_GEN = {
    "equity_energy_util":    ["Energy", "Utilities"],
    "equity_materials_ind":  ["Basic Materials", "Industrials"],
    "equity_staples_health": ["Consumer Defensive", "Healthcare"],
}

# module-level sector registry so signal() can resolve sectors for ANY panel it is handed
_SECTOR_REGISTRY = {}
_EQ_CACHE = {}


def _register(sector_map):
    _SECTOR_REGISTRY.update(sector_map)


def _me_freq():
    test = pd.Series([0.0], index=pd.to_datetime(["2020-01-31"]))
    for f in ("ME", "M"):
        try:
            test.resample(f).last()
            return f
        except Exception:
            continue
    return "ME"


_MEF = _me_freq()


def _eq_full():
    if "u" not in _EQ_CACHE:
        t, s = sector_universe(marketcap="Mid", top_n_per_sector=200)
        _EQ_CACHE["u"] = (list(t), dict(s))
    return _EQ_CACHE["u"]


# ----------------------------------------------------------------------------- data loaders
def load_data() -> pd.DataFrame:
    px = yf_panel(list(_COMMODITY), start=_START)
    keep = [c for c in _COMMODITY if c in px.columns]
    px = px.reindex(columns=keep).sort_index().dropna(how="all")
    smap = {c: _COMMODITY[c] for c in px.columns}
    _register(smap)
    px.attrs["sector_map"] = smap
    px.attrs["name"] = "commod_value"
    return px


def load_gen_data(label) -> pd.DataFrame:
    want = set(_GEN[label])
    tickers, smap = _eq_full()
    sel = [t for t in tickers if smap.get(t) in want]
    px = sep_panel(sel, start=_START).sort_index()
    px = px.dropna(how="all", axis=1).dropna(how="all", axis=0)
    smap2 = {t: str(smap.get(t, "OTHER")).upper() for t in px.columns}
    _register(smap2)
    px.attrs["sector_map"] = smap2
    px.attrs["name"] = label
    return px


# ----------------------------------------------------------------------------- the signal
def _book_weights(px, lookback_m=60, tercile_frac=1.0 / 3.0, vol_lb=63,
                  target_vol=0.10, min_names=6, gross_cap=2.0):
    """Same-day (UN-lagged) target weights; signal() applies the 1-day execution lag."""
    px = px.sort_index()
    rets = px.pct_change()

    # 5y (60-month) reversal value on month-end levels
    mp = px.resample(_MEF).last()
    val = -np.log(mp / mp.shift(lookback_m))
    val = val.replace([np.inf, -np.inf], np.nan)

    # cross-sectional terciles (cheapest -> long, dearest -> short)
    n = val.count(axis=1)
    r = val.rank(axis=1, pct=True)
    long_mask = r.ge(1.0 - tercile_frac) & n.ge(min_names).values[:, None] if False else (
        r.ge(1.0 - tercile_frac).where(n.ge(min_names), other=False))
    short_mask = r.le(tercile_frac).where(n.ge(min_names), other=False)
    long_mask = long_mask.fillna(False)
    short_mask = short_mask.fillna(False)

    # equal-RISK (inverse-vol), trailing daily vol sampled at month-end
    mvol = rets.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std().resample(_MEF).last()
    iv = 1.0 / mvol.clip(lower=1e-4)

    long_w = iv.where(long_mask, 0.0).fillna(0.0)
    short_w = iv.where(short_mask, 0.0).fillna(0.0)
    long_w = long_w.div(long_w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    short_w = short_w.div(short_w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    mW = 0.5 * long_w - 0.5 * short_w                       # dollar-neutral, gross ~1

    # vol-target the whole book to ~target_vol annualized (trailing, lagged -> no look-ahead)
    Wd = mW.reindex(px.index, method="ffill").reindex(columns=px.columns).fillna(0.0)
    base_ret = (Wd.shift(1) * rets).sum(axis=1, skipna=True)
    pv = base_ret.rolling(vol_lb, min_periods=max(20, vol_lb // 2)).std()
    scale = (target_vol / np.sqrt(252.0)) / pv
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=4.0).fillna(0.0)
    W = Wd.mul(scale, axis=0)

    # gross cap (<= 2x)
    gross = W.abs().sum(axis=1)
    capf = pd.Series(1.0, index=W.index)
    over = gross > gross_cap
    capf[over] = gross_cap / gross[over]
    W = W.mul(capf, axis=0)
    return W


def signal(panel, **params):
    p = dict(lookback_m=60, tercile_frac=1.0 / 3.0, vol_lb=63, target_vol=0.10,
             min_names=6, gross_cap=2.0, cost_bps=8.0)
    p.update(params or {})

    px = panel.sort_index()
    W = _book_weights(px, lookback_m=p["lookback_m"], tercile_frac=p["tercile_frac"],
                      vol_lb=p["vol_lb"], target_vol=p["target_vol"],
                      min_names=p["min_names"], gross_cap=p["gross_cap"])
    Wl = W.shift(1).fillna(0.0)                              # explicit 1-day execution lag

    rets = px.pct_change().reindex(index=Wl.index, columns=Wl.columns).fillna(0.0)
    smap = panel.attrs.get("sector_map") or {
        c: _SECTOR_REGISTRY.get(c, "OTHER") for c in px.columns}
    name = panel.attrs.get("name", "commod_value")

    daily = net_of_cost(Wl, rets, cost_bps=p["cost_bps"], name=name)
    trades = trades_from_weights(Wl, rets, smap)
    return daily, trades


# ----------------------------------------------------------------------------- soft expectations
def _chk_market_neutral(ctx):
    """Book is dollar/market-neutral -> |beta to equal-weight commodity universe| small
    (so MCPT may use the ABSOLUTE null). Sliced to the search window."""
    search = ctx.get("search"); panel = ctx.get("panel")
    if search is None or panel is None or len(search) < 60:
        return {"pass": True, "observed": "n/a"}
    ew = panel.pct_change().mean(axis=1)
    df = pd.concat([pd.Series(search).rename("p"), ew.rename("m")], axis=1).dropna()
    df = df[df.index < pd.Timestamp(ctx["holdout_start"])]
    if len(df) < 60:
        return {"pass": True, "observed": "n/a"}
    beta = float(np.polyfit(df["m"].values, df["p"].values, 1)[0])
    return {"pass": abs(beta) <= 0.30, "observed": round(beta, 3)}


def _chk_slow_signal(ctx):
    """A 5y reversal is a SLOW, low-turnover signal -> mean trade hold is multi-week."""
    trades = ctx.get("trades") or []
    if not trades:
        return {"pass": True, "observed": 0}
    hd = float(np.mean([t.get("hold_days", 0) for t in trades]))
    return {"pass": hd >= 21, "observed": round(hd, 1)}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="commod_value_xs_amp_v1",
    family="value",
    title="Commodity cross-sectional VALUE (5y real-price reversal, AMP 2013) — standalone leg",
    markets=["commodity_futures"],
    data_desc=("Yahoo continuous futures, 16 owned roots (energy CL/NG/HO/RB, metals GC/SI/HG/PL, "
               "grains+oilseeds ZC/ZS/ZW/ZL/ZM, livestock LE/HE/GF; PA dropped). Equity "
               "generalization via survivorship-clean Sharadar SEP mid-cap GICS sectors."),
    pre_registration=(
        "PREMIUM: long-horizon mean-reversion (value) risk premium in commodity futures — contrarian "
        "compensation for holding distressed-cheap commodities through demand uncertainty (AMP 2013).\n"
        "SIGNAL (frozen): monthly cross-section, value_t = -log(P_t / P_{t-60m}) on back-adjusted "
        "continuous prices; equal-RISK (inverse-vol) LONG top tercile / SHORT bottom tercile; "
        "dollar-neutral; vol-target ~10% ann; gross<=2x; signals lagged 1 day; 8bps cost on turnover.\n"
        "DEFERRED: the inventory-fundamentals value-trap VETO (eia_series + usda_nass) is NOT in this "
        "run — those adapters are not in the tested-import whitelist (eia key pending; no usda adapter). "
        "Per 'test value STANDALONE first', this registers the value leg only; the veto is re-run once "
        "the loaders are provisioned + verified. Implementing them here = un-whitelisted raw loaders = "
        "an unseen look-ahead surface.\n"
        "MACHINE-CHECKED EXPECTATIONS: (1) |beta of book to equal-weight commodity universe| <= 0.30 "
        "over the search window (absolute MCPT null valid); (2) mean trade hold >= 21 trading days "
        "(5y reversal is a slow, low-turnover signal).\n"
        "PASS BAR: positive OOS Sharpe on the 2022-01-01+ commodity holdout, MCPT (absolute null).\n"
        "GENERALIZATION (broad): value / long-horizon reversal is claimed universal. A 16-root cross-"
        "section cannot host 150-400-name universes, so the SAME frozen signal+params is stress-tested "
        "on its disjoint equity cousin — long-term reversal (DeBondt-Thaler), the equity analog of the "
        "identical 5y price-reversal — across three DISJOINT mid-cap sector slices (energy+utilities / "
        "materials+industrials / staples+healthcare) that share no tickers. >=60% of holdouts OOS-"
        "positive or the commodity result is an overfit/lucky-complex outlier and is rejected. The "
        "inventory veto is storage-native and is NOT expected to generalize (and is not in this book).\n"
        "DESIGN FROZEN. Grid {default, lb_48, lb_72, quartile} declares the search burden for DSR."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "lb_48": {"lookback_m": 48},
        "lb_72": {"lookback_m": 72},
        "quartile": {"tercile_frac": 0.25},
    },
    scope="broad",
    generalization_universes=list(_GEN),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=14,
    expectations=[
        {"name": "market_neutral",
         "claim": "|beta of book to equal-weight commodity universe| <= 0.30 over search window",
         "check": _chk_market_neutral},
        {"name": "slow_value_signal",
         "claim": "mean trade hold >= 21 trading days (5y reversal is slow / low-turnover)",
         "check": _chk_slow_signal},
    ],
)