"""
Illiquidity-Premium × Trend Crisis-Alpha — TURNOVER-HARDENED DEPLOYABLE-SLEEVE variant.

Two-premium book:
  LEG A (alpha)  : Amihud illiquidity premium, tranched long/short cross-section in US
                   small-caps (long the most-illiquid quintile, short the most-liquid).
  LEG B (sleeve) : cross-asset time-series-trend crisis alpha on a FIXED 5-ETF list
                   (SPY, EFA, TLT, GLD, DBC), signal evaluated DAILY (stays responsive to
                   fast crashes) but EXECUTED through a pre-registered no-trade band
                   (>20% relative drift) + $50 min-trade floor, and charged a conservative
                   retail cost model (per-ETF half-spread + 2bps slippage on every fill).

The ONLY new claim vs the parent deployable-sleeve book is whether the combination's
complementarity survives once the trend sleeve is charged for the trades it ACTUALLY makes
(the last untested seam: trading frictions).  All gates are evaluated NET of these costs.

HONESTY NOTE (recorded in pre_registration): the parent's deployed Amihud leg would, on a
real promotion, be HAND-EDITED in place (never re-codegen'd from prose — crucible lesson).
This research module is a self-contained, faithful RECONSTRUCTION of the tranched Amihud
long/short used for in-harness validation of the combination; it is not a byte-for-byte
import of the deployed module.  The mutation under test (band + floor + cost model) is the
novel, pre-registered code.

NO look-ahead: every signal is lagged 1 day (inv_vol_position returns already-lagged weekly
positions; the trend target weights are explicitly .shift(1) before execution).
NO external side effects (in-memory caches only; no writes / capital / config).
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, inv_vol_position, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- constants
START = "2005-01-01"
ETFS = ["SPY", "EFA", "TLT", "GLD", "DBC"]            # fixed cross-asset trend sleeve
SMALLCAP_TOP_PER_SECTOR = 130                          # ~1400 sector-spread small-caps

# conservative retail friction model (bps, per executed ETF trade).  Half-spread is set
# ABOVE typical Alpaca/IB quotes for each instrument; +2bps slippage on top.  If real
# spreads were ever tighter than this, this assumption is simply too cautious (the safe
# direction).  gate0 (e) requires confirming these are conservative before pre-registering.
HALF_SPREAD_BPS = {"SPY": 1.0, "EFA": 2.5, "TLT": 1.5, "GLD": 1.5, "DBC": 3.5}
SLIPPAGE_BPS = 2.0
COST_BPS_VEC = np.array([HALF_SPREAD_BPS[t] + SLIPPAGE_BPS for t in ETFS], dtype=float)

DEFAULTS = dict(
    # --- Amihud leg (frozen reconstruction) ---
    illiq_lb=60,            # Amihud averaging window (days)
    tranche_q=0.20,         # long top 20% illiquid, short bottom 20% (tranched -> low churn)
    amihud_target_vol=0.10, # leg vol target for inv_vol sizing
    vol_lb=60,
    max_pos=0.05,           # per-name weight cap
    # --- trend sleeve (canonical, daily signal) ---
    trend_lookbacks=(63, 126, 252),
    sleeve_target_vol=0.15,
    # --- execution mutation (pre-registered single values, NOT tuned, NOT a grid) ---
    band=0.20,              # rebalance an ETF only when target drifts >20% (relative)
    min_trade=50.0,         # skip any fill < $50 (fractional-share / dust floor)
    capital=5000.0,         # $5K retail book (only used for the $ min-trade floor)
    # --- blend ---
    sleeve_risk=0.25,       # trend at 25% of book risk, Amihud at 100% (both vol-matched)
    blend_target_vol=0.10,
    blend_vol_lb=60,
)

# in-memory caches (pure performance; no side effects) -------------------------------------
_UNIV_CACHE, _VOL_CACHE, _ETF_CACHE, _ALL_CACHE = {}, {}, {}, {}


def _universe():
    if "u" not in _UNIV_CACHE:
        _UNIV_CACHE["u"] = sector_universe(marketcap="Small",
                                           top_n_per_sector=SMALLCAP_TOP_PER_SECTOR)
    return _UNIV_CACHE["u"]


def _volume(tickers):
    key = tuple(sorted(tickers))
    if key not in _VOL_CACHE:
        _VOL_CACHE[key] = sep_panel(list(tickers), start=START, field="volume")
    return _VOL_CACHE[key]


def _etf_prices():
    if "p" not in _ETF_CACHE:
        px = yf_panel(ETFS, start=START)
        cols = [c for c in ETFS if c in px.columns]
        _ETF_CACHE["p"] = px[cols]
    return _ETF_CACHE["p"]


# ------------------------------------------------------------------- Amihud illiquidity leg
def _amihud_signal(px, vol, p):
    """Cross-sectional Amihud illiquidity z-score (high z = illiquid). NaN-preserving."""
    rets = px.pct_change()
    dvol = (px * vol)
    dvol = dvol.where(dvol > 0)                      # dollar volume, drop zero/halt days
    illiq = (rets.abs() / dvol) * 1e6               # Amihud (|ret| per $MM traded)
    illiq_avg = illiq.rolling(p["illiq_lb"],
                              min_periods=max(10, p["illiq_lb"] // 2)).mean()
    z = xs_zscore(illiq_avg)                         # winsorized x-sec z (kit)
    return z, rets


def _tranche(z, q):
    """+1 = most-illiquid q tranche (long), -1 = most-liquid q tranche (short)."""
    rank = z.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    sig = sig.mask(rank >= 1.0 - q, 1.0)
    sig = sig.mask(rank <= q, -1.0)
    return sig.where(~z.isna(), np.nan)


# -------------------------------------------------------------------- trend sleeve (5 ETFs)
def _trend_target_weights(px, p):
    """Canonical multi-lookback time-series momentum, per-instrument vol-targeted.
    Same-day weights (lagged by the caller via .shift(1))."""
    rets = px.pct_change()
    sig = None
    for lb in p["trend_lookbacks"]:
        s = np.sign(px / px.shift(lb) - 1.0)
        sig = s if sig is None else sig + s
    sig = sig / float(len(p["trend_lookbacks"]))             # ensemble in [-1, 1]
    vol = rets.rolling(p["vol_lb"], min_periods=p["vol_lb"] // 2).std()
    inst = (p["sleeve_target_vol"] / np.sqrt(252)) / vol.clip(lower=1e-4)
    w = (sig * inst).clip(-2.0, 2.0)
    return w / float(px.shape[1])                            # equal-weight across instruments


def _run_sleeve(target_W, etf_rets, p, banded=True):
    """Stateful execution of the sleeve.  target_W MUST already be lagged.

    banded=True  -> only fill an ETF when |target-held|/|held| > band AND |fill$| >= min_trade
    banded=False -> full daily rebalance to target (frictionless-churn baseline)
    Costs (half-spread + slippage) are charged on every executed fill in BOTH modes.
    Returns (net_series, gross_series, ann_one_way_turnover)."""
    dates = target_W.index
    n = len(ETFS)
    tW = target_W.reindex(columns=ETFS).values
    rW = etf_rets.reindex(columns=ETFS).values
    held = np.zeros(n)
    cap, band, mintrade = p["capital"], p["band"], p["min_trade"]
    net, gross, turn = [], [], 0.0
    for i in range(len(dates)):
        tgt = tW[i].copy()
        nan_mask = np.isnan(tgt)
        tgt[nan_mask] = held[nan_mask]                       # no signal yet -> hold
        if banded:
            new = held.copy()
            for j in range(n):
                delta = abs(tgt[j] - held[j])
                drift = delta / max(abs(held[j]), 1e-6)
                if drift > band and delta * cap >= mintrade:
                    new[j] = tgt[j]
        else:
            new = tgt
        traded = np.abs(new - held)
        cost_frac = float(np.nansum(traded * COST_BPS_VEC) / 1e4)
        turn += float(np.nansum(traded))
        held = new
        r = np.where(np.isnan(rW[i]), 0.0, rW[i])
        g = float(np.nansum(held * r))
        gross.append(g)
        net.append(g - cost_frac)
        held = held * (1.0 + r)                              # positions drift with price
    idx = dates
    ann_turn = (turn / max(len(dates), 1)) * 252.0
    return (pd.Series(net, index=idx), pd.Series(gross, index=idx), ann_turn)


# ----------------------------------------------------------------------------- blend helpers
def _target_vol(r, target_ann, lb):
    """Scale a return stream to a trailing (lagged) annualized vol target. No look-ahead."""
    dv = target_ann / np.sqrt(252)
    v = r.rolling(lb, min_periods=lb // 2).std().shift(1)
    scale = (dv / v).replace([np.inf, -np.inf], np.nan).clip(upper=4.0).fillna(0.0)
    return r * scale


def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _maxdd(r):
    c = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return float(-(c / c.cummax() - 1.0).min())


# ------------------------------------------------------------------- full book (memoized)
def _compute_all(panel, **params):
    p = {**DEFAULTS, **params}
    key = (panel.index[0], panel.index[-1], panel.shape,
           tuple(sorted((k, v) for k, v in params.items())))
    if key in _ALL_CACHE:
        return _ALL_CACHE[key]

    tickers = list(panel.columns)
    _, sector_map = _universe()
    vol = _volume(tickers).reindex(panel.index)
    px = panel

    # ---- LEG A : tranched Amihud long/short (inv-vol sized, weekly, kit-lagged) ----
    z, rets = _amihud_signal(px, vol, p)
    tr = _tranche(z, p["tranche_q"])
    W = inv_vol_position(tr, rets, target_vol=p["amihud_target_vol"],
                         vol_lb=p["vol_lb"], max_pos=p["max_pos"], rebalance="W")
    # inv_vol_position returns already-lagged weekly positions -> do NOT shift again.
    amihud_net = net_of_cost(W, rets, cost_bps=8.0, name="amihud")
    trades = trades_from_weights(W, rets, sector_map)        # kit stamps entry_regime

    # ---- LEG B : throttled 5-ETF trend sleeve (signal daily, execution banded) ----
    etf_px = _etf_prices().reindex(panel.index).ffill()
    etf_rets = etf_px.pct_change()
    target_W = _trend_target_weights(etf_px, p).shift(1)     # explicit 1-day signal lag
    sleeve_net, sleeve_gross, turn_b = _run_sleeve(target_W, etf_rets, p, banded=True)
    sleeve_net_u, _, turn_u = _run_sleeve(target_W, etf_rets, p, banded=False)

    # ---- vol-matched blend : Amihud @1.0 risk, sleeve @0.25 risk ----
    a_scaled = _target_vol(amihud_net.dropna(), p["blend_target_vol"], p["blend_vol_lb"])
    s_scaled = _target_vol(sleeve_net, p["blend_target_vol"], p["blend_vol_lb"])
    s_scaled = s_scaled.reindex(a_scaled.index).fillna(0.0)
    combined = a_scaled + p["sleeve_risk"] * s_scaled
    nz = a_scaled.ne(0)
    if nz.any():                                             # trim leading warmup (no-position)
        combined = combined.loc[a_scaled.index[nz.values.argmax()]:]

    out = dict(amihud_net=amihud_net, amihud_scaled=a_scaled.reindex(combined.index),
               sleeve_net=sleeve_net, sleeve_net_unbanded=sleeve_net_u,
               sleeve_gross=sleeve_gross, turnover_banded=turn_b, turnover_unbanded=turn_u,
               combined=combined, trades=trades)
    _ALL_CACHE[key] = out
    return out


# ------------------------------------------------------------------------ harness contract
def load_data():
    """Survivorship-clean small-cap close panel (the Amihud leg's universe). The ETF sleeve
    and per-name volume are loaded inside signal() via adapters (kept off the main panel
    because they have a different shape)."""
    tickers, _ = _universe()
    return sep_panel(tickers, start=START, field="closeadj")


def signal(panel, **params):
    res = _compute_all(panel, **params)
    out = res["combined"].copy()
    out.name = "illiq_x_trend_throttled"
    return out, res["trades"]


def load_gen_data(label):
    """scope='local': the generalization battery does not run (both legs' standalone
    universality is already settled). Defined for signature completeness only."""
    return pd.DataFrame()


# --------------------------------------------------------- soft expectations (pre-registered)
def _pre(ctx):
    panel = ctx["panel"]
    hs = pd.Timestamp(ctx["holdout_start"])
    return panel.loc[panel.index < hs]


def _safe(fn):
    def wrap(ctx):
        try:
            return fn(ctx)
        except Exception as e:                               # soft fail -> recorded, not raised
            return {"pass": False, "observed": f"error: {type(e).__name__}"}
    return wrap


@_safe
def _chk_turnover_cut(ctx):
    r = _compute_all(_pre(ctx))
    tb, tu = r["turnover_banded"], r["turnover_unbanded"]
    red = (1.0 - tb / tu) if tu > 0 else 0.0
    return {"pass": red >= 0.30, "observed": round(float(red), 3)}


@_safe
def _chk_leg_corr(ctx):
    r = _compute_all(_pre(ctx))
    df = pd.concat([r["amihud_net"], r["sleeve_net"]], axis=1).dropna()
    c = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return {"pass": c <= 0.10, "observed": round(c, 3)}


@_safe
def _chk_sharpe_degrade(ctx):
    r = _compute_all(_pre(ctx))
    sb, sc = _sharpe(r["amihud_scaled"]), _sharpe(r["combined"])
    degr = (sb - sc) / abs(sb) if sb != 0 else 1.0
    return {"pass": degr <= 0.10, "observed": round(float(degr), 3)}


@_safe
def _chk_maxdd_reduce(ctx):
    r = _compute_all(_pre(ctx))
    db, dc = _maxdd(r["amihud_scaled"]), _maxdd(r["combined"])
    red = (1.0 - dc / db) if db > 0 else 0.0
    return {"pass": red >= 0.20, "observed": round(float(red), 3)}


@_safe
def _chk_crisis_survives(ctx):
    r = _compute_all(_pre(ctx))
    w = r["sleeve_net"].loc["2020-02-15":"2020-04-30"]
    cum = float((1.0 + w).prod() - 1.0)
    return {"pass": cum > 0.0, "observed": round(cum, 4)}    # crisis alpha survives the band


@_safe
def _chk_trend_track(ctx):
    r = _compute_all(_pre(ctx))
    tr, _ = trend_returns()
    tr = tr.loc[tr.index < pd.Timestamp(ctx["holdout_start"])]
    df = pd.concat([r["sleeve_net"], tr], axis=1).dropna()
    if len(df) < 60:
        return {"pass": False, "observed": "insufficient overlap"}
    c = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return {"pass": c >= 0.30, "observed": round(c, 3)}      # throttled sleeve ~ canonical CTA


# ------------------------------------------------------------------------------------ SPEC
SPEC = StrategySpec(
    id="illiq_x_trend_throttled_v1",
    family="illiquidity_trend_combo",
    title="Illiquidity-Premium × Trend Crisis-Alpha — turnover-hardened deployable sleeve "
          "(banded execution + conservative retail cost model)",
    markets=["US small-cap equities (Amihud tranched L/S)",
             "cross-asset ETF trend sleeve: SPY/EFA/TLT/GLD/DBC"],
    data_desc="Sharadar SEP/TICKERS (survivorship-clean, owned) close+volume for the small-cap "
              "Amihud leg; yfinance daily closes (free) for the 5 trend ETFs. No new data; the "
              "cost model is a fixed conservative parametric assumption, not a dataset.",
    pre_registration=(
        "FROZEN, SINGLE-SPEC, NO GRID (effective search burden N=1). Two-premium book: a tranched "
        "Amihud illiquidity long/short in small-caps (long most-illiquid quintile, short most-liquid; "
        "60d Amihud, inv-vol sized, weekly, 8bps) hedged by a 5-ETF time-series-trend sleeve whose "
        "SIGNAL is evaluated DAILY (stays responsive to fast crashes) but whose EXECUTION is routed "
        "through a PRE-REGISTERED no-trade band: rebalance an ETF only on >20% relative drift, skip any "
        "fill < $50, and charge a conservative per-ETF half-spread + 2bps slippage on every fill. ALL "
        "gates and success criteria are evaluated NET of these costs. Legs vol-matched (trailing-60d), "
        "Amihud @1.0 risk, trend @0.25 risk. PRE-REGISTERED success criteria vs standalone (vol-matched) "
        "Amihud over the identical pre-holdout window, all net: (1) MaxDD reduced >=20%; (2) net-Sharpe "
        "degradation <=10%; (3) leg correlation <= +0.1; (4) the throttled sleeve stays POSITIVE in the "
        "2020 crash window (if the band kills the crisis-alpha sign, the book FAILS HONESTLY — that is "
        "the entire point). These four + a band-turnover-cut check + a CTA tracking check are declared "
        "machine-checkable in expectations[]. Diagnostics reported-not-gated: gross-vs-net sleeve Sharpe "
        "and turnover with/without the band. The band/floor/cost numbers are deployment-reality values "
        "written down before any backtest and NEVER tuned or re-rolled. HONESTY: this is a self-contained "
        "RECONSTRUCTION of the deployed Amihud tranched L/S for in-harness combination testing — a real "
        "promotion would HAND-EDIT the deployed module in place, never re-codegen it from prose."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}},                       # frozen design: one pre-registered configuration
    scope="local",                              # only NEW claims are book complementarity + cost-survival
    generalization_universes=[],                # standalone universality of both legs already settled
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",                 # 2015-16 & 2020 stress in search; 2022 stress is OOS
    deploy_max_positions=20,
    hedge_tickers=ETFS,                         # the trend sleeve = declared whitelisted ETF overlay
    hedge_cap=0.50,                             # gated on whitelist + position-day cap, not single_name
    expectations=[
        {"name": "band_cuts_turnover",
         "claim": "no-trade band cuts trend-sleeve annualized turnover by >=30% vs daily rebalance",
         "check": _chk_turnover_cut},
        {"name": "leg_correlation_low",
         "claim": "Amihud leg vs throttled trend sleeve correlation <= +0.1",
         "check": _chk_leg_corr},
        {"name": "sharpe_degradation_capped",
         "claim": "combined net Sharpe degrades <=10% vs standalone vol-matched Amihud",
         "check": _chk_sharpe_degrade},
        {"name": "maxdd_reduced",
         "claim": "combined MaxDD reduced >=20% vs standalone vol-matched Amihud",
         "check": _chk_maxdd_reduce},
        {"name": "crisis_alpha_survives_band",
         "claim": "throttled trend sleeve remains positive across the 2020-02-15..2020-04-30 crash",
         "check": _chk_crisis_survives},
        {"name": "tracks_canonical_trend",
         "claim": "throttled 5-ETF sleeve tracks canonical 21-market CTA (corr >= 0.3)",
         "check": _chk_trend_track},
    ],
)