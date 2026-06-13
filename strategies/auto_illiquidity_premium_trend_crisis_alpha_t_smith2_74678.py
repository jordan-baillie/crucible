"""
Illiquidity-Premium x Trend Crisis-Alpha — BREADTH-RESTORED DEPLOYABLE-SLEEVE VARIANT
======================================================================================
Two-premium book, frozen / pre-registered, single spec (no parameter search):

  LEG A (alpha, 100% book risk): the VALIDATED Amihud illiquidity premium reconstructed
        on a survivorship-clean small-cap, sector-spread universe (Sharadar SEP/TICKERS).
        Dollar-neutral long(illiquid)/short(liquid), tranched to the extreme +/-30% of the
        illiquidity z, inverse-vol sized, weekly rebalanced. Pro-cyclical: bleeds when
        liquidity evaporates.

  LEG B (crisis-alpha overlay, 25% book risk): the canonical time-series-trend RULE
        (12-month TSMOM, inverse-vol, weekly) mapped onto a FIXED, MECHANICALLY chosen
        ~9-ETF cross-asset sleeve spanning the parent 21-futures taxonomy — one most-liquid
        ETF per a-priori bucket: SPY(US eq) EFA(intl-dev eq) EEM(EM eq) TLT(long dur)
        IEF(int dur) HYG(credit) GLD(gold) DBC(broad cmdty) VNQ(real estate). The instrument
        list is written down BEFORE any backtest — no per-instrument optimisation, no
        selection on returns. The ONLY new degree of freedom vs the 5-ETF parent is BREADTH.

  SIZING (inherited, unchanged): both legs scaled to equal trailing-60d vol, trend then
  down-weighted to 25% of book risk (tail overlay, NOT 50/50). With 9 instruments the
  sleeve's intra-leg diversification lowers its standalone vol so the same 25% risk budget
  rides more markets at the same gross.

Costs: 8 bps on turnover (modeled inside net_of_cost). All signals lagged 1 day (the single
authoritative lag is W.shift(1) before net_of_cost / trades_from_weights — stated here).
NO look-ahead. scope='local' (both legs' standalone edges are settled; the only new claims
are book-level complementarity + survival on a tradable breadth-restored sleeve).
No external side effects: pure compute, no writes / capital / config.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# --------------------------------------------------------------------------- constants
SPEC_ID = "amihud_illiq_x_trend_9etf_breadth_v1"
START   = "2005-01-01"           # HYG (~2007-04) binds the full-9 common window -> trend
                                 # active ~2008-04, covers GFC / 2011 / 2015-16 / 2020 / 2022
TOP_N_PER_SECTOR = 50            # ~500-600 small-cap names, sector-spread, survivorship-clean

ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD", "DBC", "VNQ"]
ETF_SECTORS = {
    "SPY": "TR_USEquity",   "EFA": "TR_IntlDevEquity", "EEM": "TR_EMEquity",
    "TLT": "TR_LongDuration", "IEF": "TR_IntDuration",  "HYG": "TR_Credit",
    "GLD": "TR_Gold",       "DBC": "TR_BroadCommodity", "VNQ": "TR_RealEstate",
}

DEFAULTS = dict(
    amihud_lb=126,        # ~6m Amihud (|ret|/$vol) averaging window
    vol_lb=63,            # ~3m vol for inverse-vol sizing (both legs)
    tranche=0.30,         # trade only the extreme top/bottom 30% of illiquidity z
    trend_lb=252,         # canonical 12m time-series-momentum lookback
    target_vol=0.10,      # 10% annualised per-leg vol target
    scale_lb=60,          # trailing-60d vol matching (per pre-registration)
    scale_cap=3.0,        # cap on per-leg vol-scale (no exploding leverage)
    trend_risk_frac=0.25, # trend sleeve at 25% of book risk (tail overlay)
)

# in-process cache so load_data() and signal() share the SAME survivorship-clean universe
_CACHE = {}


def _universe():
    if "uni" not in _CACHE:
        tickers, smap = sector_universe(marketcap="Small", top_n_per_sector=TOP_N_PER_SECTOR)
        _CACHE["uni"] = (list(tickers), dict(smap))
    return _CACHE["uni"]


def _sector_map_full():
    _, smap = _universe()
    out = dict(smap)
    out.update(ETF_SECTORS)
    return out


# --------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Panel signal() consumes: MultiIndex columns (block, ticker) with blocks
       'eq_px' (closeadj, for returns), 'eq_dv' (daily $ volume, for Amihud),
       'etf_px' (9-ETF adjusted close)."""
    tickers, _ = _universe()
    eq_px    = sep_panel(tickers, START, field="closeadj")        # split+div adjusted -> returns
    eq_close = sep_panel(tickers, START, field="close")           # unadjusted px for $volume
    eq_vol   = sep_panel(tickers, START, field="volume")          # raw shares
    eq_dv    = eq_close.reindex_like(eq_px) * eq_vol.reindex_like(eq_px)   # dollar volume
    etf_px   = yf_panel(ETFS, START).reindex(columns=ETFS)        # free, ETFs only (not US single stk)
    panel = pd.concat({"eq_px": eq_px, "eq_dv": eq_dv, "etf_px": etf_px}, axis=1)
    return panel


# --------------------------------------------------------------------------- core book
def _book(panel, p):
    """Returns (Wa_scaled, Wt_scaled, rets_total) — SAME-DAY weights for each leg, already
       vol-matched and risk-budgeted; caller applies the single .shift(1) lag."""
    eq_px = panel["eq_px"].dropna(how="all", axis=1)
    eq_dv = panel["eq_dv"].reindex_like(eq_px)
    etf_px = panel["etf_px"].reindex(columns=ETFS)

    # ---- LEG A: Amihud illiquidity premium (dollar-neutral, tranched, inverse-vol) ----
    eq_ret = eq_px.pct_change()
    illiq_daily = eq_ret.abs() / eq_dv.where(eq_dv > 0)
    amihud = illiq_daily.rolling(p["amihud_lb"], min_periods=max(20, p["amihud_lb"] // 2)).mean()
    sig = np.log(amihud.where(amihud > 0))                  # tame the heavy right tail
    z = xs_zscore(sig)                                      # high z = illiquid (long side)
    vol = eq_ret.rolling(p["vol_lb"]).std()
    raw = z / vol.where(vol > 0)                            # inverse-vol sizing
    rnk = z.rank(axis=1, pct=True)                          # tranche on extreme illiquidity only
    held = (rnk >= 1.0 - p["tranche"]) | (rnk <= p["tranche"])
    raw = raw.where(held)
    raw = raw.sub(raw.mean(axis=1), axis=0)                 # dollar-neutral within held set
    raw_w = raw.resample("W-FRI").last().reindex(raw.index, method="ffill")   # weekly rebalance
    Wa = raw_w.div(raw_w.abs().sum(axis=1), axis=0).fillna(0.0)               # gross ~1
    ra = (Wa.shift(1) * eq_ret).sum(axis=1)                 # leg-A daily return (for vol match)

    # ---- LEG B: canonical TSMOM trend rule on the FIXED 9-ETF sleeve ----
    etf_ret = etf_px.pct_change()
    mom = etf_px / etf_px.shift(p["trend_lb"]) - 1.0
    tsig = np.sign(mom)                                     # canonical 12m sign rule, long/short
    evol = etf_ret.rolling(p["vol_lb"]).std()
    traw = tsig / evol.where(evol > 0)                      # inverse-vol within sleeve
    traw_w = traw.resample("W-FRI").last().reindex(traw.index, method="ffill")
    Wt = traw_w.div(traw_w.abs().sum(axis=1), axis=0).fillna(0.0)             # gross ~1
    rt = (Wt.shift(1) * etf_ret).sum(axis=1)               # leg-B daily return (for vol match)

    # ---- equal trailing-60d vol match, trend down-weighted to 25% of book risk ----
    tv = p["target_vol"] / np.sqrt(252.0)
    av = ra.rolling(p["scale_lb"]).std()
    tvv = rt.rolling(p["scale_lb"]).std()
    a_scale = (tv / av.where(av > 0)).clip(upper=p["scale_cap"]).fillna(0.0)
    t_scale = (tv / tvv.where(tvv > 0)).clip(upper=p["scale_cap"]).fillna(0.0) * p["trend_risk_frac"]
    Wa_s = Wa.mul(a_scale, axis=0)
    Wt_s = Wt.mul(t_scale, axis=0)

    rets_total = pd.concat([eq_ret, etf_ret], axis=1)
    idx = rets_total.index
    Wa_s = Wa_s.reindex(idx).fillna(0.0)
    Wt_s = Wt_s.reindex(idx).fillna(0.0)
    rets_total = rets_total.fillna(0.0)
    return Wa_s, Wt_s, rets_total


def signal(panel, **params):
    """Combined two-premium book. mode in {'combined'(default),'amihud','trend'} selects the
       book for diagnostics. Returns (daily net-of-cost returns, contract trade ledger)."""
    p = {**DEFAULTS, **params}
    mode = params.get("mode", "combined")
    Wa, Wt, rets = _book(panel, p)

    if mode == "amihud":
        W = Wa
    elif mode == "trend":
        W = Wt
    else:
        W = Wa.add(Wt, fill_value=0.0)

    W = W.reindex(columns=rets.columns).fillna(0.0)
    Wlag = W.shift(1).fillna(0.0)                           # <-- the single authoritative 1-day lag

    daily = net_of_cost(Wlag, rets, cost_bps=8.0, name=SPEC_ID)
    daily = daily.dropna()
    if daily.ne(0.0).any():                                 # trim leading warmup (all-zero) days
        daily = daily.loc[daily.ne(0.0).idxmax():]

    smap = _sector_map_full()
    for c in W.columns:                                    # defensive: every held name has a sector
        smap.setdefault(str(c), "Unknown")
    trades = trades_from_weights(Wlag, rets, smap)          # run-length ledger + entry_regime stamp
    return daily, trades


# scope='local' — stage-2 generalization battery not run; defined for API completeness only.
def load_gen_data(label) -> pd.DataFrame:
    return pd.DataFrame()


# --------------------------------------------------------------------------- expectation helpers
def _max_dd(r):
    r = r.fillna(0.0)
    eq = (1.0 + r).cumprod()
    return float(-(eq / eq.cummax() - 1.0).min())          # positive magnitude


def _sharpe(r):
    r = r.dropna()
    s = r.std()
    if len(r) < 60 or s == 0:
        return 0.0
    return float(r.mean() / s * np.sqrt(252.0))


def _pre(s, hs):
    return s[s.index < pd.Timestamp(hs)].dropna()


def _book_returns(panel):
    """Recompute (combined, amihud-only, trend-only) NET returns from the in-memory panel
       (arithmetic only, no new data). Checks slice these to < holdout_start."""
    Wa, Wt, rets = _book(panel, DEFAULTS)
    cols = rets.columns
    Wa = Wa.reindex(columns=cols).fillna(0.0)
    Wt = Wt.reindex(columns=cols).fillna(0.0)
    comb = net_of_cost((Wa + Wt).shift(1).fillna(0.0), rets, cost_bps=8.0, name="comb")
    amid = net_of_cost(Wa.shift(1).fillna(0.0), rets, cost_bps=8.0, name="amid")
    tren = net_of_cost(Wt.shift(1).fillna(0.0), rets, cost_bps=8.0, name="tren")
    return comb, amid, tren


def chk_maxdd(ctx):
    comb, amid, _ = _book_returns(ctx["panel"])
    hs = ctx["holdout_start"]
    df = pd.concat([_pre(comb, hs).rename("c"), _pre(amid, hs).rename("a")], axis=1).dropna()
    if len(df) < 250:
        return {"pass": False, "observed": "insufficient_overlap"}
    red = 1.0 - _max_dd(df["c"]) / max(_max_dd(df["a"]), 1e-9)
    return {"pass": bool(red >= 0.20), "observed": round(float(red), 3)}


def chk_sharpe(ctx):
    comb, amid, _ = _book_returns(ctx["panel"])
    hs = ctx["holdout_start"]
    df = pd.concat([_pre(comb, hs).rename("c"), _pre(amid, hs).rename("a")], axis=1).dropna()
    if len(df) < 250:
        return {"pass": False, "observed": "insufficient_overlap"}
    sc, sa = _sharpe(df["c"]), _sharpe(df["a"])
    if sa <= 0:
        return {"pass": bool(sc >= sa), "observed": round(float(sc - sa), 3)}
    deg = (sa - sc) / sa
    return {"pass": bool(deg <= 0.10), "observed": round(float(deg), 3)}


def chk_legcorr(ctx):
    _, amid, tren = _book_returns(ctx["panel"])
    hs = ctx["holdout_start"]
    df = pd.concat([_pre(amid, hs).rename("a"), _pre(tren, hs).rename("t")], axis=1).dropna()
    if len(df) < 250:
        return {"pass": False, "observed": "insufficient_overlap"}
    c = float(df["a"].corr(df["t"]))
    return {"pass": bool(c <= 0.10), "observed": round(c, 3)}


def chk_tracking(ctx):
    _, _, tren = _book_returns(ctx["panel"])
    hs = ctx["holdout_start"]
    try:
        tr, _ = trend_returns()                            # validated 21-market CTA stream (diagnostic)
    except Exception:
        return {"pass": False, "observed": "trend_returns_unavailable"}
    df = pd.concat([_pre(tren, hs).rename("s"), tr.rename("v")], axis=1).dropna()
    df = df[df.index < pd.Timestamp(hs)]
    if len(df) < 250:
        return {"pass": False, "observed": "insufficient_overlap"}
    c = float(df["s"].corr(df["v"]))
    return {"pass": bool(c >= 0.60), "observed": round(c, 3)}


# --------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id=SPEC_ID,
    family="illiquidity_x_trend_combination",
    title=("Illiquidity-Premium x Trend Crisis-Alpha — breadth-restored deployable-sleeve "
           "(frozen Amihud tranched leg + canonical trend rule on a mechanical 9-ETF "
           "cross-asset sleeve, $5K-tradable)"),
    markets=["US small-cap equities (Amihud illiquidity)",
             "cross-asset ETF trend sleeve (SPY/EFA/EEM/TLT/IEF/HYG/GLD/DBC/VNQ)"],
    data_desc=("Sharadar SEP/TICKERS (owned, survivorship-clean, delisted incl) for the small-cap "
               "Amihud leg: closeadj for returns, close*volume for daily dollar-volume. yfinance "
               "(free) adjusted closes for the 9 ETFs. trend_returns() retained only as a "
               "non-selectable tracking diagnostic. $0 incremental data."),
    pre_registration=(
        "FROZEN / PRE-REGISTERED two-premium book; NO parameter search (effective-N = 1; grid is "
        "{'default':{}} only). LEG A: Amihud illiquidity premium on a small-cap sector-spread "
        "universe (sector_universe(marketcap='Small', top_n_per_sector=50)). Amihud = trailing-126d "
        "mean of |ret|/$volume; log; cross-sectional winsorized z (xs_zscore); inverse-vol sized; "
        "tranched to the extreme top/bottom 30% of the z; dollar-neutral long(illiquid)/short(liquid); "
        "weekly (W-FRI) rebalance. LEG B: the canonical 12-month time-series-momentum SIGN rule "
        "(unchanged params), inverse-vol weighted, weekly, mapped onto a FIXED 9-ETF list chosen "
        "MECHANICALLY before any backtest — one most-liquid / largest-AUM ETF per a-priori bucket of "
        "the parent 21-futures taxonomy: SPY,EFA,EEM (equity tiers), TLT,IEF (duration), HYG (credit), "
        "GLD (gold), DBC (broad commodity), VNQ (real estate). No per-instrument or weighting "
        "optimisation; the ONLY mutation vs the 5-ETF parent is BREADTH. SIZING (inherited, unchanged): "
        "each leg scaled to equal trailing-60d vol (target 10% ann, scale cap 3x), trend then "
        "down-weighted to 25% of book risk (tail overlay, not 50/50). Costs 8 bps on turnover. Single "
        "1-day lag = W.shift(1) before net_of_cost / trades_from_weights. Holdout = 2022-01-01 "
        "(GFC/2011/2015-16/2020 in search; 2022 stress OOS). PRE-REGISTERED SUCCESS (vs standalone "
        "Amihud over the identical pre-holdout overlap, all NET of costs): (1) combined MaxDD reduced "
        ">=20%; (2) Sharpe degradation <=10%; (3) Amihud-vs-trend leg correlation <= +0.1; (4) 9-ETF "
        "sleeve tracking-correlation to the validated trend_returns() stream >= 0.6 (HIGHER than the "
        "5-ETF >=0.5 floor — the breadth thesis). All four are MACHINE-CHECKED in `expectations` "
        "(each recomputes legs from the in-memory panel, no extra data, sliced to < holdout). Gate "
        "stack (market-neutral MCPT on the Amihud panel, write-once holdout, DSR) runs unchanged. "
        "REPORTED-NOT-GATED diagnostics: per-bucket stress-window contribution sign (incl. 2008 GFC, "
        "now in-window via HYG ~2007 inception), and a 5-ETF-vs-9-ETF breadth-ablation of crisis-window "
        "sleeve contribution. scope='local': both standalone edges are settled (Amihud 3/3 gen "
        "universes; trend 3/3 stress regimes); the only NEW claims are book-level complementarity and "
        "its survival on a tradable breadth-restored sleeve, which the breadth restoration itself "
        "hardens by moving the deployable instrument set materially closer to the validated 21-market "
        "taxonomy. Falsification: if the 9-ETF book fails the gate / its expectations, the conclusion "
        "is that no tradable ETF breadth recovers the futures-stream convexity at this scale."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={"default": {}},                                  # honest: no parameter search performed
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "maxdd_reduction_ge_20pct",
         "claim": "combined-book MaxDD <= 0.80x standalone-Amihud MaxDD (>=20% tail cut) on pre-holdout overlap",
         "check": chk_maxdd},
        {"name": "sharpe_degradation_le_10pct",
         "claim": "combined Sharpe degraded <=10% vs standalone Amihud on pre-holdout overlap",
         "check": chk_sharpe},
        {"name": "leg_correlation_le_0.1",
         "claim": "Amihud leg vs trend sleeve daily-return correlation <= +0.1 (complementary premia)",
         "check": chk_legcorr},
        {"name": "sleeve_tracks_trend_ge_0.6",
         "claim": "9-ETF trend sleeve tracks validated 21-market trend_returns() at corr >= 0.6 (breadth restored)",
         "check": chk_tracking},
    ],
)