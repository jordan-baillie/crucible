"""
Earnings-announcement risk premium (Frazzini–Lamont style), market-beta-hedged.

THESIS (event-time RISK premium, NOT a surprise forecast / not PEAD):
  You are paid to hold idiosyncratic earnings-uncertainty risk ACROSS a firm's
  scheduled reporting window. Each calendar month we go long an EQUAL-WEIGHTED basket
  of every name PREDICTED to report that month, and short a broad index ETF sized to
  the basket's trailing beta. The residual market-neutral spread = the premium.

NO EXTERNAL CALENDAR: the expected reporting month is reconstructed point-in-time from
each firm's OWN historical Sharadar SF1 `datekey` (SEC filing date) cadence — firms report
on a near-fixed quarterly cadence, so the calendar month a firm filed a given fiscal quarter
12 months ago predicts this year's month. Prediction at month P uses ONLY filings from
month P-12 and earlier => strictly backward-looking, no look-ahead.

The only novel code here is the announcement-month predictor + the beta hedge; sizing,
costing, the trade ledger (and its regime stamping) all go through the mandated kit.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START   = "2000-01-01"
HEDGE   = "SPY"                      # broad-market beta hedge (whitelisted sleeve)
SPEC_ID = "earn_announce_premium"
_LAST_SM = {}                        # sector_map fallback if DataFrame.attrs is dropped


# ----------------------------------------------------------------------------- universes
# Search universe = liquid MID caps (where the higher-uncertainty premium is expected to live
# yet stays tradable). Generalization universes are DISJOINT cap-tier / liquidity slices.
def _univ(kind):
    if   kind == "search":     tks, sm = sector_universe("Mid",   30)   # ~330 names
    elif kind == "large_cap":  tks, sm = sector_universe("Large", 30)
    elif kind == "small_cap":  tks, sm = sector_universe("Small", 30)
    elif kind == "mid_deep":   tks, sm = sector_universe("Mid",   65)   # ranks 31-65 after diff
    else: raise ValueError(kind)
    if kind != "search":                                                # guarantee disjoint
        s = set(_search_set())
        tks = [t for t in tks if t not in s]
        sm  = {t: sm[t] for t in tks}
    return tks, sm


def _search_set():
    if not _LAST_SM.get("_search_tks"):
        tks, _ = sector_universe("Mid", 30)
        _LAST_SM["_search_tks"] = tks
    return _LAST_SM["_search_tks"]


# ----------------------------------------------------------------------------- panels
def _announce_panel(tickers, px):
    """Daily boolean panel: is `ticker` PREDICTED to report in this calendar month?
    Predictor: filed in the same calendar month 12 months ago AND >=8 prior filings.
    Uses only filings dated before the start of the predicted month -> no look-ahead."""
    cols = list(px.columns)
    sf = sf1(tickers, ["revenue"], dimension="ARQ")
    if not {"ticker", "datekey"}.issubset(getattr(sf, "columns", [])):
        sf = sf.reset_index()
    sf = sf[["ticker", "datekey"]].dropna()
    sf["datekey"] = pd.to_datetime(sf["datekey"])

    months = pd.period_range(px.index.min(), px.index.max(), freq="M")
    am = pd.DataFrame(False, index=months, columns=cols)
    for tk, g in sf.groupby("ticker"):
        if tk not in am.columns:
            continue
        dks = np.sort(g["datekey"].dropna().unique())
        if dks.size < 8:                                   # need >=8 quarters of cadence
            continue
        d8 = pd.Timestamp(dks[7])                          # 8th filing -> history requirement
        fp = pd.PeriodIndex(pd.to_datetime(dks), freq="M")
        pred = (fp + 12).unique()                          # same month, next year
        pred = pred[pred.to_timestamp() > d8]              # only after >=8 filings exist
        pred = pred.intersection(months)
        if len(pred):
            am.loc[pred, tk] = True

    announce = am.reindex(px.index.to_period("M"))         # months -> daily
    announce.index = px.index
    return announce.astype(bool)


def _panel_for(tickers, sector_map):
    px = sep_panel(tickers, START, field="closeadj").dropna(how="all", axis=1)
    tickers = list(px.columns)
    announce = _announce_panel(tickers, px)

    h = yf_panel([HEDGE], START)
    h = h.to_frame() if isinstance(h, pd.Series) else h
    h = h.reindex(px.index).ffill()
    h.columns = [HEDGE]

    sm = {t: sector_map.get(t, "Unknown") for t in tickers}
    sm[HEDGE] = "Hedge"
    panel = pd.concat({"price": px, "announce": announce, "hedge": h}, axis=1)
    panel.attrs["sector_map"] = sm
    _LAST_SM["sm"] = sm
    return panel


def load_data() -> pd.DataFrame:
    return _panel_for(*_univ("search"))


def load_gen_data(label) -> pd.DataFrame:
    return _panel_for(*_univ(label))


# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    beta_lb  = int(params.get("beta_lb",  126))
    name_cap = float(params.get("name_cap", 0.05))
    cost_bps = float(params.get("cost_bps", 10.0))

    px  = panel["price"].astype(float)
    ann = panel["announce"].reindex(columns=px.columns).fillna(False).astype(bool)
    hpx = panel["hedge"]
    sector_map = panel.attrs.get("sector_map") or _LAST_SM.get("sm") or {}

    rets = px.pct_change()

    # --- EQUAL-WEIGHT LONG over predicted announcers (gross = 1) ---
    annf   = ann.astype(float)
    W_long = annf.div(annf.sum(axis=1).replace(0.0, np.nan), axis=0)
    W_long = W_long.clip(upper=name_cap)                                  # per-name cap
    W_long = W_long.div(W_long.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # --- monthly rebalance / hysteresis: hold month-start weights ---
    mp = px.index.to_period("M")
    first_of_month = (~pd.Series(mp, index=px.index).duplicated()).values
    held = np.where(first_of_month[:, None], W_long.values, np.nan)
    W_target = pd.DataFrame(held, index=px.index, columns=px.columns).ffill().fillna(0.0)
    gross_long = W_target.sum(axis=1)

    # --- market-beta hedge via the declared index-ETF sleeve ---
    spy_ret = hpx.iloc[:, 0].pct_change().reindex(px.index)
    basket  = (W_target.shift(1) * rets).sum(axis=1)                      # realized (lagged)
    cov = basket.rolling(beta_lb, min_periods=beta_lb // 2).cov(spy_ret)
    var = spy_ret.rolling(beta_lb, min_periods=beta_lb // 2).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    beta = beta.clip(lower=0.4, upper=1.1).ffill().fillna(0.8)            # bound -> cap hedge share
    hedge_w = -(beta * gross_long)                                        # 0 when not invested

    # --- assemble alpha + sleeve, then lag ONCE (weights decided at t earn t+1) ---
    rets_full = rets.copy()
    rets_full[HEDGE] = spy_ret
    W_full = W_target.reindex(columns=rets_full.columns).fillna(0.0)
    W_full[HEDGE] = hedge_w.reindex(W_full.index).values
    W_held = W_full.shift(1)

    daily  = net_of_cost(W_held, rets_full, cost_bps=cost_bps, name=SPEC_ID)
    trades = trades_from_weights(W_held, rets_full, sector_map)           # kit stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------------------- soft expectations
def _exp_announcer_vs_control(ctx):
    """Isolate the EVENT: announcers should out-earn matched non-announcers in the same months."""
    try:
        panel = ctx["panel"]; hs = pd.Timestamp(ctx["holdout_start"])
        px  = panel["price"].astype(float)
        ann = panel["announce"].reindex(columns=px.columns).fillna(False).astype(bool)
        rets = px.pct_change()
        rets = rets[px.index < hs]
        annL = ann.shift(1).reindex(rets.index).fillna(False)
        a = rets.where(annL).mean(axis=1)
        c = rets.where(~annL).mean(axis=1)
        diff = float((a - c).mean())
        return {"pass": bool(diff > 0), "observed": diff}
    except Exception as e:
        return {"pass": False, "observed": str(e)}


def _exp_breadth(ctx):
    """Long basket must be diversified, not 1-2 names: median monthly announcer count >= 15."""
    try:
        ann = ctx["panel"]["announce"].fillna(False).astype(bool)
        ann = ann[ann.index < pd.Timestamp(ctx["holdout_start"])]
        monthly = ann.groupby(ann.index.to_period("M")).max().sum(axis=1)
        med = float(monthly.median())
        return {"pass": bool(med >= 15), "observed": med}
    except Exception as e:
        return {"pass": False, "observed": str(e)}


def _exp_market_neutral(ctx):
    """The ETF hedge should leave little residual market beta (|beta| < 0.4)."""
    try:
        r   = ctx["search"].dropna()
        spy = ctx["panel"]["hedge"].iloc[:, 0].pct_change()
        df  = pd.concat([r, spy], axis=1).dropna()
        df  = df[df.index < pd.Timestamp(ctx["holdout_start"])]
        b   = float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / np.var(df.iloc[:, 1]))
        return {"pass": bool(abs(b) < 0.4), "observed": b}
    except Exception as e:
        return {"pass": False, "observed": str(e)}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id=SPEC_ID,
    family="earnings_announcement_premium",
    title="Earnings-announcement risk premium (predicted-reporting-month basket, index-ETF beta-hedged)",
    markets=["us_equity"],
    data_desc="Sharadar SEP closeadj (survivorship-clean, delisted incl) + SF1 datekey filing cadence + SPY ETF hedge",
    pre_registration=(
        "Mechanism: a recurring EVENT-RISK premium for bearing idiosyncratic earnings uncertainty across a "
        "firm's scheduled reporting window (Frazzini-Lamont 2006); NOT a forecast of the surprise sign/size "
        "(distinct from PEAD/SUE) and NOT a characteristic sort (value/momentum/quality/low-vol/issuance/"
        "accruals) nor a liquidity rank (the Amihud parent). "
        "Construction: each calendar month, EQUAL-WEIGHTED LONG every name predicted to report that month, "
        "SHORT SPY sized to the basket's trailing beta -> market-neutral residual = the premium. "
        "Prediction is point-in-time from each firm's OWN SF1 `datekey` (SEC filing date proxy for the reporting "
        "window) at month granularity: announcer iff it filed the same calendar month 12 months ago AND has >=8 "
        "prior filings; only filings strictly before the predicted month are used -> no look-ahead. Weights are "
        "lagged one day before earning returns. Monthly rebalance with month-hold hysteresis. NET of 10bps/turnover "
        "(round-trip on liquid mid + ETF hedge). Frozen PRIMARY config; the grid declares the search burden. "
        "Predictions (gated): (1) announcers out-earn matched non-announcers in the SAME month (event isolation); "
        "(2) the ETF hedge removes market beta (|residual beta|<0.4); (3) the basket is diversified (median >=15 "
        "names/month). Scope=broad: a universal event mechanism must GENERALISE -> the frozen signal is run on the "
        "HOLDOUTS of disjoint cap/liquidity slices (large, small, deep-mid); a risk premium should appear across "
        "cap tiers (mid/small higher-uncertainty expected somewhat stronger). Decay & cost/turnover are the named "
        "risks and are pre-registered as hard gates. The continuous SPY short is the DECLARED hedge sleeve "
        "(whitelist + position-day cap), so the deployment gate judges the long alpha book alone. "
        "Not-machine-checked here (stated as prose): cross-cap-tier MONOTONICITY in uncertainty (needs the other "
        "universes, unavailable in a single check ctx) and robustness to +/- a few days of the window definition "
        "(a load_data-level reconstruction, not a signal param) — both are evaluated by the stage-2 battery."
    ),
    load_data=load_data,
    signal=signal,
    default_params={"beta_lb": 126, "name_cap": 0.05, "cost_bps": 10.0},
    grid={
        "default":  {},
        "beta_252": {"beta_lb": 252},
        "cap_03":   {"name_cap": 0.03},
        "cost_15":  {"cost_bps": 15.0},
    },
    scope="broad",
    generalization_universes=["large_cap", "small_cap", "mid_deep"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=50,
    hedge_tickers=[HEDGE],
    hedge_cap=0.55,
    expectations=[
        {"name": "announcer_beats_control",
         "claim": "search-window mean daily return of predicted-announcers > non-announcers (same months)",
         "check": _exp_announcer_vs_control},
        {"name": "hedge_market_neutral",
         "claim": "residual market beta of net returns |beta| < 0.4 after the index-ETF hedge",
         "check": _exp_market_neutral},
        {"name": "basket_breadth",
         "claim": "median monthly predicted-announcer count >= 15 (diversified, not 1-2 names)",
         "check": _exp_breadth},
    ],
)