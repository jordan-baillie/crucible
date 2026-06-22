"""
Net-Share-Issuance Premium (US equity, cross-sectional, dollar-neutral)
=======================================================================
LONG buyback firms (shrinking share count), SHORT issuers (growing share count).

PREMIUM: the net-issuance / external-financing anomaly (Pontiff-Woodgate,
Daniel-Titman). Firms that issue equity systematically underperform; firms that
retire shares outperform — a robust, cross-market mispricing concentrated in
small/illiquid names where arbitrage is costly. Because the mechanism is a
UNIVERSAL corporate-financing premium (not a single-universe quirk), scope='broad':
a stage-1 pass MUST generalise to untouched (mid-cap) sector slices.

NO external side effects: only OWNED Sharadar reads (sep_panel / sf1, point-in-time
via pit_panel on the filing datekey — never calendardate) + pure compute. No network.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd

# ---------------------------------------------------------------- constants ---
ID = "equity_net_share_issuance_v1"
SEARCH_START = "2010-01-01"
HOLDOUT = "2020-01-01"
SMALL_PER = 55                # ~600 small-cap names (anomaly lives in small/illiquid)
MID_PER = 45                  # mid-cap pool for the disjoint generalization slices

DEFAULTS = dict(
    iss_lb=252,              # 1y change in shares outstanding (issuance window)
    vol_lb=60,              # realized-vol lookback for inverse-vol sizing
    q=0.20,                # top/bottom quantile each leg
    cost_bps=8.0,          # ~8bps on turnover (round-trip on weekly rebalance)
)

# generalization slices: MID-cap (disjoint from the SMALL-cap search by tier),
# split into 3 distinct sector groups so each shares NO tickers with the search.
_GEN_GROUPS = {
    "mid_cyclical":  {"Industrials", "Consumer Cyclical", "Basic Materials", "Energy"},
    "mid_defensive": {"Healthcare", "Consumer Defensive", "Utilities", "Real Estate"},
    "mid_growth":    {"Technology", "Financial Services", "Communication Services"},
}

_SECTOR_MAP = {}             # global fallback ticker->sector (populated on every build)
_MID_CACHE = {}


# ------------------------------------------------------------- panel builds ---
def _build(tickers, smap):
    """price (split+div adj) + point-in-time shares-outstanding panel."""
    tickers = sorted(set(tickers))
    px = sep_panel(tickers, SEARCH_START)
    px = px.loc[:, [c for c in px.columns if c in set(tickers)]].sort_index()
    keep = list(px.columns)
    sf = sf1(keep, ["sharesbas"], dimension="ARQ")
    shares = pit_panel(sf, "sharesbas", px.index, keep)        # datekey-based, ffilled
    have = [c for c in keep if c in shares.columns and shares[c].notna().any()]
    px, shares = px[have], shares[have]
    panel = pd.concat({"price": px, "shares": shares}, axis=1)
    panel.columns = panel.columns.set_names(["field", "ticker"])
    final = {t: smap.get(t, "unknown") for t in have}
    panel.attrs["sector_map"] = final          # survives harness pass-through
    _SECTOR_MAP.update(final)                   # belt-and-suspenders fallback
    return panel


def load_data() -> pd.DataFrame:
    tickers, smap = sector_universe("Small", SMALL_PER)
    return _build(tickers, smap)


def _mid_universe():
    if not _MID_CACHE:
        t, sm = sector_universe("Mid", MID_PER)
        _MID_CACHE["t"], _MID_CACHE["sm"] = t, sm
    return _MID_CACHE["t"], _MID_CACHE["sm"]


def load_gen_data(label) -> pd.DataFrame:
    """ONE generalization universe (mid-cap sector slice; disjoint from search)."""
    sectors = _GEN_GROUPS[label]
    mids, smap = _mid_universe()
    sel = [t for t in mids if smap.get(t) in sectors]
    return _build(sel, {t: smap[t] for t in sel})


# ----------------------------------------------------------------- signal -----
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    px = panel["price"]
    shares = panel["shares"]
    cols = list(px.columns)
    rets = px.pct_change()

    # --- net-issuance signal: LONG share-shrinkers, SHORT issuers -------------
    iss = shares / shares.shift(p["iss_lb"]) - 1.0          # 1y change in share count
    sig = -iss                                              # high = buyback (long)
    z = xs_zscore(sig)                                      # X-sectional winsor z (kit)

    # --- inverse-realized-vol sizing (causal trailing vol) -------------------
    vol = rets.rolling(p["vol_lb"], min_periods=max(20, p["vol_lb"] // 2)).std()
    inv_vol = 1.0 / vol.clip(lower=0.005)

    # --- top/bottom quantile legs, equal-risk, dollar-neutral (gross=1) -------
    rk = z.rank(axis=1, pct=True)
    hi = rk >= (1.0 - p["q"])                               # long leg (buybacks)
    lo = rk <= p["q"]                                       # short leg (issuers)
    lw = inv_vol.where(hi)
    sw = inv_vol.where(lo)
    lw = lw.div(lw.sum(axis=1), axis=0)
    sw = sw.div(sw.sum(axis=1), axis=0)
    W_target = (0.5 * lw.fillna(0.0)) - (0.5 * sw.fillna(0.0))

    # --- weekly rebalance: hold weights between the first trading day of weeks -
    idx = W_target.index
    periods = pd.Series(pd.PeriodIndex(idx, freq="W"), index=idx)
    reb = ~periods.duplicated()
    W = W_target.copy()
    W.loc[~reb.values] = np.nan
    W = W.ffill().fillna(0.0)

    # weights decided end-of-day t -> held into t+1: the lag is OUR responsibility
    Wl = W.shift(1).fillna(0.0)

    smap = panel.attrs.get("sector_map") or {}
    sector_map = {c: smap.get(c, _SECTOR_MAP.get(c, "unknown")) for c in cols}

    net = net_of_cost(Wl, rets, cost_bps=p["cost_bps"], name=ID)
    trades = trades_from_weights(Wl, rets, sector_map)      # auto-stamps entry_regime
    return net, trades


# ---------------------------------------------------- soft expectation checks ---
def _check_hold(ctx):
    tr = ctx.get("trades") or []
    if not tr:
        return {"pass": False, "observed": 0}
    med = float(np.median([t["hold_days"] for t in tr]))
    return {"pass": med >= 5.0, "observed": round(med, 2)}


def _check_market_neutral(ctx):
    r = ctx["search"].dropna()
    mkt = ctx["panel"]["price"].pct_change().mean(axis=1)
    mkt = mkt[mkt.index < ctx["holdout_start"]]
    df = pd.concat([r, mkt], axis=1).dropna()
    if len(df) < 60:
        return {"pass": False, "observed": "insufficient"}
    c = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    beta = float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else 0.0
    return {"pass": abs(beta) < 0.3, "observed": round(beta, 3)}


def _check_subperiods(ctx):
    r = ctx["search"].dropna()
    if len(r) < 120:
        return {"pass": False, "observed": "insufficient"}
    h = len(r) // 2
    a, b = float(r.iloc[:h].mean()), float(r.iloc[h:].mean())
    return {"pass": (a > 0) and (b > 0), "observed": f"h1={a:.5f},h2={b:.5f}"}


def _check_issuance_dispersion(ctx):
    sh = ctx["panel"]["shares"]
    sh = sh[sh.index < ctx["holdout_start"]]
    iss = sh / sh.shift(252) - 1.0
    disp = float(iss.std(axis=1).median())
    return {"pass": disp > 0.01, "observed": round(disp, 4)}


# ----------------------------------------------------------------- the spec ---
SPEC = StrategySpec(
    id=ID,
    family="equity_issuance",
    title="Net-Share-Issuance Premium (US equity, X-sectional, dollar-neutral)",
    markets=["us_equity"],
    data_desc=("OWNED Sharadar: split+div-adjusted daily close (sep_panel) + point-in-time "
               "shares-outstanding (sf1 'sharesbas', ARQ, via pit_panel on filing datekey). "
               "Search = sector-spread SMALL-cap (sector_universe('Small',55), ~600 names) "
               "where the anomaly is least arbitraged. No network; survivorship-clean "
               "(delisted included)."),
    pre_registration=(
        "PREMIUM: the net-share-issuance / external-financing anomaly. Firms that grow their "
        "share count (issuance, secondary offerings) underperform; firms that shrink it "
        "(buybacks) outperform — a robust corporate-financing mispricing concentrated in "
        "small/illiquid names with high arbitrage costs.\n"
        "MECHANISM: signal = -(1y change in shares outstanding), point-in-time on the filing "
        "datekey (never calendardate). Cross-sectional winsorized z; LONG the top quantile "
        "(buybacks), SHORT the bottom (issuers); inverse-realized-vol equal-risk legs; "
        "dollar-neutral gross=1; WEEKLY rebalance; ~8bps turnover cost; signals lagged 1 day.\n"
        "SINGLE PRIMARY CONFIG (no cherry-pick): iss_lb=252, vol_lb=60, q=0.20.\n"
        "SCOPE broad: this is a UNIVERSAL premium, so a stage-1 pass must GENERALISE. Stage-2 "
        "battery runs the frozen signal+default params on the HOLDOUT of 3 MID-cap sector "
        "slices (cyclical / defensive / growth) — each disjoint from the small-cap search by "
        "cap tier; >=60% must be OOS-positive or the candidate is an overfit outlier.\n"
        "FALSIFIABLE (machine-checked): (1) median hold >= 5d (weekly rebalance keeps names); "
        "(2) search book beta to the equal-weight universe within +-0.3 (dollar-neutral); "
        "(3) positive return in BOTH search-period halves (not one regime); (4) real "
        "cross-sectional dispersion in 1y issuance (the signal exists).\n"
        "FALSIFIED IF: holdout Sharpe<=0, fails to generalise to >=60% of the mid-cap slices, "
        "edge confined to one sub-period, or no issuance dispersion."),
    load_data=load_data,
    signal=signal,
    default_params=DEFAULTS,
    grid={
        "default": {},
        "iss_lb_126": {"iss_lb": 126},
        "q15": {"q": 0.15},
        "q25": {"q": 0.25},
        "vol_lb_120": {"vol_lb": 120},
    },
    scope="broad",
    generalization_universes=["mid_cyclical", "mid_defensive", "mid_growth"],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=40,
    expectations=[
        {"name": "median_hold_multiweek",
         "claim": "median trade hold_days >= 5 (weekly rebalance + persistent signal)",
         "check": _check_hold},
        {"name": "market_neutral",
         "claim": "search book beta to equal-weight universe within +-0.3 (dollar-neutral)",
         "check": _check_market_neutral},
        {"name": "robust_subperiods",
         "claim": "search return positive in BOTH halves of the search window",
         "check": _check_subperiods},
        {"name": "issuance_dispersion",
         "claim": "median cross-sectional std of 1y issuance > 0.01 (real signal)",
         "check": _check_issuance_dispersion},
    ],
)