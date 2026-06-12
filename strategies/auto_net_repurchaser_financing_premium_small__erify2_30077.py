"""
Net-Repurchaser Financing Premium — Small/Mid-Cap Long-Only Tilt (cost-hardened, low-turnover).

Mechanism (limits-to-arbitrage): net share repurchasers earn a persistent premium over net
issuers; the LONG-repurchaser side in small caps is compensation for concentrated idiosyncratic
risk where arbitrage capital is structurally absent. The liquid large-cap long-short version of
this anomaly already FAILED in this lab (net_share_issuance_factor, top-1000 most-liquid) — that
fail is the pre-registered falsification anchor, not a duplicate.

Construction is FROZEN per proposal: 12m % change in split-adjusted shares outstanding
(sharesbas*sharefactor, PIT via filing datekey), long-only equal-weight bottom-issuance-quintile
book (issuance <= 0 required to enter), $1M 21d-median-ADV floor, monthly rebalance with
hysteresis (enter bottom quintile, exit only on leaving bottom two quintiles), costs stressed at
30bps default (8/50bps grid variants). Search universe = even-alphabetical half of the small-cap
sector-spread universe; generalization universes are fully DISJOINT: the odd-alphabetical half
split into low-ADV and high-ADV terciles (ADV measured pre-holdout only) plus an untouched
mid-cap band.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, sf1
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights, pit_panel

START = "2003-01-01"
HOLDOUT_START = "2022-01-01"

_CACHE = {}


# ----------------------------------------------------------------------------- universes

def _universes():
    """Small + Mid sector-spread universes and a combined sector map (cached)."""
    if "smap" not in _CACHE:
        s_tk, s_map = sector_universe(marketcap="Small", top_n_per_sector=130)
        m_tk, m_map = sector_universe(marketcap="Mid", top_n_per_sector=35)
        smap = dict(m_map)
        smap.update(s_map)
        _CACHE["small_tk"] = sorted(s_tk)
        _CACHE["mid_tk"] = sorted(m_tk)
        _CACHE["smap"] = smap
    return _CACHE["small_tk"], _CACHE["mid_tk"], _CACHE["smap"]


def _build_panel(tickers):
    """Panel with MultiIndex columns: ('px', t) closeadj, ('dv', t) dollar volume,
    ('iss', t) trailing-12m % change in split-adjusted shares (PIT via datekey)."""
    px = sep_panel(tickers, START, field="closeadj")
    cl = sep_panel(tickers, START, field="close")
    vo = sep_panel(tickers, START, field="volume")
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    dv = (cl.reindex(columns=tickers) * vo.reindex(columns=tickers))

    fund = sf1(tickers, fields=["sharesbas", "sharefactor"], dimension="ARQ").copy()
    fund["shares_adj"] = fund["sharesbas"] * fund["sharefactor"]
    # pit_panel joins on FILING datekey (never calendardate) and ffills — no look-ahead.
    shares = pit_panel(fund, "shares_adj", px.index, tickers)
    iss = shares / shares.shift(252) - 1.0

    return pd.concat({"px": px, "dv": dv, "iss": iss}, axis=1).sort_index(axis=1)


def load_data():
    """Search universe = EVEN-alphabetical half of the small-cap universe (~700 names).
    The odd half is reserved untouched for generalization."""
    small_tk, _, _ = _universes()
    return _build_panel(small_tk[0::2])


def load_gen_data(label):
    small_tk, mid_tk, _ = _universes()
    odd = small_tk[1::2]
    if label == "midcap":
        small_set = set(small_tk)
        tk = [t for t in mid_tk if t not in small_set][:350]
        return _build_panel(tk)

    if "odd_panel" not in _CACHE:
        _CACHE["odd_panel"] = _build_panel(odd)
    panel = _CACHE["odd_panel"]
    # Classify ADV on PRE-HOLDOUT data only — no peeking into the gen holdout.
    med_adv = panel["dv"].loc[:HOLDOUT_START].median()
    ranks = med_adv.rank(pct=True)
    if label == "small_lowadv":
        tk = list(ranks[ranks <= 1.0 / 3.0].index)
    elif label == "small_highadv":
        tk = list(ranks[ranks > 2.0 / 3.0].index)
    else:
        raise ValueError(f"unknown gen universe: {label}")
    return pd.concat({k: panel[k][tk] for k in ("px", "dv", "iss")}, axis=1).sort_index(axis=1)


# ----------------------------------------------------------------------------- signal

def signal(panel, cost_bps=30.0, adv_floor=1_000_000.0, entry_pct=0.20, exit_pct=0.40,
           max_names=50, min_pool=30, weighting="equal"):
    px = panel["px"]
    dv = panel["dv"]
    iss = panel["iss"]
    rets = px.pct_change()
    adv = dv.rolling(21).median()

    idx = px.index
    rebal_dates = idx.to_series().groupby(idx.to_period("M")).max().values

    _, _, smap = _universes()
    sector_map = {t: smap.get(t, "Unknown") for t in px.columns}

    Wreb = pd.DataFrame(0.0, index=pd.DatetimeIndex(rebal_dates), columns=px.columns)
    holdings = []
    for d in rebal_dates:
        iss_d = iss.loc[d]
        elig = (adv.loc[d] > adv_floor) & iss_d.notna() & px.loc[d].notna()
        pool = iss_d[elig]
        if len(pool) == 0:
            holdings = []
            continue
        pct = pool.rank(pct=True)  # low rank = most negative issuance (repurchasers)
        # Hysteresis: incumbents stay while inside bottom TWO quintiles and still eligible.
        keep = [t for t in holdings if t in pct.index and pct[t] <= exit_pct]
        if len(pool) >= min_pool:
            cands = pct[(pct <= entry_pct) & (pool <= 0.0)].sort_values()
            new = [t for t in cands.index if t not in keep]
            holdings = keep + new[: max(0, max_names - len(keep))]
        else:
            holdings = keep
        if not holdings:
            continue
        if weighting == "inv_vol":
            vol = rets[holdings].loc[:d].tail(63).std()
            w = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
            w = w / w.sum() if w.sum() > 0 else w
        else:
            w = pd.Series(1.0 / len(holdings), index=holdings)
        Wreb.loc[d, w.index] = w.values

    W = Wreb.reindex(idx).ffill().fillna(0.0)
    # LAG: weights formed at month-end close d are held from d+1 — the shift is here.
    W_lag = W.shift(1).fillna(0.0)

    daily = net_of_cost(W_lag, rets, cost_bps=cost_bps, name="net_repurchaser_smallcap")
    trades = trades_from_weights(W_lag, rets, sector_map)
    return daily, trades


# ----------------------------------------------------------------------------- expectations

def _sharpe(s):
    s = pd.Series(s).dropna()
    if len(s) < 60 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(252))


def _check_hold_days(ctx):
    hd = [t["hold_days"] for t in ctx["trades"]]
    med = float(np.median(hd)) if hd else 0.0
    return {"pass": med >= 63, "observed": med}


def _check_hysteresis_not_costly(ctx):
    g = ctx["grid"]
    s_def, s_no = _sharpe(g["default"]), _sharpe(g["no_hysteresis"])
    return {"pass": s_def >= s_no - 0.05, "observed": f"hyst={s_def:.2f} no_hyst={s_no:.2f}"}


def _check_beats_universe_ew(ctx):
    s = pd.Series(ctx["search"]).dropna()
    ew = ctx["panel"]["px"].pct_change().mean(axis=1).reindex(s.index)
    return {"pass": _sharpe(s) > _sharpe(ew),
            "observed": f"book={_sharpe(s):.2f} ew_universe={_sharpe(ew):.2f}"}


def _check_repurchaser_breadth(ctx):
    iss = ctx["panel"]["iss"].loc[: ctx["holdout_start"]]
    me = iss.resample("ME").last() if hasattr(pd.offsets, "MonthEnd") else iss.resample("M").last()
    denom = me.notna().sum(axis=1).replace(0, np.nan)
    frac = ((me < 0).sum(axis=1) / denom).dropna()
    obs = float(frac.median()) if len(frac) else 0.0
    return {"pass": obs >= 0.10, "observed": obs}


PRE_REG = (
    "FROZEN design. Universe: survivorship-clean Sharadar small caps (sector-spread, "
    "include_delisted), even-alphabetical half (~700 names) as search; odd half + mid caps "
    "reserved untouched. Signal: trailing 12m % change in split-adjusted shares outstanding "
    "(sharesbas*sharefactor, PIT as-of FILING datekey). Book: LONG-ONLY equal-weight, enter "
    "bottom issuance quintile with issuance<=0, exit only on leaving bottom two quintiles "
    "(hysteresis), $1M 21d-median-ADV floor, monthly rebalance, costs 30bps default "
    "(8/50bps stress variants in grid). NO short leg — issuers are the hard-to-borrow side. "
    "PREDICTIONS: (1) the known large-cap NEGATIVE result is the falsification anchor — if "
    "this small-cap book is also flat/negative the issuance family closes permanently; "
    "(2) premium amplitude is monotone in limits-to-arbitrage: gen universes should rank "
    "small_lowadv >= small_highadv >= midcap (all expected OOS-positive, midcap weakest); if "
    "the premium is STRONGEST in the most liquid slices, suspect artifact and kill; "
    "(3) hysteresis keeps median hold > one quarter without costing Sharpe vs no-hysteresis. "
    "MCPT WARNING (beta-confound trap): this is a long-only beta>0 book — the permutation "
    "test must use the benchmark-relative statistic (book Sharpe minus equal-weight-universe "
    "Sharpe) against a long-only permutation null, not raw Sharpe; the beats_universe_ew "
    "expectation records the same comparison on the search window. ADV-monotonicity across "
    "gen universes is checked by the stage-2 battery itself (separate universes), hence prose "
    "here rather than an expectation fn."
)

SPEC = StrategySpec(
    id="net_repurchaser_smallcap_v1",
    family="share_issuance",
    title="Net-Repurchaser Financing Premium — Small-Cap Long-Only Tilt",
    markets=["us_equity_small_mid"],
    data_desc=("Sharadar SF1 sharesbas*sharefactor (PIT datekey, ARQ) + SEP closeadj/close/volume; "
               "survivorship-clean sector-spread Small universe (even half = search), "
               "odd-half ADV terciles + Mid band as disjoint generalization universes"),
    pre_registration=PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "cost_8bps": {"cost_bps": 8.0},
        "cost_50bps": {"cost_bps": 50.0},
        "no_hysteresis": {"exit_pct": 0.20},
        "inv_vol": {"weighting": "inv_vol"},
        "deciles": {"entry_pct": 0.10, "exit_pct": 0.30},
    },
    scope="broad",
    generalization_universes=["small_lowadv", "small_highadv", "midcap"],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=50,
    expectations=[
        {"name": "hysteresis_holds_long",
         "claim": "hysteresis yields median hold >= 63 trading days (>1 quarter)",
         "check": _check_hold_days},
        {"name": "hysteresis_not_costly",
         "claim": "hysteresis variant Sharpe >= no-hysteresis variant (turnover savings, search window)",
         "check": _check_hysteresis_not_costly},
        {"name": "beats_universe_ew",
         "claim": "net book Sharpe exceeds gross equal-weight-universe Sharpe in search window (beta-confound control)",
         "check": _check_beats_universe_ew},
        {"name": "repurchaser_breadth",
         "claim": ">=10% of small-cap names show negative 12m issuance in a typical month",
         "check": _check_repurchaser_breadth},
    ],
)