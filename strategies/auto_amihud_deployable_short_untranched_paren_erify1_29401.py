"""
amihud_illiq_wideband_untranched_v1
====================================
HEAD-TO-HEAD construction experiment vs the VALIDATED amihud_illiq_tranched_v3.

The 2026-06-12 retro-verification FALSIFIED tranched_v3's pre-registered turnover claim
(tranching RAISED long-leg turnover 39%, it did not halve it). Tranching's surviving
benefit is rebalance-date-luck immunity, bought for ~0.1 search Sharpe. HYPOTHESIS:
WIDER HYSTERESIS BANDS on the UNTRANCHED parent buy the same date-robustness CHEAPER
(fewer marginal swaps => the specific month-end you form on matters less), while keeping
the untranched book's higher Sharpe and lower turnover.

Frozen design inherited verbatim from the tranched_v3 frozen parent EXCEPT bands and
n_tranches: within-size-tercile Amihud sort; long = most-illiquid quintile EW (10% name
cap); short = top-15 most-liquid per tercile; $10-$500 price filter; residual-only IWM
trim to |beta|<=0.30, declared hedge sleeve cap 0.35; costs 60bps RT long / 15bps RT
short + 50bps/yr borrow; monthly SINGLE-DATE rebalance. PRIMARY: long_band=2.0,
short_band=2.0 (parent used 1.5/1.6).

All four mechanism claims are MACHINE-CHECKED via StrategySpec(expectations=[...]):
E1 date-luck (offsets 0/7/14 via pre-declared grid variants — zero extra signal calls),
E2 turnover <= 1.32x/yr untranched-parent baseline, E3 Sharpe retention >= 90% of the
narrow-band reference, E4 strict band->turnover monotonicity.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, yf_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------- constants
START = "2011-01-01"
HOLDOUT_START = "2022-01-01"

AMI_LB = 63                # Amihud lookback (days)
DVOL_LB = 63               # size/liquidity proxy lookback
BETA_LB = 126              # per-name beta window for the residual IWM trim
PRICE_MIN, PRICE_MAX = 10.0, 500.0
LONG_QUINTILE = 0.20       # entry: most-illiquid quintile within tercile
SHORT_N = 15               # entry: top-15 most-liquid per tercile
NAME_CAP = 0.10            # long-leg per-name cap
BETA_TOL = 0.30            # residual beta tolerance before IWM trim
HEDGE_CAP = 0.35           # |IWM| sleeve cap (also declared on spec)
LONG_COST_BPS = 30.0       # per side (60bps RT, frozen design)
SHORT_COST_BPS = 7.5       # per side (15bps RT, frozen design)
HEDGE_COST_BPS = 5.0       # IWM is hyper-liquid
BORROW_BPS_YR = 50.0       # short borrow
MIN_ELIGIBLE = 90          # skip reform in pathologically thin months (gate0 check)

SEARCH_SECTORS = {"Technology", "Healthcare", "Financial Services",
                  "Industrials", "Consumer Cyclical"}

GEN_UNIVERSES = {
    # disjoint from search by SECTOR (same small+mid tier)
    "smallmid_energy_materials_utils": {
        "sectors": {"Energy", "Basic Materials", "Utilities"},
        "tiers": ("Small", "Mid"), "top_n": 55},
    "smallmid_defensive_comms_re": {
        "sectors": {"Consumer Defensive", "Communication Services", "Real Estate"},
        "tiers": ("Small", "Mid"), "top_n": 55},
    # disjoint from search by CAP TIER (all sectors)
    "largecap_all_sectors": {
        "sectors": None,  # all
        "tiers": ("Large",), "top_n": 30},
}

# module-level caches: signal() deposits measured long-leg turnover here so the
# expectation checks can read it without any extra signal() calls.
_TURNOVER = {}      # (n_tickers, long_band, short_band, n_tranches, offset) -> ann. turnover
_SECTOR_MAPS = {}   # n_tickers -> {ticker: sector}  (fallback if panel.attrs stripped)


# ----------------------------------------------------------------------------- data
def _build_panel(sectors, tiers, top_n_per_sector, label):
    tickers, smap = [], {}
    for tier in tiers:
        t, m = sector_universe(marketcap=tier, top_n_per_sector=top_n_per_sector)
        for tk in t:
            sec = m.get(tk)
            if sectors is not None and sec not in sectors:
                continue
            if tk not in smap:
                tickers.append(tk)
                smap[tk] = sec
    tickers = sorted(set(tickers))

    px = sep_panel(tickers, START, field="closeadj")
    cl = sep_panel(tickers, START, field="close")
    vo = sep_panel(tickers, START, field="volume")
    common = sorted(set(px.columns) & set(cl.columns) & set(vo.columns))
    px, cl, vo = px[common], cl[common], vo[common]

    iwm = yf_panel(["IWM"], START)
    iwm = iwm.reindex(px.index).ffill()
    iwm.columns = ["IWM"]

    panel = pd.concat({"closeadj": px, "close": cl, "volume": vo, "hedge": iwm}, axis=1)
    smap = {t: smap.get(t, "Unknown") for t in common}
    panel.attrs["sector_map"] = smap
    panel.attrs["universe_label"] = label
    _SECTOR_MAPS[len(common)] = smap
    return panel


def load_data() -> pd.DataFrame:
    """Search slice: small+mid, 5 sectors (disjoint from all gen universes)."""
    return _build_panel(SEARCH_SECTORS, ("Small", "Mid"), 45, "search")


def load_gen_data(label) -> pd.DataFrame:
    cfg = GEN_UNIVERSES[label]
    return _build_panel(cfg["sectors"], cfg["tiers"], cfg["top_n"], label)


# ----------------------------------------------------------------------------- helpers
def _form_dates(idx, offset_days):
    """Month-end trading days, optionally shifted by k calendar days then snapped
    to the next trading day (E1 date-luck probe)."""
    me = idx.to_series().groupby(idx.to_period("M")).max()
    out = []
    for d in me:
        if offset_days == 0:
            out.append(d)
        else:
            pos = idx.searchsorted(d + pd.Timedelta(days=int(offset_days)))
            if pos < len(idx):
                out.append(idx[pos])
    return sorted(set(out))


def _sharpe(s):
    s = pd.Series(s).dropna()
    if len(s) < 60 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(252))


# ----------------------------------------------------------------------------- signal
def signal(panel, long_band=2.0, short_band=2.0, n_tranches=1, form_offset_days=0):
    closeadj = panel["closeadj"]
    close = panel["close"]
    volume = panel["volume"]
    iwm_px = panel["hedge"]["IWM"]
    idx = closeadj.index
    ntick = closeadj.shape[1]

    rets = closeadj.pct_change()
    iwm_ret = iwm_px.pct_change()

    dollar_vol = (close * volume).replace(0.0, np.nan)
    # Amihud illiquidity: mean(|r| / $vol) over trailing window (scale-free, used in ranks)
    ami = (rets.abs() / dollar_vol).rolling(AMI_LB, min_periods=40).mean() * 1e9
    dvol_med = dollar_vol.rolling(DVOL_LB, min_periods=40).median()

    n_tranches = max(1, int(n_tranches))
    fdates = [d for d in _form_dates(idx, form_offset_days) if d >= idx[min(140, len(idx) - 1)]]

    cohorts = [{"long": set(), "short": set(),
                "wl": pd.Series(dtype=float), "ws": pd.Series(dtype=float)}
               for _ in range(n_tranches)]
    WL_rows, WS_rows, WH_rows = {}, {}, {}
    long_keep_frac = min(LONG_QUINTILE * float(long_band), 0.90)
    short_keep_n = int(round(SHORT_N * float(short_band)))

    for i, d in enumerate(fdates):
        a, dv, p = ami.loc[d], dvol_med.loc[d], close.loc[d]
        elig = a.notna() & dv.notna() & (dv > 0) & (p >= PRICE_MIN) & (p <= PRICE_MAX)
        names = a.index[elig]
        if len(names) >= MIN_ELIGIBLE:
            terc = pd.qcut(dv[names].rank(method="first"), 3, labels=False)
            long_entry, long_keep = set(), set()
            short_structs = []  # per tercile: (keep_set, liquid-first ordered list)
            for t in (0, 1, 2):
                tn = names[terc == t]
                if len(tn) < 10:
                    continue
                ill_pct = a[tn].rank(ascending=False, pct=True)   # low pct = most illiquid
                long_entry |= set(ill_pct.index[ill_pct <= LONG_QUINTILE])
                long_keep |= set(ill_pct.index[ill_pct <= long_keep_frac])
                liq_rank = a[tn].rank(ascending=True, method="first")  # 1 = most liquid
                order = list(liq_rank.sort_values().index)
                short_structs.append((set(order[:short_keep_n]), order))

            k = i % n_tranches  # cohort reforming this month
            co = cohorts[k]
            new_long = (co["long"] & long_keep) | long_entry
            new_short = set()
            for keep_set, order in short_structs:
                keeps = co["short"] & keep_set
                adds = [x for x in order if x not in keeps][:max(0, SHORT_N - len(keeps))]
                new_short |= keeps | set(adds)

            if len(new_long) >= 20 and len(new_short) >= 20:
                wl = pd.Series(1.0 / len(new_long), index=sorted(new_long)).clip(upper=NAME_CAP)
                wl = wl / wl.sum()
                ws = pd.Series(-1.0 / len(new_short), index=sorted(new_short))
                co["long"], co["short"], co["wl"], co["ws"] = new_long, new_short, wl, ws

        # combine cohorts (each contributes 1/n_tranches of the book)
        cl_w, cs_w = pd.Series(dtype=float), pd.Series(dtype=float)
        for co in cohorts:
            cl_w = cl_w.add(co["wl"] / n_tranches, fill_value=0.0)
            cs_w = cs_w.add(co["ws"] / n_tranches, fill_value=0.0)
        comb = cl_w.add(cs_w, fill_value=0.0)

        # residual-only IWM trim: hedge ONLY the beta beyond +/-BETA_TOL, cap at sleeve cap
        h = 0.0
        if len(comb) > 0:
            win = rets.loc[:d].tail(BETA_LB)
            iw = iwm_ret.loc[:d].tail(BETA_LB)
            var = iw.var()
            if var and var > 0 and len(iw) >= 60:
                sub = win.reindex(columns=comb.index)
                betas = (sub.sub(sub.mean()).mul(iw - iw.mean(), axis=0).mean() / var).fillna(1.0)
                book_beta = float((comb * betas).sum())
                if abs(book_beta) > BETA_TOL:
                    h = float(np.clip(-(book_beta - np.sign(book_beta) * BETA_TOL),
                                      -HEDGE_CAP, HEDGE_CAP))
        WL_rows[d], WS_rows[d], WH_rows[d] = cl_w, cs_w, h

    # daily weight matrices (held between monthly reforms), execution lag = shift(1)
    WL = (pd.DataFrame.from_dict(WL_rows, orient="index")
          .reindex(columns=rets.columns).sort_index()
          .reindex(idx).ffill().fillna(0.0))
    WS = (pd.DataFrame.from_dict(WS_rows, orient="index")
          .reindex(columns=rets.columns).sort_index()
          .reindex(idx).ffill().fillna(0.0))
    WH = pd.Series(WH_rows).sort_index().reindex(idx).ffill().fillna(0.0).to_frame("IWM")

    r_long = net_of_cost(WL.shift(1), rets, cost_bps=LONG_COST_BPS, name="long")
    r_short = net_of_cost(WS.shift(1), rets, cost_bps=SHORT_COST_BPS, name="short")
    r_hedge = net_of_cost(WH.shift(1), iwm_ret.to_frame("IWM"),
                          cost_bps=HEDGE_COST_BPS, name="hedge")
    borrow = WS.shift(1).abs().sum(axis=1) * (BORROW_BPS_YR / 1e4 / 252.0)

    daily = (r_long.add(r_short, fill_value=0.0)
                   .add(r_hedge, fill_value=0.0)
                   .sub(borrow, fill_value=0.0))
    if fdates:
        daily = daily.loc[daily.index >= fdates[0]]
    daily.name = "amihud_illiq_wideband_untranched_v1"

    # deposit measured long-leg annualized one-sided turnover (search window only)
    # for the E2/E4 machine checks — no extra signal() calls needed at check time.
    cut = WL.loc[WL.index < pd.Timestamp(HOLDOUT_START)]
    if len(cut) > 400:
        years = len(cut) / 252.0
        turn = float(cut.diff().abs().sum(axis=1).sum() / 2.0 / years)
        _TURNOVER[(ntick, float(long_band), float(short_band),
                   int(n_tranches), int(form_offset_days))] = turn

    # trade ledger (longs + shorts + declared IWM sleeve)
    smap = dict(panel.attrs.get("sector_map") or _SECTOR_MAPS.get(ntick, {}))
    W_all = WL.add(WS, fill_value=0.0)
    W_all["IWM"] = WH["IWM"]
    rets_all = rets.copy()
    rets_all["IWM"] = iwm_ret
    full_map = {t: smap.get(t, "Unknown") for t in W_all.columns}
    full_map["IWM"] = "ETF-Hedge"
    trades = trades_from_weights(W_all.shift(1), rets_all, full_map)

    return daily, trades


# ----------------------------------------------------------------------------- expectations
def _chk_date_luck(ctx):
    """E1: forming at month-end +0/+7/+14 calendar days must all be net-positive in the
    search window, with cross-offset Sharpe spread < 50% of the mean (tranched_v3's bar)."""
    g = ctx.get("grid", {}) or {}
    vals = [_sharpe(g.get("default")), _sharpe(g.get("offset7")), _sharpe(g.get("offset14"))]
    m = float(np.mean(vals))
    ok = all(v > 0 for v in vals) and m > 0 and (max(vals) - min(vals)) < 0.5 * m
    return {"pass": bool(ok),
            "observed": f"offset Sharpes 0/7/14d = {vals[0]:.2f}/{vals[1]:.2f}/{vals[2]:.2f}, "
                        f"spread={max(vals)-min(vals):.2f} vs bar {0.5*m:.2f}"}


def _chk_turnover(ctx):
    """E2: primary long-leg annualized turnover <= 1.32x/yr (the measured untranched
    narrow-band parent baseline). Wider bands must REDUCE churn or the story is wrong."""
    ntick = ctx["panel"]["closeadj"].shape[1]
    t = _TURNOVER.get((ntick, 2.0, 2.0, 1, 0))
    ok = (t is not None) and (t <= 1.32)
    return {"pass": bool(ok), "observed": f"long-leg ann. turnover = {t if t is not None else 'missing'}"}


def _chk_sharpe_retention(ctx):
    """E3: primary (bands 2.0/2.0) search Sharpe >= 90% of the exact-parent reference
    (bands 1.5/1.6) — wider bands hold staler books; >10% Sharpe cost fails the trade-off."""
    g = ctx.get("grid", {}) or {}
    s_def, s_ref = _sharpe(g.get("default")), _sharpe(g.get("bands_parent_1p5_1p6"))
    ok = s_def >= 0.90 * s_ref
    return {"pass": bool(ok), "observed": f"primary {s_def:.3f} vs 0.90*ref {0.90*s_ref:.3f}"}


def _chk_band_monotone(ctx):
    """E4: long-leg turnover strictly decreasing across bands 1.5 -> 2.0 -> 2.5 -> 3.0.
    Non-monotone = the band parameter is not doing what the mechanism story says."""
    ntick = ctx["panel"]["closeadj"].shape[1]
    keys = [(ntick, 1.5, 1.6, 1, 0), (ntick, 2.0, 2.0, 1, 0),
            (ntick, 2.5, 2.5, 1, 0), (ntick, 3.0, 3.0, 1, 0)]
    ts = [_TURNOVER.get(k) for k in keys]
    ok = all(t is not None for t in ts) and all(ts[i] > ts[i + 1] for i in range(3))
    return {"pass": bool(ok),
            "observed": "turnover 1.5->3.0: " + "/".join("NA" if t is None else f"{t:.2f}" for t in ts)}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="amihud_illiq_wideband_untranched_v1",
    family="illiquidity_premium",
    title=("Amihud deployable-short — UNTRANCHED parent with WIDER HYSTERESIS BANDS "
           "(head-to-head vs validated tranched_v3)"),
    markets=["US_smallmid_equities"],
    data_desc=("Sharadar SEP close/closeadj/volume (survivorship-clean, delisted incl.), "
               "small+mid 5-sector search slice identical to tranched_v3; IWM closes via "
               "yfinance for the declared residual hedge sleeve only. Owned, $0."),
    pre_registration=(
        "MOTIVATION: retro-verification (2026-06-12) of validated amihud_illiq_tranched_v3 "
        "FALSIFIED its turnover claim — tranched long-leg turnover 1.83x/yr vs untranched "
        "1.32x/yr (ratio 1.39 vs the pre-registered <=0.60). Tranching's surviving benefit "
        "is rebalance-date-luck immunity, bought for ~0.1 search Sharpe. HYPOTHESIS: wider "
        "hysteresis bands on the UNTRANCHED parent buy the same date-robustness cheaper. "
        "FROZEN (all inherited verbatim from the tranched_v3 frozen parent except bands and "
        "n_tranches): within-size-tercile Amihud sort, long = most-illiquid quintile EW with "
        "10% cap, short = top-15 most-liquid per tercile, $10-$500 filter, residual-only IWM "
        "trim |beta|<=0.30 declared sleeve cap 0.35, costs 60bps RT long / 15bps RT short + "
        "50bps/yr borrow, monthly single-date rebalance. PRIMARY: long_band=2.0, "
        "short_band=2.0. Grid pre-registered for the DSR burden: parent bands 1.5/1.6 "
        "(exact-parent reference), 2.5/2.5, 3.0/3.0, tranched n=3 at parent bands "
        "(= validated v3, head-to-head reference), and form-date offsets +7/+14d at primary "
        "bands (these double as the E1 date-luck probe — zero extra signal calls). "
        "MACHINE-CHECKED EXPECTATIONS: E1 all three offsets positive with Sharpe spread "
        "<50% of mean; E2 primary long-leg turnover <=1.32x/yr; E3 primary search Sharpe "
        ">=90% of narrow-band reference; E4 turnover strictly decreasing across bands "
        "1.5->3.0. HEAD-TO-HEAD VERDICT RULE: prefer this construction for scale-up ONLY if "
        "(i) it passes ALL gates, (ii) all four expectations hold, and (iii) holdout Sharpe "
        ">= tranched_v3's 1.46. Any other outcome: tranched_v3 remains the family's "
        "deployable construction and this is banked as the negative branch. NOT "
        "tune-to-rescue: tranched_v3 passed and stays deployed; same family, so the FDR bar "
        "accounts for the additional shot. Scope broad: the band mechanism is "
        "construction-level and must replicate wherever the premium does — 3 disjoint gen "
        "universes (2 sector-disjoint small+mid slices, 1 cap-tier-disjoint largecap)."),
    load_data=load_data,
    signal=signal,
    default_params={"long_band": 2.0, "short_band": 2.0, "n_tranches": 1, "form_offset_days": 0},
    grid={
        "default": {},
        "bands_parent_1p5_1p6": {"long_band": 1.5, "short_band": 1.6},
        "bands_2p5": {"long_band": 2.5, "short_band": 2.5},
        "bands_3p0": {"long_band": 3.0, "short_band": 3.0},
        "tranched_v3_ref": {"long_band": 1.5, "short_band": 1.6, "n_tranches": 3},
        "offset7": {"form_offset_days": 7},
        "offset14": {"form_offset_days": 14},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES.keys()),
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT_START,
    deploy_max_positions=75,
    hedge_tickers=["IWM"],
    hedge_cap=0.35,
    expectations=[
        {"name": "date_luck_immunity",
         "claim": "month-end +0/+7/+14d formations all positive, Sharpe spread <50% of mean",
         "check": _chk_date_luck},
        {"name": "turnover_reduced",
         "claim": "primary long-leg annualized turnover <= 1.32x/yr (narrow-band parent baseline)",
         "check": _chk_turnover},
        {"name": "sharpe_retention",
         "claim": "primary search Sharpe >= 90% of exact-parent narrow-band reference",
         "check": _chk_sharpe_retention},
        {"name": "band_turnover_monotone",
         "claim": "long-leg turnover strictly decreasing across bands 1.5 -> 2.0 -> 2.5 -> 3.0",
         "check": _chk_band_monotone},
    ],
)