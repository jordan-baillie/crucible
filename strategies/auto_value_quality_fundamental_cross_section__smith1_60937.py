# ======================================================================================
# Value x Quality Fundamental Cross-Section  (sector-neutral, dollar-neutral L/S)
#
#   A SLOW-HORIZON FUNDAMENTALS book built in the balance-sheet/income-statement domain,
#   deliberately orthogonal to the portfolio's price/microstructure parent (Amihud x Trend).
#   Two complementary, decades-OOS Fama-French-style risk premia:
#       VALUE   (HML-ish)  -> getting paid to bear cheapness/distress risk  (pro-cyclical tail)
#       QUALITY (RMW-ish)  -> profitability / low-accruals / earnings stability (defensive tail)
#   Low/negative mutual correlation => the COMBINATION is the edge.  Frozen, pre-registered.
#
#   Lookahead discipline: fundamentals are point-in-time via pit_panel (datekey ffill, never
#   calendardate); all weights are applied LAGGED one day (W.shift(1)) -> our responsibility.
#
#   UNIVERSE FIX: frozen design is 'us_universe survivorship-clean -> keep top ~1200 by
#   trailing-60d median dollar volume (ADV>$5M)'.  We pull a broad survivorship-clean list
#   (preferring us_universe; sector_universe supplies the {ticker:sector} map needed for
#   sector-neutralization), then SELECT the top ~1200 BY TRAILING-60d MEDIAN DOLLAR VOLUME.
#   No mid-cap-only tilt; the dollar-volume ranking is the binding inclusion rule.
# ======================================================================================
from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel
import numpy as np, pandas as pd
import warnings

# ----- frozen design constants (NO grid tuning of these) ------------------------------
_START     = "2008-01-01"   # warmup (accruals need 1y, ROA-stability needs 2y) before 2012+ active book
_ADV_MIN   = 5e6            # $5M trailing-60d median dollar volume => borrowable, clean fills
_TOP_N     = 1200          # frozen universe size: top ~1200 by trailing-60d median dollar volume
_GROSS     = 1.5           # gross leverage; long +0.75 / short -0.75 => dollar-neutral, gross<=1.5x
_REBAL     = 21           # trading days between rebalances == MONTHLY (pre-registered, see note)
_NAME_CAP  = 0.05        # |weight| <= 5% of gross (deployment-sanity, anti-pattern #8)
_MIN_SIDE  = 40         # >=40 names per side floor (diversification + borrowability)
_LAST_SECTOR_MAP = {}  # process-global fallback for the sector map (attrs is primary)

_SF1_FIELDS = ["ebit", "revenue", "gp", "netinc", "equity", "assets",
               "debt", "cashneq", "sharesbas", "assetsc", "liabilitiesc"]


# ======================================================================================
# DATA
# ======================================================================================
def _build_panel(tickers, sector_map):
    """Pack returns + liquidity + 6 point-in-time fundamental ratio panels into one
       MultiIndex-column DataFrame (level0=field, level1=ticker). All ratios update DAILY
       via the daily (unadjusted) price; numerators are PIT-ffilled from filing date."""
    closeadj = sep_panel(tickers, _START, field="closeadj")   # split/div adj -> returns only
    close    = sep_panel(tickers, _START, field="close")      # unadjusted -> consistent with shares basis
    vol      = sep_panel(tickers, _START, field="volume")
    keep = [t for t in closeadj.columns if t in close.columns and t in vol.columns]
    closeadj, close, vol = closeadj[keep], close[keep], vol[keep]
    dates = closeadj.index

    ret = closeadj.pct_change()
    adv = (close * vol).rolling(60, min_periods=30).median()  # trailing-60d median $ volume

    # POINT-IN-TIME fundamentals: ART = TTM income statement + latest balance sheet,
    # ffilled by datekey (filing date) -> structurally no look-ahead.
    sf = sf1(keep, _SF1_FIELDS, dimension="ART")
    f = {fld: pit_panel(sf, fld, dates, keep) for fld in _SF1_FIELDS}

    mcap = (f["sharesbas"] * close);          mcap = mcap.where(mcap > 0)
    ev   = (mcap + f["debt"] - f["cashneq"]); ev   = ev.where(ev > 0)
    assets = f["assets"].where(f["assets"] > 0)

    # --- VALUE composite inputs (higher = cheaper) ---
    ey = f["ebit"]    / ev      # earnings yield  (EBIT / EV)
    bp = f["equity"]  / mcap    # book-to-price
    sp = f["revenue"] / mcap    # sales-to-price
    # --- QUALITY composite inputs (higher = better) ---
    gp  = f["gp"] / assets                                   # gross profitability (Novy-Marx)
    nwc = f["assetsc"] - f["liabilitiesc"]
    accr  = -(nwc - nwc.shift(252)) / assets                 # low accruals (= -dNWC/assets, YoY)
    roa   = f["netinc"] / assets
    estab = -roa.rolling(504, min_periods=126).std()         # earnings stability (= -ROA volatility)

    panel = pd.concat({"ret": ret, "adv": adv, "ey": ey, "bp": bp, "sp": sp,
                       "gp": gp, "accr": accr, "estab": estab}, axis=1)
    panel.attrs["sector_map"] = dict(sector_map)
    return panel


def _universe_and_sectors():
    """FROZEN universe pull: a broad survivorship-clean list + {ticker:sector} map.
       Prefer us_universe (the thesis-specified survivorship-clean liquid mid/large universe);
       sector_universe (broad pull) supplies the sector map required for sector-neutralization.
       The binding top-~1200 inclusion rule (by trailing-60d median dollar volume) is applied
       downstream in load_data -- here we only assemble the candidate list + sectors."""
    sector_map = {}
    sec_tickers = []
    try:
        # broad pull (large top_n_per_sector) so dollar-volume ranking, not a mid-cap cap,
        # is what selects the deployable universe; also yields the {ticker:sector} map.
        sec_tickers, sector_map = sector_universe(marketcap="Mid", top_n_per_sector=600)
        sector_map = dict(sector_map)
    except Exception:
        sec_tickers, sector_map = [], {}
    try:
        tickers = list(us_universe(_START))               # thesis-specified survivorship-clean universe
        if not tickers:
            tickers = list(sec_tickers)
    except Exception:
        tickers = list(sec_tickers)
    if not tickers:
        tickers = list(sec_tickers)
    sector_map = {t: sector_map.get(t, "UNK") for t in tickers}
    return tickers, sector_map


def load_data():
    # FROZEN: survivorship-clean universe -> keep top ~1200 by trailing-60d median dollar volume.
    # (ADV>$5M is additionally enforced at every rebalance inside _build_weights.)
    tickers, sector_map = _universe_and_sectors()
    panel = _build_panel(tickers, sector_map)
    adv_rank = panel["adv"].median().dropna().sort_values(ascending=False)  # by trailing-60d $ volume
    keep = list(adv_rank.index[:_TOP_N])
    sub = panel.loc[:, panel.columns.get_level_values(1).isin(keep)]
    sm = {t: sector_map.get(t, "UNK") for t in keep}
    sub.attrs["sector_map"] = sm
    global _LAST_SECTOR_MAP
    _LAST_SECTOR_MAP = sm
    return sub


# ======================================================================================
# SIGNAL HELPERS
# ======================================================================================
def _mean_panels(panels):
    """Element-wise skipna mean across same-shaped panels (a missing component does not nuke
       the composite)."""
    arr = np.stack([p.to_numpy(dtype=float) for p in panels], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(arr, axis=0)
    return pd.DataFrame(m, index=panels[0].index, columns=panels[0].columns)


def _sector_demean(df, sector_map):
    """Per-date sector-mean demeaning => sector-NEUTRAL exposure on the composite z."""
    secs = pd.Series([sector_map.get(t, "UNK") for t in df.columns], index=df.columns)
    out = df.copy()
    for _, grp in secs.groupby(secs):
        cols = list(grp.index)
        sub = df[cols]
        out[cols] = sub.sub(sub.mean(axis=1, skipna=True), axis=0)
    return out


def _side(names, vol_row, target):
    """Inverse-vol weights within one side, summing to `target`, capped at _NAME_CAP, renormalized."""
    vv = vol_row.reindex(names)
    iv = 1.0 / vv.replace(0.0, np.nan)
    med = iv.median()
    iv = iv.fillna(med if np.isfinite(med) else 1.0)
    if iv.sum() == 0 or not np.isfinite(iv.sum()):
        iv = pd.Series(1.0, index=names)
    w = iv / iv.sum() * target
    w = w.clip(lower=-_NAME_CAP, upper=_NAME_CAP)
    tot = w.sum()
    if tot != 0:
        w = w * (target / tot)
    return w


def _build_weights(combined, ret, adv, sector_map):
    """Monthly: long top-quintile / short bottom-quintile of the eligible (ADV>$5M) names,
       inverse-vol within side, dollar-neutral (gross 1.5x), held between rebalances."""
    vol  = ret.rolling(60, min_periods=20).std()
    cols = combined.columns
    W    = pd.DataFrame(np.nan, index=combined.index, columns=cols)
    half = _GROSS / 2.0
    for dt in combined.index[::_REBAL]:
        s = combined.loc[dt]
        a = adv.loc[dt]
        elig = s.notna() & a.notna() & (a > _ADV_MIN)
        s = s[elig]
        if s.shape[0] < 100:
            continue
        qlo, qhi = s.quantile(0.2), s.quantile(0.8)
        longs  = list(s.index[s >= qhi])
        shorts = list(s.index[s <= qlo])
        if len(longs) < _MIN_SIDE or len(shorts) < _MIN_SIDE:
            continue
        v   = vol.loc[dt]
        row = pd.Series(0.0, index=cols)          # full row: non-selected -> 0 (exit)
        wl  = _side(longs,  v,  half)
        ws  = _side(shorts, v, -half)
        row.loc[wl.index] = wl.values
        row.loc[ws.index] = ws.values
        W.loc[dt] = row.values
    return W.ffill().fillna(0.0)                    # hold between rebalances; flat before first


def signal(panel, **params):
    """legs='both' (default composite) | 'value' | 'quality'  -> daily net returns + trade ledger."""
    legs = params.get("legs", "both")
    sector_map = panel.attrs.get("sector_map") or _LAST_SECTOR_MAP
    ret, adv = panel["ret"], panel["adv"]

    value_z   = _mean_panels([xs_zscore(panel[f]) for f in ("ey", "bp", "sp")])      # winsorized xs-z
    quality_z = _mean_panels([xs_zscore(panel[f]) for f in ("gp", "accr", "estab")])
    if legs == "value":
        combined = value_z
    elif legs == "quality":
        combined = quality_z
    else:
        combined = _mean_panels([value_z, quality_z])     # fixed 0.5/0.5 pre-registered weights

    combined = _sector_demean(combined, sector_map)        # sector-neutral
    W = _build_weights(combined, ret, adv, sector_map)

    W_lag = W.shift(1)                                      # <-- the 1-day lag is OUR responsibility
    daily = net_of_cost(W_lag, ret, cost_bps=8.0, name="value_quality_%s" % legs)
    trades = trades_from_weights(W_lag, ret, sector_map)   # kit stamps entry_regime (contract)

    active = W_lag.abs().sum(axis=1) > 0                    # trim leading flat period
    if active.any():
        daily = daily.loc[active.idxmax():]
    return daily, trades


# ======================================================================================
# GEN-DATA (defined for completeness; scope='local' => stage-2 battery is NOT run, forward
#           validation confirms instead. Functional sub-universe slicer if ever introspected.)
# ======================================================================================
def load_gen_data(label):
    panel = load_data()
    sm = dict(panel.attrs.get("sector_map", _LAST_SECTOR_MAP))
    tickers = list(panel["ret"].columns)
    if label.startswith("sector:"):
        sec = label.split(":", 1)[1]
        keep = [t for t in tickers if sm.get(t) == sec]
    elif label in ("liquid_half", "less_liquid_half"):
        med = panel["adv"].median().sort_values()
        n = len(med) // 2
        keep = list(med.index[n:]) if label == "liquid_half" else list(med.index[:n])
    else:
        keep = tickers
    sub = panel.loc[:, panel.columns.get_level_values(1).isin(keep)]
    sub.attrs["sector_map"] = {t: s for t, s in sm.items() if t in keep}
    return sub


# ======================================================================================
# SOFT EXPECTATIONS (machine-checkable; both read ctx['grid'] -> ZERO extra signal() calls)
# ======================================================================================
def _exp_complement(ctx):
    """Pre-reg success criterion (6): value-leg vs quality-leg are COMPLEMENTS, not the same trade."""
    g = ctx.get("grid", {})
    v, q = g.get("value_only"), g.get("quality_only")
    if v is None or q is None:
        return {"pass": False, "observed": "legs unavailable"}
    df = pd.concat([v.rename("v"), q.rename("q")], axis=1).dropna()
    if len(df) < 60:
        return {"pass": False, "observed": "insufficient overlap"}
    rho = float(df["v"].corr(df["q"]))
    return {"pass": rho <= 0.30, "observed": round(rho, 3)}


def _exp_opposite_tails(ctx):
    """Pre-reg diagnostic (7a): opposite tails — in 2020 flight-to-quality (in-sample, < holdout)
       the defensive QUALITY leg should out-earn the pro-cyclical VALUE leg."""
    g = ctx.get("grid", {})
    v, q = g.get("value_only"), g.get("quality_only")
    if v is None or q is None:
        return {"pass": False, "observed": "legs unavailable"}
    vw, qw = v.loc["2020-02-15":"2020-04-30"], q.loc["2020-02-15":"2020-04-30"]
    if len(vw) < 5 or len(qw) < 5:
        return {"pass": False, "observed": "no 2020 window"}
    vcum = float((1 + vw).prod() - 1)
    qcum = float((1 + qw).prod() - 1)
    return {"pass": qcum >= vcum, "observed": "quality=%.4f value=%.4f" % (qcum, vcum)}


# ======================================================================================
# SPEC
# ======================================================================================
SPEC = StrategySpec(
    id="value_quality_xs",
    family="fundamental_value_quality",
    title=("Value x Quality Fundamental Cross-Section -- sector-neutral, dollar-neutral, "
           "beta-hedged L/S on liquid US mid/large-caps (slow-horizon fundamentals book, "
           "orthogonal to the price/liquidity Amihud x Trend parent)"),
    markets=["US_equity"],
    data_desc=("Sharadar SF1 point-in-time fundamentals (dimension='ART', datekey-as-of, "
               "NEVER calendardate); Sharadar SEP daily closeadj/close/volume "
               "(survivorship-clean, delisted included); SPY via yfinance for the declared "
               "beta-hedge sleeve. $0 incremental data."),
    pre_registration=(
        "FROZEN, PRE-REGISTERED single composite — no parameter tuning. "
        "UNIVERSE: us_universe survivorship-clean liquid US equities -> KEEP TOP ~1200 BY "
        "TRAILING-60d MEDIAN DOLLAR VOLUME (dollar-volume ranking is the binding inclusion rule; "
        "sector_universe supplies the {ticker:sector} map only). Additionally restricted at each "
        "rebalance to ADV>$5M so every long AND short leg is liquid/borrowable. "
        "VALUE z = mean of cross-sectional winsorized z-scores of {EBIT/EV, book-to-price, "
        "sales-to-price}; QUALITY z = mean z of {gross-profit/assets, -dNWC/assets (low accruals), "
        "-ROA volatility (earnings stability)}; all numerators are PIT-ffilled from datekey, "
        "denominators update DAILY via the unadjusted price. COMPOSITE = 0.5*value_z + 0.5*quality_z "
        "(fixed weights), then PER-DATE SECTOR-DEMEANED => sector-neutral exposure. "
        "BOOK: long top-quintile / short bottom-quintile of the eligible composite, inverse-vol "
        "within side, dollar-neutral, gross<=1.5x, |name|<=5% of gross, >=40 names/side. "
        "REBALANCE: every 21 trading days (MONTHLY). This is a DELIBERATE, pre-registered deviation "
        "from the harness weekly default: the driver is quarterly point-in-time fundamentals that "
        "only move on filing dates, so weekly rebalancing would churn 8bps against a near-static "
        "signal with no informational benefit. COSTS: 8bps on turnover via net_of_cost on the LAGGED "
        "weight matrix (W.shift(1)); turnover is filing/price-drift driven. "
        "BETA HEDGE: declared as the SPY sleeve on the spec (hedge_tickers=['SPY'], hedge_cap=0.35) — "
        "the deployment gate judges the ALPHA book alone and gates the sleeve on whitelist+cap; the "
        "reported returns are the dollar-neutral L/S alpha (the harness applies/gates the residual-beta "
        "trim), avoiding the undeclared-continuous-ETF single_name_share failure mode. "
        "SUCCESS = full gate stack (MCPT market-neutral absolute null, write-once holdout, DSR over the "
        "declared grid, FDR) PASS. "
        "MACHINE-CHECKABLE EXPECTATIONS (non-gating): (1) value-leg vs quality-leg net-return "
        "correlation <= +0.30 over the search window (complements, not the same trade) — checked via "
        "the declared value_only/quality_only grid returns (no extra signal call); (2) opposite tails — "
        "in the 2020 flight-to-quality the quality leg out-earns the value leg. "
        "PROSE-ONLY (not machine-checkable here because the parent's Amihud/trend streams are not in "
        "ctx — to be confirmed in forward paper): falsifiable orthogonality target |rho|<=0.2 of the "
        "combined book to BOTH the Amihud and trend_returns streams, run in paper from day one with an "
        "identical start date alongside the deployed Amihud book. "
        "GRID is diagnostic decomposition only (legs), not selection; 'default' (both) is primary and "
        "the legs carry honest DSR search burden."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}, "value_only": {"legs": "value"}, "quality_only": {"legs": "quality"}},
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=120,
    hedge_tickers=["SPY"],
    hedge_cap=0.35,
    expectations=[
        {"name": "value_quality_complement",
         "claim": "value-leg vs quality-leg net-return correlation <= +0.30 over the search window",
         "check": _exp_complement},
        {"name": "opposite_tails_2020",
         "claim": "in the 2020 flight-to-quality (2020-02-15..04-30) the defensive quality leg "
                  "out-earns the pro-cyclical value leg",
         "check": _exp_opposite_tails},
    ],
)