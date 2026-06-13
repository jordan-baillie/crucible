# Treasury auction-cycle concession premium — getting paid to absorb duration supply.
#
# FROZEN, zero-optimization event-window book: after each NOMINAL COUPON auction, go long the
# nearest-duration Treasury ETF at the auction-day close and hold a fixed 5 trading days; cash
# otherwise. Auctions conclude ~1:00pm ET and results print immediately, so a same-day 4:00pm
# close entry is look-ahead-safe by construction (and the lagged weight matrix earns only
# t+1..t+5). Mechanism: balance-sheet-constrained dealers must warehouse fresh supply and shed
# it over the following days; the post-auction price recovery is the liquidity provider's fee
# (Lou-Yan-Zhang; Fleming/Rosenberg; Sigaux). This is a supply-absorption rent, NOT a rate call.
#
# FAITHFULNESS FIX: the proposal's FROZEN rule is 10y->IEF + 30y->TLT, equal-weight, 1x gross
# PER ETF (no cross-ETF vol normalization). The DEFAULT signal now implements exactly that. The
# broader nominal curve (SHY/IEI for 2/3/5/7y) is retained only as an explicitly-labeled grid
# variant for deployment diversification; it is NOT the primary tested book.

import re
import numpy as np, pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, us_universe, sf1, yf_panel, fred_series, trend_returns, inv_vol_position
from sdk.adapters import treasury_auctions
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights, pit_panel

# ---- static maps (frozen) -------------------------------------------------------------------
_STD = [2, 3, 5, 7, 10, 20, 30]
_TENOR_ETF = {2: "SHY", 3: "SHY", 5: "IEI", 7: "IEI", 10: "IEF", 20: "TLT", 30: "TLT"}
_ETF_SECTOR = {"SHY": "UST_1_3Y", "IEI": "UST_3_7Y", "IEF": "UST_7_10Y", "TLT": "UST_20Y_PLUS"}
_ETFS = ["SHY", "IEI", "IEF", "TLT"]
_HOLD = 5
_START = "2009-06-01"          # buffer ahead of the 2010 sample


# ---- auction-calendar normalization (defensive about the owned adapter's exact schema) -------
def _find_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for nm in names:
        if nm in df.columns:
            return nm
        if nm.lower() in low:
            return low[nm.lower()]
    for nm in names:
        for lc, orig in low.items():
            if nm.lower() in lc:
                return orig
    return None


def _parse_term_years(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower()
    yrs = 0.0
    m = re.search(r'(\d+)\s*-?\s*year', s)
    if m:
        yrs += float(m.group(1))
    m2 = re.search(r'(\d+)\s*-?\s*month', s)
    if m2:
        yrs += float(m2.group(1)) / 12.0
    if yrs == 0.0:
        m3 = re.search(r'(\d+(?:\.\d+)?)', s)
        if m3:
            yrs = float(m3.group(1))
    return yrs if yrs > 0 else np.nan


def _map_std_tenor(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(v):
        return np.nan
    for t in _STD:
        if (t - 1.0) < v <= (t + 0.2):
            return float(t)
    return np.nan


def _snap_to_trading(dates, trading_index):
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    pos = trading_index.searchsorted(di, side="left")
    out = [trading_index[p] if p < len(trading_index) else pd.NaT for p in pos]
    return pd.DatetimeIndex(out)


def _auctions_normalized(start):
    raw = treasury_auctions()
    df = raw.copy()
    date_c = _find_col(df, ["auction_date", "auctiondate", "auctionDate", "date", "auction_dt"])
    term_c = _find_col(df, ["security_term", "securityTerm", "term", "original_term", "tenor_str"])
    tenor_c = _find_col(df, ["tenor", "tenor_years", "years", "maturity_years"])
    type_c = _find_col(df, ["security_type", "securityType", "type", "instrument_type"])
    btc_c = _find_col(df, ["bid_to_cover", "bidToCoverRatio", "bid_to_cover_ratio", "btc", "bid_cover"])
    tail_c = _find_col(df, ["tail", "tail_bps", "auction_tail", "tail_basis_points"])
    if date_c is None:
        raise ValueError("treasury_auctions(): no auction-date column; cols=%s" % list(df.columns))

    dates = pd.to_datetime(df[date_c], errors="coerce")
    if tenor_c is not None:
        ty = pd.to_numeric(df[tenor_c], errors="coerce")
        if ty.isna().mean() > 0.5 and term_c is not None:
            ty = df[term_c].map(_parse_term_years)
    elif term_c is not None:
        ty = df[term_c].map(_parse_term_years)
    else:
        raise ValueError("treasury_auctions(): no tenor/term column; cols=%s" % list(df.columns))
    std = ty.map(_map_std_tenor)

    txt = pd.Series("", index=df.index, dtype="object")
    if type_c is not None:
        txt = txt.str.cat(df[type_c].astype(str).str.lower(), sep=" ")
    if term_c is not None:
        txt = txt.str.cat(df[term_c].astype(str).str.lower(), sep=" ")
    bad = txt.str.contains("bill|cmb|tips|inflation|frn|float|strip", regex=True, na=False)

    out = pd.DataFrame({
        "date": dates,
        "tenor": std,
        "btc": pd.to_numeric(df[btc_c], errors="coerce") if btc_c is not None else np.nan,
        "tail": pd.to_numeric(df[tail_c], errors="coerce") if tail_c is not None else np.nan,
    })
    out = out[(~bad) & out["tenor"].notna() & out["date"].notna()]
    out = out[out["date"] >= pd.Timestamp(start)].sort_values("date").reset_index(drop=True)
    return out


# ---- panel: ETF closes + per-tenor event flags + tail/btc diagnostics -----------------------
def load_data():
    px = yf_panel(_ETFS, start=_START)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.reindex(columns=_ETFS).sort_index()
    px = px[~px.index.duplicated(keep="last")].dropna(how="all")
    px = px.ffill(limit=2)

    auc = _auctions_normalized(_START)
    trading = px.index

    panel = pd.DataFrame(index=trading)
    for e in _ETFS:
        panel["px_" + e] = px[e]
    for t in _STD:
        ev_c, tl_c, bc_c = "ev_%dY" % t, "tail_%dY" % t, "btc_%dY" % t
        panel[ev_c] = 0.0
        panel[tl_c] = np.nan
        panel[bc_c] = np.nan
        sub = auc[auc["tenor"] == t]
        if not len(sub):
            continue
        td = _snap_to_trading(sub["date"].values, trading)
        s = pd.DataFrame({"tdate": td, "tail": sub["tail"].values, "btc": sub["btc"].values})
        s = s.dropna(subset=["tdate"]).drop_duplicates("tdate", keep="first").set_index("tdate")
        panel.loc[panel.index.isin(s.index), ev_c] = 1.0
        panel[tl_c] = s["tail"].reindex(panel.index)
        panel[bc_c] = s["btc"].reindex(panel.index)

    panel = panel.dropna(subset=["px_IEF", "px_TLT"], how="all")
    panel.attrs["tenor_etf"] = dict(_TENOR_ETF)
    return panel


def load_gen_data(label):
    return load_data()


# ---- the only novel code: the frozen event-window signal ------------------------------------
def signal(panel, hold_days=_HOLD, tenors=(10, 30), cost_bps=8.0, **params):
    # FROZEN DEFAULT: 10y->IEF + 30y->TLT (the proposal's headline). Other tenors only via grid.
    tenors = sorted({int(t) for t in tenors if int(t) in _TENOR_ETF})
    etfs = sorted({_TENOR_ETF[t] for t in tenors})
    px = panel[["px_" + e for e in etfs]].copy()
    px.columns = etfs
    rets = px.pct_change()
    idx = panel.index
    n = len(idx)

    # Event-window exposure: 1x in the nearest-duration ETF for hold_days closes starting at the
    # auction-day close. Overlapping events sharing one ETF -> union (average of 1x exposures = 1x),
    # CAPPED AT 1x GROSS PER ETF (frozen rule). NO cross-ETF normalization, NO inverse-vol tilt:
    # both IEF and TLT can be held at 1x simultaneously (up to 2x total gross), long-only, cash off-event.
    E = pd.DataFrame(0.0, index=idx, columns=etfs)
    for t in tenors:
        col = "ev_%dY" % t
        if col not in panel.columns:
            continue
        etf = _TENOR_ETF[t]
        active = np.zeros(n)
        for i in np.where(panel[col].fillna(0.0).values > 0)[0]:
            active[i:i + hold_days] = 1.0
        E[etf] = np.maximum(E[etf].values, active)        # 1x per ETF when any event active

    # LAG: weights decided at close(t) earn return(t+1)..(t+hold). Pass the SAME lagged matrix to
    # both kit functions so the cost model and the trade ledger are mutually consistent.
    Wl = E.shift(1).fillna(0.0)

    daily = net_of_cost(Wl, rets, cost_bps=cost_bps, name="treasury_auction_concession")
    if getattr(daily, "name", None) is None:
        daily.name = "treasury_auction_concession"
    trades = trades_from_weights(Wl, rets, {e: _ETF_SECTOR[e] for e in etfs})
    return daily, trades


# ---- soft expectations: the proposal's falsifiable mechanism claims --------------------------
def _ann_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _chk_both_tenors(ctx):
    g = ctx.get("grid", {}) or {}
    ief, tlt = g.get("ief_10y"), g.get("tlt_30y")
    if ief is None or tlt is None:
        return {"pass": True, "observed": "sub-book grid variants unavailable"}
    si, st = _ann_sharpe(ief), _ann_sharpe(tlt)
    return {"pass": bool(si > 0 and st > 0),
            "observed": "IEF/10y Sharpe=%.2f, TLT/30y Sharpe=%.2f" % (si, st)}


def _chk_subsample(ctx):
    r = pd.Series(ctx["search"]).dropna()
    cut = pd.Timestamp("2015-01-01")
    sp, sq = _ann_sharpe(r[r.index < cut]), _ann_sharpe(r[r.index >= cut])
    return {"pass": bool(sp > 0 and sq > 0),
            "observed": "pre-2015 Sharpe=%.2f, post-2015 Sharpe=%.2f" % (sp, sq)}


def _event_fwd(panel, holdout_start, with_tail, tenors=(10, 30)):
    hs = pd.Timestamp(holdout_start)
    idx = panel.index
    fwd, tails = [], []
    for t in tenors:
        etf = _TENOR_ETF.get(t)
        if etf is None:
            continue
        ev_c, px_c, tl_c = "ev_%dY" % t, "px_" + etf, "tail_%dY" % t
        if ev_c not in panel.columns or px_c not in panel.columns:
            continue
        px = panel[px_c].values
        for i in np.where(panel[ev_c].fillna(0.0).values > 0)[0]:
            if idx[i] >= hs or i + _HOLD >= len(idx):
                continue
            if pd.isna(px[i]) or pd.isna(px[i + _HOLD]) or px[i] == 0:
                continue
            r = px[i + _HOLD] / px[i] - 1.0
            if with_tail:
                tl = panel[tl_c].iloc[i]
                if pd.isna(tl):
                    continue
                tails.append(float(tl))
            fwd.append(r)
    return np.array(fwd), np.array(tails)


def _chk_event_premium(ctx):
    fwd, _ = _event_fwd(ctx["panel"], ctx["holdout_start"], with_tail=False)
    if len(fwd) < 30:
        return {"pass": True, "observed": "only %d events" % len(fwd)}
    m = float(fwd.mean())
    return {"pass": bool(m > 0),
            "observed": "mean post-auction 5d ETF return=%.4f over %d events" % (m, len(fwd))}


def _chk_weak_auction(ctx):
    fwd, tails = _event_fwd(ctx["panel"], ctx["holdout_start"], with_tail=True)
    if len(tails) < 30:
        return {"pass": True, "observed": "only %d events with tail data" % len(tails)}
    hi, lo = np.quantile(tails, 2 / 3.0), np.quantile(tails, 1 / 3.0)
    weak = float(fwd[tails >= hi].mean())
    strong = float(fwd[tails <= lo].mean())
    return {"pass": bool(weak >= strong),
            "observed": "weak(high-tail) 5d=%.4f vs strong(low-tail) 5d=%.4f" % (weak, strong)}


SPEC = StrategySpec(
    id="treasury_auction_concession",
    family="supply_absorption",
    title="Treasury auction-cycle concession premium — event-window long in duration ETFs after "
          "nominal coupon auctions (FROZEN: 10y/IEF & 30y/TLT, equal-weight, 1x per ETF)",
    markets=["rates", "us_treasury_etf"],
    data_desc="OWNED treasury_auctions() event calendar (auction date + tenor + bid-to-cover/tail, "
              "nominal coupons only; bills/TIPS/FRN/STRIPS excluded) + FREE yfinance daily closes for "
              "SHY/IEI/IEF/TLT (ETFs -> survivorship-clean). 2009-06 -> present.",
    pre_registration=(
        "FROZEN RULE (zero optimization): on each 10-year note auction day buy IEF at that day's close "
        "and hold exactly 5 trading days; on each 30-year bond auction day buy TLT at that day's close "
        "and hold exactly 5 trading days; cash otherwise. Overlapping events within one ETF are unioned "
        "and held at 1x gross PER ETF (equal-weight average of 1x exposures = 1x); IEF and TLT are held "
        "independently at 1x each (no cross-ETF vol normalization), long-only, NO leverage beyond 1x/ETF. "
        "Costs 8bps on turnover. "
        "LOOK-AHEAD SAFETY: auctions conclude ~1:00pm ET and results print immediately, so a same-day "
        "close entry uses only known info; the weight matrix is lagged one day before the kit sees it, "
        "so only returns t+1..t+5 are earned. "
        "DEPLOYMENT NOTE: a 2-name book cannot satisfy single_name_share<=40%. The same frozen rule "
        "applied across the liquid nominal curve (2/3y->SHY, 5/7y->IEI, 10y->IEF, 20/30y->TLT) is offered "
        "ONLY as the pre-declared 'full_curve' grid variant for diversification; the PRIMARY tested book "
        "is the frozen 10y/IEF + 30y/TLT headline. Single-tenor sub-books and a 3-day hold are also "
        "pre-declared grid variants (honest search burden). "
        "MECHANISM (falsifiable): dealers absorb scheduled supply under post-SLR balance-sheet "
        "constraints and shed it over the next days; the price recovery is the liquidity-provision fee "
        "(Lou-Yan-Zhang; Fleming/Rosenberg; Sigaux) — a repeated-shock rent, not a rate forecast. "
        "FALSIFIERS (soft-checked): (1) premium present in BOTH 10y and 30y sub-books; (2) high-tail "
        "(weak) auctions recover >= low-tail (strong) auctions; (3) pre- vs post-2015 both positive; "
        "(4) pooled post-auction 5d ETF return > 0. "
        "VALIDATION: stage-1 gates + the harness MCPT; the proposal's intended additional null is a "
        "calendar-randomization test (same count of random 5-day windows in the same ETFs) to isolate "
        "auction-anchored timing from generic duration beta. "
        "STRESS: holdout 2022+ contains the worst duration bear market on record; any loss must come "
        "from the held windows (duration risk honestly borne), not from construction bugs."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":      {},                                       # FROZEN: 10y/IEF + 30y/TLT (primary)
        "ief_10y":      {"tenors": (10,)},                        # 10y sub-book (both-tenor check)
        "tlt_30y":      {"tenors": (30,)},                        # 30y sub-book (both-tenor check)
        "full_curve":   {"tenors": (2, 3, 5, 7, 10, 20, 30)},     # deployment-diversified variant
        "hold3":        {"hold_days": 3},                         # shorter-window robustness
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=4,
    expectations=[
        {"name": "both_core_tenors_positive",
         "claim": "premium appears in BOTH the 10y/IEF and 30y/TLT sub-books (one-tenor result fails)",
         "check": _chk_both_tenors},
        {"name": "weak_auction_larger_recovery",
         "claim": "high-tail (weak) auctions show >= recovery vs low-tail (strong) — the supply-absorption fingerprint",
         "check": _chk_weak_auction},
        {"name": "subsample_stable_pre_post_2015",
         "claim": "pre-2015 and post-2015 search-window Sharpe are both positive",
         "check": _chk_subsample},
        {"name": "event_premium_positive",
         "claim": "pooled post-auction 5d ETF return is positive in the search window",
         "check": _chk_event_premium},
    ],
)