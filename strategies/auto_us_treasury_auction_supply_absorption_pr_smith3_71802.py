"""
US Treasury auction supply-absorption premium — post-auction duration reversal.

Mechanism (Lou-Yan-Zhang, "anticipated repeated shocks"): primary dealers demand a
yield concession to warehouse fresh duration at scheduled Treasury auctions; the
concession reverses as supply is distributed to end-investors over the following days.
We get PAID to absorb a KNOWN, recurring inventory shock — it is NOT a rates forecast.

FROZEN, fully calendar-driven, zero parameters fit to returns:
  - buy IEF at the CLOSE of each 10y-note auction day, hold exactly 5 trading days, exit at close
  - buy TLT at the CLOSE of each 30y-bond auction day, hold exactly 5 trading days, exit at close
  - cash otherwise (in-market only ~24% of the time per leg)
The ONLY input is the public auction calendar (auction DATE, not announce/settle date — that
is the look-ahead trap for this substrate). No price signal, no threshold, no lookback,
nothing to overfit. Constant 1x notional per active leg (gross ~2x only when 10y+30y overlap).
"""
import re
import numpy as np, pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, us_universe, sf1, yf_panel, fred_series,
                          trend_returns, inv_vol_position, treasury_auctions)
from sdk.universe import sector_universe
from sdk.signal_kit import (xs_zscore, net_of_cost, trades_from_weights, pit_panel,
                            market_regime)

PRICE_COLS = ["IEF", "TLT"]
SECTOR_MAP = {"IEF": "UST_7_10Y", "TLT": "UST_20_30Y"}
_START = "2009-06-01"   # warmup buffer before the 2010 event sample


# ----------------------------------------------------------------------------- helpers
def _pick(lc, cands):
    for c in cands:
        if c in lc:
            return lc[c]
    return None


def _term_years(s):
    s = str(s).lower()
    m = re.search(r'(\d+)\s*-?\s*year', s)
    return (int(m.group(1)) if m else None), ('month' in s)


def _excluded(blob):
    blob = blob.lower()
    return any(x in blob for x in ("tips", "inflation", "frn", "floating",
                                   "bill", "cmb", "strip"))


def _is_10y_note(orig_term, sec_term, sec_type):
    blob = f"{orig_term} {sec_term} {sec_type}"
    if _excluded(blob):
        return False
    t = str(sec_type).lower()
    is_note = ('note' in t) or (t.strip() in ("", "nan"))
    oy, _ = _term_years(orig_term)            # prefer ORIGINAL term (clean for reopenings)
    if oy is not None:
        return is_note and oy == 10
    sy, hm = _term_years(sec_term)
    if sy == 10:
        return is_note
    if sy == 9 and hm:                        # 10y reopening reports "9-Year 1x-Month"
        return is_note
    return False


def _is_30y_bond(orig_term, sec_term, sec_type):
    blob = f"{orig_term} {sec_term} {sec_type}"
    if _excluded(blob):
        return False
    t = str(sec_type).lower()
    is_bond = ('bond' in t) or (t.strip() in ("", "nan"))
    oy, _ = _term_years(orig_term)
    if oy is not None:
        return is_bond and oy == 30           # excludes the 20y bond (reintro 2020)
    sy, hm = _term_years(sec_term)
    if sy == 30:
        return is_bond
    if sy == 29 and hm:                       # 30y reopening reports "29-Year 1x-Month"
        return is_bond
    return False


def _extract_auction_dates(auc):
    df = auc.copy()
    lc = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in df.columns}

    # AUCTION date only — never announce / issue / settle / maturity (gate0 trap #2)
    date_col = _pick(lc, ["auctiondate", "auctiondt", "auctiondatetime"])
    if date_col is None:
        for k, v in lc.items():
            if "auction" in k and "date" in k:
                date_col = v
                break
    if date_col is None:
        for k, v in lc.items():
            if "date" in k and not any(b in k for b in
                                       ("settle", "issue", "matur", "announce",
                                        "dateddate", "firstinterest", "callable")):
                date_col = v
                break
    if date_col is None:
        raise RuntimeError("treasury_auctions(): could not locate an auction-DATE column")

    orig_col = _pick(lc, ["originalsecurityterm", "originalterm", "origterm"])
    term_col = _pick(lc, ["securityterm", "term", "tenor"])
    type_col = _pick(lc, ["securitytype", "type"])

    dates = pd.to_datetime(df[date_col], errors="coerce")
    orig = df[orig_col] if orig_col else pd.Series([""] * len(df), index=df.index)
    term = df[term_col] if term_col else pd.Series([""] * len(df), index=df.index)
    typ = df[type_col] if type_col else pd.Series([""] * len(df), index=df.index)

    ten, thirty = [], []
    for i in df.index:
        d = dates.loc[i]
        if pd.isna(d):
            continue
        if _is_10y_note(orig.loc[i], term.loc[i], typ.loc[i]):
            ten.append(pd.Timestamp(d).normalize())
        elif _is_30y_bond(orig.loc[i], term.loc[i], typ.loc[i]):
            thirty.append(pd.Timestamp(d).normalize())

    ten = pd.DatetimeIndex(sorted(set(ten)))
    thirty = pd.DatetimeIndex(sorted(set(thirty)))
    if len(ten) < 30 or len(thirty) < 30:     # ~190+ expected each since 2010 -> parse failure
        raise RuntimeError(f"auction parse too sparse: 10y={len(ten)} 30y={len(thirty)} "
                           f"(check term/type schema of treasury_auctions())")
    return ten, thirty


def _load_prices(tickers, start):
    tickers = list(tickers)

    def ok(p):
        if p is None or not hasattr(p, "columns"):
            return False
        for t in tickers:
            if t not in p.columns or p[t].dropna().shape[0] < 200:
                return False
        return True

    px = None
    try:                                      # PREFER Sharadar (survivorship-clean, ETF-covered)
        px = sep_panel(tickers, start=start, field="closeadj")
    except Exception:
        px = None
    if not ok(px):                            # yfinance cache fallback (proposal-sanctioned)
        try:
            px = yf_panel(tickers, start=start)
        except Exception:
            px = None
    if not ok(px):
        raise RuntimeError("could not load IEF/TLT closes from sep_panel or yf_panel")
    return px.reindex(columns=tickers).astype(float).dropna(how="all")


def _stamp_events(panel, col, dates):
    idx = panel.index
    if len(dates) == 0:
        return
    dates = dates[(dates >= idx[0]) & (dates <= idx[-1])]
    if len(dates) == 0:
        return
    pos = idx.searchsorted(dates, side="left")   # auctions are on business days -> exact hit
    pos = np.unique(pos[pos < len(idx)])
    panel.iloc[pos, panel.columns.get_loc(col)] = 1.0


# ----------------------------------------------------------------------------- load_data
def load_data() -> pd.DataFrame:
    auc = treasury_auctions()
    ten, thirty = _extract_auction_dates(auc)
    px = _load_prices(PRICE_COLS, _START)

    panel = px.copy()
    panel["IEF_event"] = 0.0    # 1.0 on 10y-note auction days
    panel["TLT_event"] = 0.0    # 1.0 on 30y-bond auction days
    _stamp_events(panel, "IEF_event", ten)
    _stamp_events(panel, "TLT_event", thirty)
    return panel


# ----------------------------------------------------------------------------- signal
def signal(panel, hold_days=5, legs=("IEF", "TLT"), **params):
    """
    Event-driven, calendar-only, FROZEN. The ONLY input is the auction calendar — no price
    signal, no lookback, no vol sizing (the frozen design is explicit: 'no lookbacks, no
    signals on prices at all'). Constant 1x notional per leg while the 5-day event window is
    active; cash otherwise. Overlapping windows of the SAME leg do not pyramid (capped 1x);
    gross reaches ~2x only when a 10y and a 30y window overlap — exactly the proposal cap.

    LAG: weights are formed at the auction-day CLOSE (same-day), so the held weight matrix is
    Wsd.shift(1) — the position earns from the day AFTER the auction onward (the auction print
    at ~1pm ET precedes the 4pm close, so the close entry is information-clean). The shift is
    our responsibility (net_of_cost / trades_from_weights receive the already-lagged W).
    """
    hold_days = int(hold_days)
    legs = tuple(legs)
    px = panel[PRICE_COLS].astype(float)
    rets_filled = px.pct_change().fillna(0.0)
    idx = px.index
    n = len(idx)

    events = {"IEF": panel["IEF_event"].fillna(0.0).to_numpy() > 0,
              "TLT": panel["TLT_event"].fillna(0.0).to_numpy() > 0}

    Wsd = pd.DataFrame(0.0, index=idx, columns=PRICE_COLS)
    for leg in PRICE_COLS:
        if leg not in legs:
            continue
        active = np.zeros(n, dtype=bool)
        for i in np.where(events[leg])[0]:
            active[i:min(i + hold_days, n)] = True   # entry day + (hold-1); overlaps EXTEND, no pyramid
        Wsd[leg] = np.where(active, 1.0, 0.0)        # constant 1x notional per leg — zero price input

    W_held = Wsd.shift(1).fillna(0.0)                # <- the mandatory 1-day lag

    daily = net_of_cost(W_held, rets_filled, cost_bps=8.0, name="ust_auction_absorption")
    trades = trades_from_weights(W_held, rets_filled, SECTOR_MAP)  # auto-stamps entry_regime
    return daily, trades


# ----------------------------------------------------------------------------- gen data
def load_gen_data(label) -> pd.DataFrame:
    # scope='local': there is no disjoint generalization universe (the premium is specific to
    # the US Treasury issuance mechanism; bund/JGB auctions are not in owned data). Not used.
    raise NotImplementedError("scope='local' — no generalization universes")


# ----------------------------------------------------------------------------- soft checks
def _slice_lt(s, holdout_start):
    return s[s.index < pd.Timestamp(holdout_start)]


def _sharpe(x):
    x = x.fillna(0.0)
    sd = float(x.std())
    return float(x.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0


def _check_mostly_cash(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    panel = ctx["panel"]
    total = int((panel.index < hs).sum())
    held = sum(int(t.get("hold_days", 0)) for t in ctx["trades"]
               if pd.Timestamp(t["entry_date"]) < hs)
    frac = held / (2.0 * total) if total else 1.0   # mean per-leg time-in-market (2 legs)
    return {"pass": bool(frac < 0.35), "observed": round(frac, 3)}


def _check_ief_leg_positive(ctx):
    r, _ = signal(ctx["panel"], legs=("IEF",))       # one extra signal() call, sliced to search
    r = _slice_lt(r, ctx["holdout_start"]).fillna(0.0)
    cum = float((1.0 + r).prod() - 1.0)
    return {"pass": bool(cum > 0.0), "observed": round(cum, 4)}


def _check_tlt_leg_positive(ctx):
    r, _ = signal(ctx["panel"], legs=("TLT",))       # one extra signal() call, sliced to search
    r = _slice_lt(r, ctx["holdout_start"]).fillna(0.0)
    cum = float((1.0 + r).prod() - 1.0)
    return {"pass": bool(cum > 0.0), "observed": round(cum, 4)}


def _check_placebo_weaker(ctx):
    # Calendar-shifted null (proposal generalization_plan (b)): shift the SAME rule +9 business
    # days off the real auctions; the edge must be anchored to the event, not generic carry.
    panel = ctx["panel"].copy()
    for col in ("IEF_event", "TLT_event"):
        ev = panel[col].fillna(0.0).to_numpy()
        out = np.zeros_like(ev)
        on = np.where(ev > 0)[0] + 9
        on = on[on < len(ev)]
        out[on] = 1.0
        panel[col] = out
    r_plac, _ = signal(panel)                        # one extra signal() call
    r_plac = _slice_lt(r_plac, ctx["holdout_start"])
    rs = _sharpe(_slice_lt(ctx["search"], ctx["holdout_start"]))
    ps = _sharpe(r_plac)
    return {"pass": bool(rs > ps), "observed": f"real={rs:.2f} placebo={ps:.2f}"}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="ust_auction_absorption_v1",
    family="rates_supply_absorption",
    title=("US Treasury auction supply-absorption premium — 10y/30y post-auction duration "
           "reversal (IEF/TLT, auction-day entry, 5-day hold, flat otherwise)"),
    markets=["us_treasury_etf"],
    data_desc=("treasury_auctions() public auction CALENDAR (10y-note + 30y-bond auctions incl. "
               "reopenings, ~190+ events/leg since 2010; original-term parse, TIPS/FRN/20y excluded). "
               "IEF & TLT daily closeadj via sep_panel (Sharadar) with yfinance cache fallback. $0."),
    pre_registration=(
        "FROZEN calendar rule, zero parameters fit to returns and zero price inputs. Long IEF at the "
        "CLOSE of each 10y note auction day, long TLT at the CLOSE of each 30y bond auction day; hold "
        "EXACTLY 5 trading days then exit; cash otherwise. CONSTANT 1x notional per active leg (no vol "
        "sizing, no lookback — the only input is the auction calendar). Weights are formed same-day at "
        "the close and lagged one day (W.shift(1)) so PnL accrues from the day AFTER the auction — no "
        "look-ahead. The auction date is the only input; knowing the date at the close is public "
        "(announced ~1wk ahead) and the result prints ~1pm ET < the 4pm close, so the close entry is "
        "information-clean. We deliberately use the AUCTION date, never announce/issue/settle (the "
        "documented look-ahead trap for this substrate). Gross ~2x only in refunding weeks when 10y+30y "
        "windows overlap — exactly the proposal cap. 8bps/turnover. The ONE discretionary choice is the "
        "5-day hold; its honest search burden is declared in the grid (hold in {3,5,7}). PRE-REGISTERED "
        "PASS CRITERIA: (a) standalone search-window edge net of cost; (b) the +9-business-day "
        "calendar-shifted placebo is weaker (machine-checked) — return anchored to the event, not "
        "duration carry; (c) BOTH the 10y-only and 30y-only sub-books must be independently positive "
        "(machine-checked) — a one-tenor result is a FAIL; (d) on the 2022-01-01+ HOLDOUT (the 2022-23 "
        "hiking cycle when outright duration LOST money) the event alpha must survive even if smaller, "
        "since the claim is a SUPPLY effect, not a duration bet — validated by the harness on the "
        "holdout, not as a soft check. DISCLOSURE: this is a deliberately concentrated 2-asset MACRO "
        "book (deploy_max_positions=2); single_name_share is ~50% by construction and is NOT a hidden "
        "single-name bet — it is the economic design (two duration instruments, one per auctioned "
        "tenor). LOCAL scope: the premium is specific to the US Treasury issuance mechanism; nearest "
        "analogues (bund/JGB auctions) are not in owned data, so confirmation is forward-paper."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}, "hold3": {"hold_days": 3}, "hold7": {"hold_days": 7}},
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=2,
    expectations=[
        {"name": "mostly_in_cash",
         "claim": "mean per-leg time-in-market < 35% of trading days (event-driven, flat otherwise)",
         "check": _check_mostly_cash},
        {"name": "ief_10y_leg_positive",
         "claim": "the 10y-only sub-book (IEF) is independently positive over the search window",
         "check": _check_ief_leg_positive},
        {"name": "tlt_30y_leg_positive",
         "claim": "the 30y-only sub-book (TLT) is independently positive over the search window",
         "check": _check_tlt_leg_positive},
        {"name": "placebo_calendar_shift_weaker",
         "claim": "real auction-window Sharpe exceeds a +9-business-day calendar-shifted placebo "
                  "(edge anchored to the auction, not generic duration carry)",
         "check": _check_placebo_weaker},
    ],
)