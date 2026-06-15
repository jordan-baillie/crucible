"""Tested data adapters the agent's generated signal code composes (reliability > reinvention).
All no-incremental-cost / already-owned sources. The agent is told to use THESE, not raw downloads."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def _day_cache(kind: str, key_parts):
    """Per-day parquet cache path for network adapters (yfinance/FRED): the same request twice in
    one day (3 smiths, batteries, MCPT setup) = one download. Keyed by request + date, so it's
    never stale by more than a day and never wrong. Returns None if the cache dir is unwritable."""
    import hashlib
    from crucible_paths import DATA
    h = hashlib.sha256(repr(sorted(map(str, key_parts))).encode()).hexdigest()[:16]
    d = os.path.join(str(DATA), "cache", "net")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return os.path.join(d, f"{kind}_{h}_{pd.Timestamp.today():%Y%m%d}.parquet")


def yf_panel(tickers, start="2005-01-01") -> pd.DataFrame:
    """Close panel for a list of yfinance tickers (business-day grid). Handles single/multi shapes.
    Day-cached on disk (E4): repeated calls across smiths/batteries hit Yahoo once per day."""
    import yfinance as yf
    cache = _day_cache("yf", [*tickers, start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    raw = yf.download(list(tickers), start=start, progress=False, group_by="ticker", auto_adjust=False)
    cols = {}
    for t in tickers:
        try:
            s = raw[t]["Close"].dropna() if t in raw.columns.get_level_values(0) else None
        except Exception:
            s = None
        if s is None:
            try:
                c = raw["Close"]; s = (c[t] if t in getattr(c, "columns", []) else c).dropna()
            except Exception:
                s = None
        if s is not None and len(s) > 200:
            cols[t] = s
    panel = pd.DataFrame(cols).sort_index()
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(bidx).ffill(limit=3)
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


# --- FRED series provenance (pre-reg: macro-neutralization gate §4) ----------------------------
# MARKET-OBSERVED = a price/yield/spread/fx/implied-vol the market PRINTS; never revised, so using
# its as-reported value historically is exact (no look-ahead). REVISED RELEASES = estimated by an
# agency and restated later (GDP/CPI/payrolls); feeding their LATEST value into a backtest signal is
# look-ahead bias -> use point-in-time vintages (FRED ALFRED). Strict mode bans the latter.
MARKET_OBSERVED_FRED = {
    # Treasury constant-maturity yields + curve spreads + TIPS real yields/breakevens
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
    "T10Y2Y", "T10Y3M", "T5YIE", "T10YIE", "T5YIFR", "DFII5", "DFII10", "DFII30",
    # Corporate bond yields/spreads (Moody's daily; ICE/FRED spreads are market-derived)
    "DBAA", "DAAA", "BAA", "AAA", "BAA10Y", "AAA10Y",
    # Money-market / policy rates (observed)
    "DFF", "EFFR", "OBFR", "SOFR", "DPRIME", "DCPF3M", "DCPN3M",
    # Trade-weighted dollar + bilateral FX (market prices)
    "DTWEXBGS", "DTWEXAFEGS", "DTWEXEMEGS",
    "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXCHUS", "DEXMXUS", "DEXKOUS",
    # Commodity spot
    "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP",
    # CBOE implied-vol indices
    "VIXCLS", "VXVCLS", "VXDCLS", "OVXCLS", "GVZCLS",
}
REVISED_FRED_RELEASES = {
    "GDP", "GDPC1", "GDPPOT", "A191RL1Q225SBEA", "PCEC96", "PCE", "PI", "DSPIC96",
    "CPIAUCSL", "CPILFESL", "CPIAUCNS", "PCEPI", "PCEPILFE",
    "UNRATE", "U6RATE", "PAYEMS", "CIVPART", "ICSA",
    "INDPRO", "TCU", "RSAFS", "RSXFS", "HOUST", "PERMIT", "UMCSENT", "M1SL", "M2SL",
}


def _check_fred_ids(ids, allow_revised: bool):
    """Provenance guard for fred_series. Default path WARNS (stderr) on known revised releases.
    Strict mode (allow_revised=False, used by the macro-neutralization factor loader) RAISES unless
    every id is market-observed -> the look-ahead path is structurally absent, not merely discouraged."""
    revised = [s for s in ids if s in REVISED_FRED_RELEASES]
    if revised and allow_revised:
        print(f"[fred_series] LOOK-AHEAD WARNING: {revised} are REVISED macro releases; latest-revised "
              f"values are look-ahead bias if fed into a signal. OK for diagnostics; for signals use "
              f"ALFRED vintages (prereg-macro-neutralization-gate.md §4).", file=sys.stderr)
    if not allow_revised:
        bad = [s for s in ids if s not in MARKET_OBSERVED_FRED]
        if bad:
            raise ValueError(
                f"fred_series(allow_revised=False): {bad} not in the market-observed allowlist "
                f"(revised releases or unknown). Add genuinely market-observed ids to "
                f"MARKET_OBSERVED_FRED, or use ALFRED vintages for revised releases "
                f"(pre-reg: macro-neutralization gate §4).")


def fred_series(series_ids, start="2005-01-01", allow_revised: bool = True) -> pd.DataFrame:
    """Daily-ffilled FRED series panel. series_ids: dict {fred_id: column_name} or list.
    Day-cached on disk (E4).

    allow_revised (default True = legacy behaviour): when False (strict; used by the macro-
    neutralization factor loader) only MARKET_OBSERVED_FRED ids are permitted — revised/released
    macro series raise (look-ahead risk; use ALFRED vintages). See macro-neutralization gate §4."""
    import urllib.request
    from crucible_paths import SECRETS
    _ids = list(series_ids) if isinstance(series_ids, (list, tuple)) else list(series_ids.keys())
    _check_fred_ids(_ids, allow_revised)
    cache = _day_cache("fred", [*(series_ids if isinstance(series_ids, (list, tuple))
                                  else sorted(series_ids.items())), start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    key = os.environ.get("FRED_API_KEY") or json.load(open(SECRETS))["fred_api_key"]
    if isinstance(series_ids, (list, tuple)):
        series_ids = {s: s for s in series_ids}
    out = {}
    for sid, col in series_ids.items():
        u = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
             f"&api_key={key}&file_type=json&observation_start={start}")
        d = json.load(urllib.request.urlopen(u, timeout=40))
        out[col] = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                              for o in d["observations"] if o["value"] != "."}).sort_index()
    df = pd.DataFrame(out).sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B")).ffill()
    if cache:
        try:
            tmp = cache + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return df


# --- Macro-neutralization factor block (pre-reg: macro-neutralization gate, FROZEN 2026-06-15) ---
_MACRO_FRED = {  # FRED id -> temp column; ALL market-observed (never revised)
    "DGS10": "_dgs10", "T10Y2Y": "_slope", "T10YIE": "_be",
    "DBAA": "_baa", "DAAA": "_aaa", "DTWEXBGS": "_usd", "DCOILWTICO": "_oil", "VIXCLS": "_vix",
}


def macro_factor_returns(start="2003-01-01", include_crypto=False, crypto_market="spot") -> pd.DataFrame:
    """Daily MACRO FACTOR RETURNS for the neutralization gate (8 factors; +BTC/ETH if include_crypto).
    Every input is market-observed & never revised -> look-ahead-free by construction (pre-reg §4).
    Columns: dur, slope, breakeven, credit, usd, oil, gold, vol [, btc, eth]. NaN where a series does
    not yet cover a date (the harness applies the coverage/obs evaluability guard). Underlying pulls
    are day-cached, so repeated calls in a run hit disk once."""
    lv = fred_series(_MACRO_FRED, start=start, allow_revised=False)  # strict: allowlisted only
    f = pd.DataFrame(index=lv.index)
    f["dur"]       = -7.5 * (lv["_dgs10"] / 100.0).diff()         # -Dur*dy (first-order duration return)
    f["slope"]     = (lv["_slope"] / 100.0).diff()                # curve twist (10y-2y)
    f["breakeven"] = (lv["_be"] / 100.0).diff()                   # inflation compensation
    f["credit"]    = -((lv["_baa"] - lv["_aaa"]) / 100.0).diff()  # risk-on = Baa-Aaa spread tightening
    f["usd"]       = np.log(lv["_usd"]).diff()                    # broad dollar
    f["oil"]       = np.log(lv["_oil"].where(lv["_oil"] > 0)).diff()  # WTI (guard the 2020 negative print)
    f["vol"]       = (lv["_vix"] / 100.0).diff()                  # equity implied vol (risk-off)
    # gold via GLD ETF (FRED free gold-fix discontinued); ETF closes are market-observed/non-revised
    try:
        gld = yf_panel(["GLD"], start=start)["GLD"]
        f["gold"] = np.log(gld).diff().reindex(f.index)
    except Exception:
        f["gold"] = np.nan
    if include_crypto:
        try:
            kl = binance_klines(["BTCUSDT", "ETHUSDT"], market=crypto_market,
                                start=max(str(start), "2017-01-01"))
            for sym, col in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
                f[col] = (np.log(kl[(sym, "close")].astype(float)).diff().reindex(f.index)
                          if (sym, "close") in kl.columns else np.nan)
        except Exception:
            f["btc"] = np.nan
            f["eth"] = np.nan
    return f


def trend_returns(**params):
    """The validated Boreas 21-market TSMOM stream (daily_returns, trades) — a ready hedge leg."""
    sys.path.insert(0, os.environ.get("BOREAS_RESEARCH", "/root/boreas/research"))
    from tsmom import run_tsmom
    return run_tsmom(**params)


# carry_returns() removed 2026-06-10: Midas project killed, its data pipeline is gone.
# The carry+trend STRUCTURE remains validated knowledge (see wiki) — re-add only with a
# new carry leg + fresh forward validation.

def inv_vol_position(signal_df, rets, target_vol=0.10, vol_lb=60, max_pos=2.0, rebalance="W-FRI"):
    """Standard inverse-vol sizing + weekly hold + 1d lag (no look-ahead). Reusable building block."""
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    raw = (signal_df * (target_vol / vol.replace(0, np.nan))).clip(-max_pos, max_pos)
    return raw.resample(rebalance).last().reindex(rets.index, method="ffill").shift(1).fillna(0.0)


# ── Owned Sharadar — SURVIVORSHIP-CLEAN US equities (PREFER over yf_panel for US stocks) ──
from crucible_paths import DATA
SHARADAR_DIR = str(DATA / "sharadar")
_CACHE_DIR = str(DATA / "cache")


def _stale(out: str, src: str) -> bool:
    """Cache invalidation (E4): a fresh source drop must rebuild the derived parquet."""
    return (not os.path.exists(out)) or (os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(out))


def _sep_cache() -> str:
    """Build/load a cached long parquet of owned Sharadar SEP (one-time ~1-2min build).
    Rebuilds automatically when SEP.zip is newer than the cache (fresh data drop)."""
    import zipfile
    # v2: + close/closeunadj (ex-div inference needs an unadjusted companion — runtime_error 2026-06-10).
    # Versioned filename forces a rebuild on schema change (_stale only checks mtime, not columns).
    out = os.path.join(_CACHE_DIR, "sep_long_v2.parquet")
    src = os.path.join(SHARADAR_DIR, "SEP.zip")
    if _stale(out, src):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(src)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=["ticker", "date", "close", "closeadj", "closeunadj", "volume"],
                             parse_dates=["date"])
        tmp = out + ".tmp"
        df.sort_values("ticker").to_parquet(tmp, index=False, row_group_size=500_000)
        os.replace(tmp, out)
    return out


def sep_panel(tickers=None, start="2000-01-01", end=None, field="closeadj") -> pd.DataFrame:
    """SURVIVORSHIP-CLEAN US equity daily panel from OWNED Sharadar SEP (delisted incl, split+div adj).
    Wide DataFrame: business-day DatetimeIndex x ticker of `field`
    (closeadj=adjusted close | closeunadj | close | volume).
    PREFER THIS over yf_panel for US stocks (yfinance has survivorship bias — a wiki anti-pattern).
    tickers=None loads ALL (~16k, heavy) — pass a list (e.g. from us_universe())."""
    path = _sep_cache()
    # E3: push date bounds into the parquet scan (row-group pruning) instead of filtering in pandas
    filt = [("date", ">=", pd.Timestamp(start))]
    if end is not None:
        filt.append(("date", "<=", pd.Timestamp(end)))
    if tickers is not None:
        filt.append(("ticker", "in", list(tickers)))
    df = pd.read_parquet(path, columns=["ticker", "date", field], filters=[filt])
    panel = df.pivot_table(index="date", columns="ticker", values=field).sort_index()
    if panel.empty:
        return panel
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    return panel.reindex(bidx).ffill(limit=3)


def us_universe(sector=None, category="Domestic Common Stock", marketcap=None,
                include_delisted=True, top_n=None) -> list:
    """US-equity universe from OWNED Sharadar TICKERS (survivorship-clean — DELISTED incl by default).
    Filter by `sector` (e.g. 'Financial Services'), `category` (default common stock), `marketcap`
    scale substring (e.g. 'Large','Mid','Small'). **`top_n` bounds to the N MOST-LIQUID names**
    (recent median dollar volume) — USE THIS for cross-sectional equity: the full ~16k universe is
    too slow/memory-heavy for the CPCV rails (a run OOM'd at 14.5GB). Returns ticker list."""
    import glob
    p = glob.glob(os.path.join(SHARADAR_DIR, "SHARADAR_TICKERS_*.csv"))[0]
    tk = pd.read_csv(p, usecols=["ticker", "category", "sector", "isdelisted", "scalemarketcap"])
    tk = tk[tk["ticker"].notna()]
    if category:
        tk = tk[tk["category"].fillna("").str.contains(category, case=False, regex=False)]
    if sector:
        tk = tk[tk["sector"] == sector]
    if marketcap:
        tk = tk[tk["scalemarketcap"].fillna("").str.contains(marketcap, case=False, regex=False)]
    if not include_delisted:
        tk = tk[tk["isdelisted"] == "N"]
    names = sorted(tk["ticker"].dropna().unique().tolist())
    if top_n and len(names) > top_n:
        # drop the untradable nano/micro tail cheaply first so the price load stays sane,
        # then keep the top_n by recent median dollar volume (close*volume).
        if len(names) > 3000 and not marketcap:
            tk = tk[~tk["scalemarketcap"].fillna("").str.contains("Nano|Micro", case=False, regex=True)]
            names = sorted(tk["ticker"].dropna().unique().tolist())
        # E3: ONE parquet scan for both fields; then the EXACT sep_panel grid transform
        # (business-day reindex + ffill(limit=3)) so the liquidity ranking is byte-identical
        # to the historical two-call path (universe selection is part of frozen pre-regs).
        path = _sep_cache()
        df = pd.read_parquet(path, columns=["ticker", "date", "closeadj", "volume"],
                             filters=[[("date", ">=", pd.Timestamp("2015-01-01")),
                                       ("ticker", "in", names)]])

        def _grid(field):
            panel = df.pivot_table(index="date", columns="ticker", values=field).sort_index()
            bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
            return panel.reindex(bidx).ffill(limit=3)

        dollar = (_grid("closeadj") * _grid("volume")).tail(252).median().dropna()
        names = sorted(dollar.nlargest(min(top_n, len(dollar))).index.tolist())
    return names


def _sf1_cache() -> str:
    import zipfile
    out = os.path.join(_CACHE_DIR, "sf1_long.parquet")
    src = os.path.join(SHARADAR_DIR, "SF1.zip")
    if _stale(out, src):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(src)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, parse_dates=["datekey", "calendardate"], low_memory=False)
        tmp = out + ".tmp"
        df.sort_values("ticker").to_parquet(tmp, index=False, row_group_size=200_000)
        os.replace(tmp, out)
    return out


def sf1(tickers, fields=None, dimension="ARQ") -> pd.DataFrame:
    """OWNED Sharadar SF1 fundamentals (point-in-time). Long DataFrame [ticker, datekey, calendardate, <fields>]
    for `dimension` (ARQ=as-reported quarterly, MRQ=most-recent quarterly, ART=trailing-twelve as-reported).
    IMPORTANT: use `datekey` (the FILING/available date) — NOT calendardate — as the as-of date to avoid look-ahead."""
    path = _sf1_cache()
    cols = (["ticker", "datekey", "calendardate", "dimension"] + list(fields)) if fields else None
    df = pd.read_parquet(path, columns=cols,
                         filters=[("ticker", "in", list(tickers)), ("dimension", "==", dimension)])
    return df.sort_values(["ticker", "datekey"]).reset_index(drop=True)


# ── Databento GLBX — individual futures CONTRACT MONTHS (basis-momentum substrate) ──
DATABENTO_DIR = str(DATA / "databento")
_MONTH_CODE = {m: i + 1 for i, m in enumerate("FGHJKMNQUVXZ")}


def fut_curve(root, n_contracts=2, min_volume=1, field=None):
    """Daily futures CURVE panel from owned Databento GLBX daily bars (one-time pull 2026-06-12,
    17 commodity roots, all contract months, 2010+). For each business day, ranks the OUTRIGHT
    contracts that actually traded (volume >= min_volume) by expiry and returns the nearest
    `n_contracts`: columns close_1..n, volume_1..n, symbol_1..n, days_to_roll_1 (days until the
    front contract's last trade — for roll-aware execution).

    field="ret" (added 2026-06-14): the roll-SAFE within-contract front-month return — the one
    thing generated strategies reach for, and the #1 way to get futures wrong (diffing close_1
    across a roll contaminates the return with the roll gap). Pass a single root -> returns a
    Series ({root}_ret); pass a list/tuple of roots -> returns a wide DataFrame (date x root).
    Bare-string-safe. Returns at each roll boundary are NaN (a new front contract starts a new
    group), never bridging two contracts. Use this instead of computing returns by hand.

    THE point of this dataset: basis-momentum (Boons-Prado 2019) and curve signals need the first
    AND second contract separately; stitched continuous series (yf_panel) cannot express them.

    Gotchas handled here so generated code doesn't have to:
    - spread/butterfly symbols (e.g. 'CLN6-CLQ6', 'CL:BF ...') are EXCLUDED (outrights only);
    - single-digit year codes wrap each decade AND one symbol can be TWO contracts (CLZ0 rows in
      2010 = Dec-2010; CLZ0 rows in 2012+ = Dec-2020, listed ~9y out) — disambiguated per
      Databento instrument_id using each instrument's LAST trade date (≈ expiry month);
    - returns are NOT roll-adjusted: compute returns WITHIN a contract (groupby symbol) or use
      rank-1/rank-2 series with roll-day awareness — never diff close_1 across a roll naively."""
    import re
    if field is not None:
        if field != "ret":
            raise ValueError(f"fut_curve: field must be None or 'ret', got {field!r}")
        roots_list = [root] if isinstance(root, str) else list(root)
        cols = {}
        for rt in roots_list:
            cur = fut_curve(rt, n_contracts=n_contracts, min_volume=min_volume)  # field=None -> curve
            # roll-safe: group by CONTIGUOUS front-contract spans, not the symbol string — symbols
            # recycle each decade (CLZ0 = Dec-2010 AND Dec-2020), so groupby(symbol_1) would bridge
            # a 10y gap. cumsum on symbol-change gives a fresh id per span; pct_change within span
            # makes the first bar of every new front contract NaN, so the roll gap never enters.
            sym = cur["symbol_1"]
            span = sym.ne(sym.shift()).cumsum()
            px = cur["close_1"].astype(float)
            r = px.groupby(span).pct_change()
            # negative/zero price base breaks pct_change (WTI settled -$2.67 on 2020-04-20 — a
            # REAL event, not bad data): a % return off a non-positive base is undefined and
            # explodes (+439%). Mask those to NaN; the prior bar's price must be > 0.
            r = r.where(px.groupby(span).shift() > 0)
            cols[rt] = r
        panel = pd.DataFrame(cols).sort_index()
        return panel[roots_list[0]].rename(f"{root}_ret") if isinstance(root, str) else panel
    cache = os.path.join(_CACHE_DIR, f"futcurve_{root}_{n_contracts}_{min_volume}.parquet")
    src = os.path.join(DATABENTO_DIR, f"{root}_ohlcv1d.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} — root '{root}' not in the owned Databento pull "
                                f"(see wiki DATA_CATALOG; re-quote with metadata.get_cost before adding).")
    if not _stale(cache, src):
        return pd.read_parquet(cache)
    df = pd.read_parquet(src, columns=["symbol", "close", "volume", "instrument_id"])
    df["date"] = df.index.tz_localize(None).normalize()
    df = df.reset_index(drop=True)  # ts_event index has duplicate labels (one row per contract per day)
    pat = re.compile(rf"^{re.escape(root)}([FGHJKMNQUVXZ])(\d{{1,2}})$")
    m = df["symbol"].str.extract(pat)
    df = df[m[0].notna() & (df["volume"] >= min_volume)].copy()
    df["_mon"] = df["symbol"].str.extract(pat)[0].map(_MONTH_CODE)
    df["_yd"] = df["symbol"].str.extract(pat)[1].astype(int)
    # expiry per INSTRUMENT (not symbol — symbols recycle): last trade date ≈ expiry month, so the
    # expiry year is the year (last_trade_year-1 .. +1) whose last digit(s) match the year code.
    last_trade = df.groupby("instrument_id")["date"].transform("max")
    lty = last_trade.dt.year
    yd = df["_yd"]
    exp_year = np.where(yd >= 10, 2000 + yd,
                        lty + ((yd - lty % 10 + 5) % 10) - 5)  # nearest year (±5) with matching digit
    df["_exp"] = exp_year * 100 + df["_mon"]
    df["_dtr"] = (last_trade - df["date"]).dt.days
    df = df.sort_values(["date", "_exp", "_dtr"])
    df = df.drop_duplicates(["date", "_exp"])  # rare dual-listing duplicates: keep nearer-expiry instrument
    df["_rank"] = df.groupby("date").cumcount() + 1
    df = df[df["_rank"] <= n_contracts]
    out = df.pivot(index="date", columns="_rank", values=["close", "volume", "symbol"])
    out.columns = [f"{f}_{r}" for f, r in out.columns]
    dtr = df[df["_rank"] == 1].set_index("date")["_dtr"]
    out["days_to_roll_1"] = dtr
    out = out.sort_index()
    try:
        tmp = cache + ".tmp"
        out.to_parquet(tmp)
        os.replace(tmp, cache)
    except OSError:
        pass
    return out


# ── Free public sources unblocking queued families (COT / CBOE / funding / auctions) ──

def _http_get(url, timeout=60, retries=3):
    """GET with a browser-ish User-Agent (CFTC 403s python-urllib) + simple retry (Binance
    public API is transiently flaky). Returns bytes."""
    import time as _t, urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; crucible)"})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if attempt == retries - 1:
                raise
            _t.sleep(5 * (attempt + 1))

# CFTC contract market codes for our futures roots (CME/NYMEX/COMEX/CBOT contracts —
# verified 2026-06-12 against deacot2024; deliberately NOT the ICE lookalikes).
_COT_CODES = {
    "CL": "067651", "NG": "023651", "HO": "022651", "RB": "111659",
    "GC": "088691", "SI": "084691", "HG": "085692", "PL": "076651", "PA": "075651",
    "ZC": "002602", "ZS": "005602", "ZW": "001602", "ZL": "007601", "ZM": "026603",
    "LE": "057642", "HE": "054642", "GF": "061641",
}


def cot_positioning(roots=None, start_year=2010) -> pd.DataFrame:
    """CFTC Commitments of Traders (legacy futures-only) weekly positioning panel, free.
    Columns per root: {root}_comm_net (commercial net = hedgers), {root}_noncomm_net
    (speculators), {root}_oi (open interest). LOOK-AHEAD DISCIPLINE: indexed by the
    RELEASE date (as-of Tuesday + 3 days = Friday 15:30 ET publication) — a backtest may
    use a row from its index date onward, never from the as-of Tuesday.
    Hedging-pressure signal (Basu-Miffre): comm_net / oi (more short = more hedging pressure)."""
    import io, zipfile
    if isinstance(roots, str):  # footgun caught live 2026-06-13: smith passed "CL" -> iterated to 'C','L' -> KeyError
        roots = [roots]
    roots = list(roots or _COT_CODES)
    this_year = pd.Timestamp.today().year
    cache = _day_cache("cot", [*roots, start_year, this_year])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    frames = []
    for yr in range(start_year, this_year + 1):
        u = f"https://www.cftc.gov/files/dea/history/deacot{yr}.zip"
        try:
            raw = _http_get(u, timeout=60)
            z = zipfile.ZipFile(io.BytesIO(raw))
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue  # current year may not exist yet early in Jan
        df.columns = [c.strip() for c in df.columns]
        df = df[df["CFTC Contract Market Code"].astype(str).str.strip().isin(
            {_COT_CODES[r] for r in roots})]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    code2root = {_COT_CODES[r]: r for r in roots}
    df["_root"] = df["CFTC Contract Market Code"].astype(str).str.strip().map(code2root)
    df["_asof"] = pd.to_datetime(df["As of Date in Form YYYY-MM-DD"])
    df["_release"] = df["_asof"] + pd.Timedelta(days=3)  # Tue data -> Fri publication
    out = {}
    for root, g in df.groupby("_root"):
        g = g.set_index("_release").sort_index()
        comm = g["Commercial Positions-Long (All)"] - g["Commercial Positions-Short (All)"]
        nonc = g["Noncommercial Positions-Long (All)"] - g["Noncommercial Positions-Short (All)"]
        out[f"{root}_comm_net"], out[f"{root}_noncomm_net"] = comm, nonc
        out[f"{root}_oi"] = g["Open Interest (All)"]
    panel = pd.DataFrame(out).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def cboe_index(names=("VIX3M", "VVIX", "SKEW", "PUT")) -> pd.DataFrame:
    """CBOE published index history (free CDN CSVs), daily CLOSE panel on a business-day grid.
    Depth: VIX3M 2009+, VVIX 2006+, SKEW 1990+, PUT 1991+. Spot VIX itself: use FRED VIXCLS.
    Canonical contango regime signal: VIXCLS / VIX3M (backwardation when > ~1.0)."""
    import io
    if isinstance(names, str):  # footgun class (cf. cot_positioning 2026-06-13): "VVIX" -> ['V','V','I','X']
        names = [names]
    names = list(names)
    cache = _day_cache("cboe", names)
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    out = {}
    for n in names:
        u = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{n}_History.csv"
        df = pd.read_csv(io.BytesIO(_http_get(u, timeout=60)))
        df.columns = [c.strip().upper() for c in df.columns]
        col = "CLOSE" if "CLOSE" in df.columns else n.upper()
        out[n] = pd.Series(df[col].values, index=pd.to_datetime(df["DATE"])).sort_index()
    panel = pd.DataFrame(out).sort_index()
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(bidx).ffill(limit=3)
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def funding_rates(symbols=("BTCUSDT", "ETHUSDT"), source="binance") -> pd.DataFrame:
    """Perp funding-rate history (free public APIs), DAILY sum of the 8h funding prints per symbol.
    Binance depth: 2019-09+. Sign convention: positive = longs PAY shorts (short-perp earns).
    Replaces the deleted Midas carry_returns(); the carry+trend STRUCTURE is validated wiki
    knowledge but any new leg needs fresh forward validation before deployment."""
    import time as _t
    if isinstance(symbols, str):  # footgun class: "BTCUSDT" -> per-character iteration
        symbols = [symbols]
    cache = _day_cache("funding", [source, *symbols])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    out = {}
    for sym in symbols:
        rows, start = [], 1568102400000  # 2019-09-10, Binance perp launch era
        while True:
            u = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}"
                 f"&startTime={start}&limit=1000")
            batch = json.loads(_http_get(u, timeout=40))
            if not batch:
                break
            rows += batch
            if len(batch) < 1000:
                break
            start = batch[-1]["fundingTime"] + 1
            _t.sleep(0.3)  # public rate-limit politeness
        s = pd.Series({pd.Timestamp(r["fundingTime"], unit="ms"): float(r["fundingRate"])
                       for r in rows}).sort_index()
        out[sym] = s.resample("1D").sum()  # daily funding accrual
    panel = pd.DataFrame(out).sort_index()
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


# Liquid USDT-perp majors (deep funding+kline history). Smiths should still require >=N days of
# history per coin (cross-section grows over time as coins list; delisted coins drop out).
CRYPTO_MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT")

_KLINE_PERP = "https://fapi.binance.com/fapi/v1/klines"
_KLINE_SPOT = "https://api.binance.com/api/v3/klines"


def binance_klines(symbols=CRYPTO_MAJORS, market="perp", start="2019-01-01", interval="1d") -> pd.DataFrame:
    """Daily OHLCV klines from Binance public API ($0, deep history per listing). market='perp'
    (USDT-margined perps, fapi) or 'spot' (api). Returns a MultiIndex-column panel (symbol, field)
    with fields: open/high/low/close/volume/quote_volume/trades/taker_buy_quote.

    The crypto substrate beyond funding+spot: basis (perp_close vs spot_close), momentum/reversal,
    realized vol (from close or high/low), liquidity (quote_volume), and a deep-history FLOW/
    positioning proxy (taker_buy_quote / quote_volume = taker-buy fraction). Pair with funding_rates()
    for the carry leg. NOTE: Binance OI + long/short-ratio endpoints are LAST-30-DAYS only -> NOT
    backtestable (those positioning signals are DATA-GATED for deep history); taker_buy_quote is the
    free deep-history flow substitute.
    """
    import time as _t
    if isinstance(symbols, str):  # footgun class: 'BTCUSDT' -> per-character iteration
        symbols = [symbols]
    base = _KLINE_PERP if market == "perp" else _KLINE_SPOT
    cache = _day_cache("klines", [market, interval, *symbols])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    frames = {}
    for sym in symbols:
        rows, s = [], start_ms
        while True:
            u = f"{base}?symbol={sym}&interval={interval}&startTime={s}&limit=1000"
            try:
                batch = json.loads(_http_get(u, timeout=40))
            except Exception:
                break  # unlisted symbol on this venue / transient -> skip (cross-section tolerates gaps)
            if not batch:
                break
            rows += batch
            if len(batch) < 1000:
                break
            s = batch[-1][0] + 1
            _t.sleep(0.25)  # public rate-limit politeness
        if not rows:
            continue
        df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 10]]
        df.columns = ["t", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.drop(columns="t").set_index("date").astype(float)
        df = df[~df.index.duplicated(keep="last")]
        frames[sym] = df
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1)
    panel.columns.names = ["symbol", "field"]
    if cache:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel.sort_index()


_BYBIT_FUNDING = "https://api.bybit.com/v5/market/funding/history"


def bybit_funding(symbols=("BTCUSDT", "ETHUSDT"), start="2020-01-01") -> pd.DataFrame:
    """Bybit perp funding history (free public v5 API), DAILY sum of the 8h prints per symbol.
    SAME sign convention + daily-accrual shape as funding_rates() (Binance, positive = longs pay
    shorts) -> pair them for CROSS-EXCHANGE funding DISPERSION: bybit_funding - funding_rates =
    localized crowding / venue dislocation (a positioning signal Binance-alone can't see).
    Paginated backward (200 prints/page); day-cached; footgun-guarded."""
    import time as _t
    if isinstance(symbols, str):  # footgun class: 'BTCUSDT' -> per-character iteration
        symbols = [symbols]
    cache = _day_cache("bybit_funding", [*symbols])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    out, complete = {}, True
    for sym in symbols:
        rows, end = [], None
        while True:
            u = f"{_BYBIT_FUNDING}?category=linear&symbol={sym}&limit=200"
            if end is not None:
                u += f"&endTime={end}"
            try:
                r = json.loads(_http_get(u, timeout=40))
            except Exception:
                complete = False  # transient fetch error -> do NOT cache a truncated series
                break
            lst = (r.get("result") or {}).get("list") or []  # descending (newest first)
            if not lst:
                break
            rows += lst
            oldest = int(lst[-1]["fundingRateTimestamp"])
            if oldest <= start_ms or len(lst) < 200:
                break
            end = oldest - 1
            _t.sleep(0.25)
        if not rows:
            continue
        s = pd.Series({pd.Timestamp(int(x["fundingRateTimestamp"]), unit="ms"): float(x["fundingRate"])
                       for x in rows}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[sym] = s.resample("1D").sum()
    panel = pd.DataFrame(out).sort_index()
    if cache and complete and not panel.empty:  # only cache a COMPLETE pull (flake self-heals next run)
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


# Coin Metrics COMMUNITY on-chain/network metrics ($0, no key, CC BY-NC -> PERSONAL RESEARCH ONLY).
# STABLECOIN FLOWS (capital in/out of crypto) are a FREEBIE here: coinmetrics_metrics(("usdt","usdc"),
# ("SplyCur",)) -> USDT supply 2014+ (~$193B), USDC 2018+ (~$68B); growth = inflow, contraction = outflow.
# Asset tickers are lowercase CM names (btc/eth/sol/...). Common FREE community metrics:
#   PriceUSD, AdrActCnt (active addresses), TxCnt, TxTfrValAdjUSD (adjusted transfer value USD),
#   FeeTotUSD, SplyCur (current supply), CapRealUSD (realized cap), CapMrktCurUSD (market cap),
#   HashRate, DiffMean, IssTotUSD (issuance). Availability varies by asset; paid metrics 403 -> dropped.
CM_COMMUNITY_MAJORS = ("btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "avax", "link", "ltc", "dot", "trx")
_CM_COMMUNITY = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def coinmetrics_metrics(assets=CM_COMMUNITY_MAJORS, metrics=("PriceUSD", "AdrActCnt"),
                        frequency="1d", start="2010-01-01") -> pd.DataFrame:
    """Daily on-chain / network + market metrics from the Coin Metrics COMMUNITY API ($0, no key).
    LICENSE: Creative Commons BY-NC -> PERSONAL/NON-COMMERCIAL research only (not for resale/redistribution).
    Returns a MultiIndex-column panel (asset, metric) indexed by date. Day-cached, paginated, rate-limit
    polite (community: 10 req / 6s per IP). The on-chain FUNDAMENTALS layer beyond market microstructure
    (active addresses, tx value, fees, realized cap, supply, hashrate, issuance). NOTE: some metrics are
    paid-only on community (HTTP 403 -> that metric/asset is simply absent); 'reviewable' metrics may be
    minorly revised vs their original flash value (acceptable for daily backtests, flagged for honesty).
    """
    import time as _t
    if isinstance(assets, str):  # footgun class: 'btc' -> 'b','t','c'
        assets = [assets]
    if isinstance(metrics, str):
        metrics = [metrics]
    cache = _day_cache("cm_community", [frequency, *sorted(assets), *sorted(metrics)])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    base = (f"{_CM_COMMUNITY}?assets={','.join(assets)}&metrics={','.join(metrics)}"
            f"&frequency={frequency}&start_time={start}&page_size=10000&paging_from=start"
            f"&ignore_forbidden_errors=true&ignore_unsupported_errors=true")
    rows, url = [], base
    for _ in range(200):  # hard page cap (safety)
        try:
            resp = json.loads(_http_get(url, timeout=40))
        except Exception:
            break
        rows += resp.get("data", [])
        nxt = resp.get("next_page_url")
        if not nxt:
            break
        url = nxt
        _t.sleep(0.7)  # community 10 req / 6s
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    metric_cols = [c for c in df.columns if c not in ("asset", "time", "date")]
    long = df.melt(id_vars=["asset", "date"], value_vars=metric_cols, var_name="metric", value_name="val")
    long["val"] = pd.to_numeric(long["val"], errors="coerce")
    panel = long.pivot_table(index="date", columns=["asset", "metric"], values="val")
    panel.columns.names = ["asset", "metric"]
    panel = panel.sort_index()
    if cache and not panel.empty:
        try:
            tmp = cache + ".tmp"
            panel.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return panel


def treasury_auctions(types=("Note", "Bond"), start="2010-01-01") -> pd.DataFrame:
    """US Treasury auction calendar/history from the free TreasuryDirect API (depth: 1979+).
    Long DataFrame [auction_date, announcement_date, issue_date, sec_type, term, cusip,
    offering_amount] sorted by auction_date. POINT-IN-TIME: the auction is knowable from
    announcement_date (~1 week prior); supply-concession studies may condition on the
    announcement but measure returns around auction_date."""
    if isinstance(types, str):  # footgun class: "Note" -> ['N','o','t','e']
        types = [types]
    types = list(types)
    cache = _day_cache("ustauct", [*types, start])
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    rows = []
    for t in types:
        u = f"https://www.treasurydirect.gov/TA_WS/securities/search?type={t}&format=json"
        for r in json.loads(_http_get(u, timeout=120)):
            if not r.get("auctionDate"):
                continue
            rows.append({"auction_date": pd.Timestamp(r["auctionDate"]),
                         "announcement_date": pd.Timestamp(r["announcementDate"]) if r.get("announcementDate") else pd.NaT,
                         "issue_date": pd.Timestamp(r["issueDate"]) if r.get("issueDate") else pd.NaT,
                         "sec_type": t, "term": r.get("securityTerm", ""),
                         "cusip": r.get("cusip", ""),
                         "offering_amount": float(r["offeringAmount"]) if r.get("offeringAmount") else np.nan})
    df = pd.DataFrame(rows)
    df = df[df["auction_date"] >= pd.Timestamp(start)]
    df = df.sort_values("auction_date").reset_index(drop=True)
    if cache:
        try:
            tmp = cache + ".tmp"
            df.to_parquet(tmp)
            os.replace(tmp, cache)
        except OSError:
            pass
    return df
