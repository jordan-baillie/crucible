"""Tested data adapters the agent's generated signal code composes (reliability > reinvention).
All FREE / already-owned sources. The agent is told to use THESE, not raw downloads."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def yf_panel(tickers, start="2005-01-01") -> pd.DataFrame:
    """Close panel for a list of yfinance tickers (business-day grid). Handles single/multi shapes."""
    import yfinance as yf
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
    return panel.reindex(bidx).ffill(limit=3)


def fred_series(series_ids, start="2005-01-01") -> pd.DataFrame:
    """Daily-ffilled FRED series panel. series_ids: dict {fred_id: column_name} or list."""
    import urllib.request
    key = json.load(open(os.path.expanduser("~/.atlas-secrets.json")))["fred_api_key"]
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
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B")).ffill()


def trend_returns(**params):
    """The validated Boreas 21-market TSMOM stream (daily_returns, trades) — a ready hedge leg."""
    sys.path.insert(0, "/root/boreas/research")
    from tsmom import run_tsmom
    return run_tsmom(**params)


def carry_returns(**params):
    """The Midas crypto funding-carry leg return stream (the near-miss carry leg)."""
    sys.path.insert(0, "/root/midas/research/perp_validation/xs_funding_carry")
    import load_binance_vision as bv, run_xs_funding_validation as v
    F, R, Q, listing = bv.build_panel(min_days=40)
    names = [s for s in R.columns if s in listing]
    btc = R["BTCUSDT"] if "BTCUSDT" in R else pd.Series(0.0, index=R.index)
    bt = v.backtest(F, R, Q, listing, v.COST_BPS, names, btc)
    s = bt["net"].copy(); s.index = pd.to_datetime(s.index)
    try: s.index = s.index.tz_localize(None)
    except Exception: s.index = s.index.tz_convert(None)
    if hasattr(s.index, "normalize"):
        s.index = s.index.normalize()
    return s


def inv_vol_position(signal_df, rets, target_vol=0.10, vol_lb=60, max_pos=2.0, rebalance="W-FRI"):
    """Standard inverse-vol sizing + weekly hold + 1d lag (no look-ahead). Reusable building block."""
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    raw = (signal_df * (target_vol / vol.replace(0, np.nan))).clip(-max_pos, max_pos)
    return raw.resample(rebalance).last().reindex(rets.index, method="ffill").shift(1).fillna(0.0)


# ── Owned Sharadar — SURVIVORSHIP-CLEAN US equities (PREFER over yf_panel for US stocks) ──
SHARADAR_DIR = "/root/atlas/data/sharadar"
_CACHE_DIR = "/root/atlas/data/cache"


def _sep_cache() -> str:
    """Build/load a cached long parquet of owned Sharadar SEP (one-time ~1-2min build)."""
    import zipfile
    out = os.path.join(_CACHE_DIR, "sep_long.parquet")
    if not os.path.exists(out):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(os.path.join(SHARADAR_DIR, "SEP.zip"))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=["ticker", "date", "closeadj", "volume"], parse_dates=["date"])
        df.sort_values("ticker").to_parquet(out, index=False, row_group_size=500_000)
    return out


def sep_panel(tickers=None, start="2000-01-01", end=None, field="closeadj") -> pd.DataFrame:
    """SURVIVORSHIP-CLEAN US equity daily panel from OWNED Sharadar SEP (delisted incl, split+div adj).
    Wide DataFrame: business-day DatetimeIndex x ticker of `field` (closeadj=adjusted close | volume).
    PREFER THIS over yf_panel for US stocks (yfinance has survivorship bias — a wiki anti-pattern).
    tickers=None loads ALL (~16k, heavy) — pass a list (e.g. from us_universe())."""
    path = _sep_cache()
    filt = [("ticker", "in", list(tickers))] if tickers is not None else None
    df = pd.read_parquet(path, columns=["ticker", "date", field], filters=filt)
    df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    panel = df.pivot_table(index="date", columns="ticker", values=field).sort_index()
    if panel.empty:
        return panel
    bidx = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    return panel.reindex(bidx).ffill(limit=3)


def us_universe(sector=None, category="Domestic Common Stock", marketcap=None, include_delisted=True) -> list:
    """US-equity universe from OWNED Sharadar TICKERS (cheap, no price load). Includes DELISTED names
    by default (survivorship-clean). Filter by `sector` (e.g. 'Financial Services'), `category`
    (default common stock), `marketcap` scale substring (e.g. 'Large','Mid','Small'). Returns ticker list."""
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
    return sorted(tk["ticker"].dropna().unique().tolist())


def _sf1_cache() -> str:
    import zipfile
    out = os.path.join(_CACHE_DIR, "sf1_long.parquet")
    if not os.path.exists(out):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        z = zipfile.ZipFile(os.path.join(SHARADAR_DIR, "SF1.zip"))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, parse_dates=["datekey", "calendardate"], low_memory=False)
        df.sort_values("ticker").to_parquet(out, index=False, row_group_size=200_000)
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
