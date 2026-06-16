"""
Crypto on-chain VALUE premium — cross-sectional MVRV valuation factor.

PRE-REGISTRATION (frozen design; primary = grid["default"]):
  Premium  : fundamental-valuation mean-reversion to on-chain cost-basis. A genuine RISK
             premium — you are paid to hold networks trading cheap vs their own realized
             cost basis (cheap-for-fundamental-risk-reasons), on an axis ORTHOGONAL to the
             price/carry/vol substrates already exhausted (and nulled) in crypto.
  Signal   : for each coin compute a WITHIN-COIN time-series z-score of CoinMetrics
             CapMVRVCur (the FREE MVRV ratio) vs its OWN trailing 365d distribution
             (neutralises cross-coin MVRV-level confounds, e.g. BTC's level != an alt's).
             MONTHLY, cross-sectionally RANK coins by that z; LONG cheapest quintile
             (low MVRV-z, near/below cost basis), SHORT most-expensive quintile.
  Construct: dollar-neutral, inverse-vol sized per leg, 12% vol target, monthly rebalance
             with hysteresis bands (q_enter=0.20 / q_exit=0.35 => min-hold ~1 month,
             low turnover by design — monthly is DELIBERATE for a slow fundamental factor,
             which overrides the generic 'weekly' default), residual BTC-beta neutralised
             with a BTC perp leg (within-asset-class hedge, part of the frozen alpha book —
             NOT an equity-ETF sleeve, so no hedge_tickers). Net of a conservative 20bps
             round-trip taker cost (Binance taker is ~5-7bps/side, so this is punitive).
  LOOK-AHEAD: every signal is built as-of date d using only data <= d, then the FULL weight
             matrix is lagged one day (W.shift(1)) before net_of_cost / trades_from_weights.
             MVRV is reindexed to the price grid and forward-filled (point-in-time last-known
             on-chain value — never a future value).
  Breadth caveat (gate0): the free MVRV x liquid-perp intersection is bounded (~25-35 coins).
             The SEARCH cross-section is the top liquid tier (~18 alts + BTC hedge). The
             generalization tiers are necessarily SMALLER than the equity-ideal 150-400 names
             — flagged honestly; the true cross-market anchor for the value mechanism is the
             already-validated equity value premium (referenced, not re-run here).

BUG FIX (was: KeyError 'px' inside signal): binance_klines returns column labels that did NOT
  match the literal "...USDT" curated strings (base-symbol / OHLCV-MultiIndex / long shapes are
  all possible), so _survivors() found nothing, BTC was absent from the price frame, and the
  old MultiIndex-column panel silently dropped its "px" level. FIX = canonicalize every adapter
  output column to a common base-symbol space (handles base/USDT/MultiIndex/long) AND emit a
  FLAT-column panel (price cols = base symbol, mvrv cols = "MVRV_<sym>") so there is no column
  MultiIndex to lose across the harness round-trip; empty slices now return clean zeros.

PERF: the data fetch (Binance klines + CoinMetrics MVRV) is memoized in-process so it happens
  exactly once per process (load_data + each load_gen_data share the cache). In-memory cache
  ONLY — NO external side effects (no file/IO writes, no config, no capital).
"""

from sdk.harness import StrategySpec
from sdk.adapters import (
    binance_universe, binance_klines, coinmetrics_metrics,   # crypto adapters (DATA_CATALOG.md)
    trend_returns, inv_vol_position,                          # (unused fallbacks, kept tested)
)
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- config
BTC_SYM   = "BTC"       # canonical (base-symbol) name used everywhere downstream of loading
START     = "2019-01-01"
SEARCH_N  = 18          # top liquid alts in the SEARCH cross-section (BTC kept only as hedge)
MIN_PX    = 180         # min daily price obs to enter the universe
MIN_MVRV  = 400         # min MVRV obs (>= 365 z-window + warmup) -> survivorship/robustness

# curated liquid USDT-perp coins (Binance *input* symbols) that have CoinMetrics community
# CapMVRVCur history, ordered ~by liquidity/market-cap. Deterministic master list (covers every
# slice; no live binance_universe round-trip needed -> avoids repeated rate-limited fetches).
_CURATED = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT",
    "LINKUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","TRXUSDT","ATOMUSDT","ETCUSDT",
    "XLMUSDT","NEARUSDT","ICPUSDT","FILUSDT","VETUSDT","THETAUSDT","ALGOUSDT","EGLDUSDT",
    "AAVEUSDT","GRTUSDT","UNIUSDT","MKRUSDT","EOSUSDT","XTZUSDT","ZECUSDT","DASHUSDT",
    "COMPUSDT","SNXUSDT","CRVUSDT","SUSHIUSDT","YFIUSDT","ZRXUSDT","BATUSDT","OMGUSDT",
    "KSMUSDT","WAVESUSDT","ZILUSDT","QTUMUSDT",
]

def _canon(s):
    """canonical base symbol: strip stable-quote suffix + separators, upper-case.
    'BTCUSDT'->'BTC', 'btc'->'BTC', 'ETH-USD'->'ETH'. Makes Binance/CoinMetrics align."""
    s = str(s).upper().replace("-", "").replace("/", "").replace(":", "").strip()
    for suf in ("USDTM", "USDT", "USDC", "BUSD", "PERP", "USD"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s

_CANON_ORDER = list(dict.fromkeys(_canon(s) for s in _CURATED))   # liquidity-ordered base syms

_SECTOR_RAW = {
    "BTCUSDT":"store-of-value","LTCUSDT":"payments","BCHUSDT":"payments","DOGEUSDT":"payments",
    "XRPUSDT":"payments","XLMUSDT":"payments",
    "ETHUSDT":"smart-contract-l1","BNBUSDT":"exchange-l1","SOLUSDT":"smart-contract-l1",
    "ADAUSDT":"smart-contract-l1","AVAXUSDT":"smart-contract-l1","TRXUSDT":"smart-contract-l1",
    "ETCUSDT":"smart-contract-l1","NEARUSDT":"smart-contract-l1","ALGOUSDT":"smart-contract-l1",
    "EOSUSDT":"smart-contract-l1","XTZUSDT":"smart-contract-l1","EGLDUSDT":"smart-contract-l1",
    "ICPUSDT":"smart-contract-l1","WAVESUSDT":"smart-contract-l1","QTUMUSDT":"smart-contract-l1",
    "ZILUSDT":"smart-contract-l1",
    "DOTUSDT":"interop-l1","ATOMUSDT":"interop-l1","KSMUSDT":"interop-l1",
    "MATICUSDT":"scaling-l2","FILUSDT":"storage","LINKUSDT":"oracle","GRTUSDT":"data",
    "VETUSDT":"supply-chain","THETAUSDT":"media",
    "AAVEUSDT":"defi","UNIUSDT":"defi","MKRUSDT":"defi","COMPUSDT":"defi","SNXUSDT":"defi",
    "CRVUSDT":"defi","SUSHIUSDT":"defi","YFIUSDT":"defi","ZRXUSDT":"defi","BATUSDT":"defi",
    "OMGUSDT":"defi","ZECUSDT":"privacy","DASHUSDT":"privacy",
}
_SECTOR = {_canon(k): v for k, v in _SECTOR_RAW.items()}          # keyed by canonical base sym

DEFAULT_PARAMS = dict(
    z_window=365, z_min=180,        # within-coin trailing MVRV z-score
    q_enter=0.20, q_exit=0.35,      # quintile sort + hysteresis bands
    vol_lb=30, vol_lb_port=30, beta_lb=60, btc_cap=0.50,
    target_vol=0.12, scale_min=0.30, scale_max=3.0,
    min_names=6, cost_bps=20.0, name="crypto_mvrv_value",
)

# in-process memo cache (in-memory ONLY; no file writes / no external side effects)
_SURV_CACHE = {}

# ----------------------------------------------------------------------------- coercion helpers
def _ensure_dt_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        lc = {str(c).lower(): c for c in df.columns}
        for k in ("time", "date", "datetime", "timestamp", "open_time", "index"):
            if k in lc:
                df = df.set_index(lc[k]); break
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex) and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def _flatten_value_field(df, value_hint=("closeadj", "close", "price", "last", "value", "c")):
    """If columns are a (symbol, field) MultiIndex, pick a close-like field -> symbol columns."""
    if isinstance(df.columns, pd.MultiIndex):
        fields = list(df.columns.get_level_values(-1))
        cf = next((f for f in value_hint if f in fields),
                  next((f for f in fields if "close" in str(f).lower()), fields[0] if fields else None))
        if cf is not None:
            df = df.xs(cf, axis=1, level=-1)
    return df

def _maybe_pivot_long(df, value_hint):
    """If long-format (a symbol/asset column + a value column), pivot to wide symbol columns."""
    if isinstance(df.columns, pd.MultiIndex):
        return df
    lc = {str(c).lower(): c for c in df.columns}
    sym_col = next((lc[k] for k in ("symbol", "asset", "ticker", "pair") if k in lc), None)
    if sym_col is None:
        return df
    vcol = next((lc[k] for k in value_hint if k in lc), None)
    if vcol is None:
        vcol = next((c for c in df.columns if c != sym_col), None)
    if vcol is None:
        return df
    return df.pivot_table(index=df.index, columns=sym_col, values=vcol, aggfunc="last")

def _coerce_wide(raw, value_hint, daily=False):
    df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame()
    df = _flatten_value_field(df, value_hint)
    df = _ensure_dt_index(df)
    df = _maybe_pivot_long(df, value_hint)
    df.columns = [_canon(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    if daily and isinstance(df.index, pd.DatetimeIndex) and len(df):
        try:
            df = df.resample("1D").last()
        except Exception:
            pass
    return df

# ----------------------------------------------------------------------------- loaders
def _load_px(symbols, start):
    return _coerce_wide(binance_klines(symbols, start),
                        value_hint=("closeadj", "close", "price", "last", "value", "c"),
                        daily=True)

def _load_mvrv(bases, start):
    assets = [b.lower() for b in bases]
    raw = coinmetrics_metrics(assets, "CapMVRVCur", start=start)
    df = _coerce_wide(raw, value_hint=("capmvrvcur", "mvrv", "value"), daily=False)
    return df[[c for c in df.columns if c in set(bases)]] if len(df.columns) else df

def _survivors(start):
    cached = _SURV_CACHE.get(start)
    if cached is not None:
        return cached
    px_all = _load_px(list(_CURATED), start)
    bases  = [b for b in _CANON_ORDER if b in px_all.columns] or list(_CANON_ORDER)
    if BTC_SYM not in bases:
        bases = [BTC_SYM] + bases
    mv_all = _load_mvrv(bases, start)
    surv = [b for b in _CANON_ORDER
            if b in px_all.columns and b in mv_all.columns
            and px_all[b].notna().sum() > MIN_PX and mv_all[b].notna().sum() > MIN_MVRV]
    out = (surv, px_all, mv_all)
    _SURV_CACHE[start] = out
    return out

def _slice(label, surv):
    """search/gen symbol slices. ALL gen tiers are DISJOINT from the search tier
    (index >= SEARCH_N); gen tiers may overlap each other. BTC excluded (hedge only)."""
    alt = [s for s in surv if s != BTC_SYM]
    if label in (None, "default", "search"):
        return alt[:SEARCH_N]
    if label == "lower_liq":
        return alt[SEARCH_N:]
    if label == "mid_alt":
        return alt[SEARCH_N:SEARCH_N + 7]
    if label == "tail_alt":
        return alt[SEARCH_N + 5:]
    return alt[:SEARCH_N]

def _panel_from(sel, px_all, mv_all):
    """FLAT-column panel (no MultiIndex): price cols = base sym (+BTC hedge),
    valuation cols = 'MVRV_<sym>'. Robust across the harness round-trip."""
    sel    = [s for s in sel if s in px_all.columns]
    pxcols = list(dict.fromkeys([c for c in (sel + [BTC_SYM]) if c in px_all.columns]))
    px = px_all[pxcols].copy()
    mvcols = [s for s in sel if s in mv_all.columns and s != BTC_SYM]   # BTC out of cross-section
    mv = mv_all[mvcols].copy()
    mv.columns = ["MVRV_" + c for c in mv.columns]
    return pd.concat([px, mv], axis=1).sort_index()

def _rebal_dates(idx):
    df = pd.DataFrame({"d": idx})
    df["p"] = idx.to_period("M")
    return pd.DatetimeIndex(df.groupby("p", sort=True)["d"].first().values)

def _inv_vol_w(names, vol_row, gross):
    if not names:
        return pd.Series(dtype=float)
    iv = (1.0 / vol_row.reindex(names)).replace([np.inf, -np.inf], np.nan).dropna()
    if iv.empty or iv.sum() == 0:
        iv = pd.Series(1.0, index=names)
    return gross * iv / iv.sum()

# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    surv, px_all, mv_all = _survivors(START)
    return _panel_from(_slice("default", surv), px_all, mv_all)

def load_gen_data(label) -> pd.DataFrame:
    surv, px_all, mv_all = _survivors(START)
    return _panel_from(_slice(label, surv), px_all, mv_all)

# ----------------------------------------------------------------------------- signal
def signal(panel, **params):
    p = dict(DEFAULT_PARAMS); p.update(params or {})

    cols   = [str(c) for c in panel.columns]
    mvcols = [c for c in cols if c.startswith("MVRV_")]
    pxcols = [c for c in cols if not c.startswith("MVRV_")]
    px   = panel[pxcols].sort_index()
    mvrv = panel[mvcols].copy()
    mvrv.columns = [c[len("MVRV_"):] for c in mvrv.columns]

    idx   = px.index
    empty = pd.Series(0.0, index=idx, name=p["name"])
    if px.shape[1] == 0:
        return empty, []

    mvrv = mvrv.sort_index().reindex(idx).ffill()         # PIT: last-known on-chain value
    rets = px.pct_change()
    alt  = [c for c in mvrv.columns if c != BTC_SYM and c in px.columns]
    if len(alt) < 2:
        return empty, []

    # --- within-coin trailing MVRV z (no look-ahead: trailing rolling window only) ---
    m  = mvrv[alt]
    mu = m.rolling(int(p["z_window"]), min_periods=int(p["z_min"])).mean()
    sd = m.rolling(int(p["z_window"]), min_periods=int(p["z_min"])).std()
    z  = (m - mu) / sd.replace(0, np.nan)
    rv = rets[alt].rolling(int(p["vol_lb"]), min_periods=10).std()

    qen, qex = float(p["q_enter"]), float(p["q_exit"])
    reb = _rebal_dates(idx)

    # --- monthly long-cheap / short-expensive book with hysteresis (dollar-neutral) ---
    rows, prev_long, prev_short = {}, set(), set()
    for d in reb:
        zd = z.loc[d].dropna()
        if len(zd) < int(p["min_names"]):
            continue
        r = zd.rank(pct=True)
        long_e,  long_k  = set(r[r <= qen].index),     set(r[r <= qex].index)
        short_e, short_k = set(r[r >= 1 - qen].index), set(r[r >= 1 - qex].index)
        new_long  = list((long_e  | (prev_long  & long_k))  - short_e)
        new_short = list((short_e | (prev_short & short_k)) - set(new_long))
        volrow = rv.loc[d]
        wl = _inv_vol_w(new_long,  volrow, +0.5)
        ws = _inv_vol_w(new_short, volrow, -0.5)
        wrow = pd.Series(0.0, index=px.columns)
        for k, v in wl.items(): wrow[k] = v
        for k, v in ws.items(): wrow[k] = v
        rows[d] = wrow
        prev_long, prev_short = set(new_long), set(new_short)

    if not rows:
        return empty, []

    target = pd.DataFrame(rows).T.reindex(columns=px.columns).fillna(0.0)
    W = target.reindex(idx).ffill().fillna(0.0)          # hold between monthly rebalances

    # --- residual BTC-beta neutralisation (estimated on the unhedged book, held monthly) ---
    if BTC_SYM in rets.columns:
        btc_ret = rets[BTC_SYM]
        r_book = (W.shift(1) * rets).sum(axis=1)          # unhedged book (lagged weights)
        cov = r_book.rolling(int(p["beta_lb"]), min_periods=20).cov(btc_ret)
        var = btc_ret.rolling(int(p["beta_lb"]), min_periods=20).var().replace(0, np.nan)
        beta = (cov / var).reindex(reb).ffill().reindex(idx).ffill().fillna(0.0)
        W[BTC_SYM] = (-beta).clip(-float(p["btc_cap"]), float(p["btc_cap"])).values

    # --- 12% vol target (scale set on trailing realised vol, updated monthly) ---
    r_book2 = (W.shift(1) * rets).sum(axis=1)
    realized = r_book2.rolling(int(p["vol_lb_port"]), min_periods=20).std() * np.sqrt(252)
    scale = (float(p["target_vol"]) / realized.replace(0, np.nan))
    scale = scale.reindex(reb).ffill().reindex(idx).ffill().fillna(1.0) \
                 .clip(float(p["scale_min"]), float(p["scale_max"]))
    W = W.mul(scale, axis=0)

    # --- LAG one day (our responsibility), then net-of-cost + contract ledger ---
    Wlag   = W.shift(1).fillna(0.0)
    daily  = net_of_cost(Wlag, rets, cost_bps=float(p["cost_bps"]), name=p["name"])
    smap   = {c: _SECTOR.get(c, "altcoin") for c in px.columns}
    trades = trades_from_weights(Wlag, rets, smap)
    return daily, trades

# ----------------------------------------------------------------------------- soft expectations
def _chk_btc_beta(ctx):
    try:
        s = ctx.get("search"); panel = ctx.get("panel")
        if s is None or panel is None or BTC_SYM not in panel.columns:
            return {"pass": True, "observed": "n/a"}
        df = pd.concat([s, panel[BTC_SYM].pct_change()], axis=1).dropna()
        df = df[df.index < pd.Timestamp(ctx["holdout_start"])]
        if len(df) < 60:
            return {"pass": True, "observed": "n/a"}
        y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
        beta = float(np.cov(x, y)[0, 1] / np.var(x))
        return {"pass": abs(beta) <= 0.30, "observed": round(beta, 3)}
    except Exception as e:
        return {"pass": True, "observed": f"err:{e}"}

def _chk_monthly_hold(ctx):
    try:
        tr = ctx.get("trades") or []
        hs = [t["hold_days"] for t in tr
              if t.get("entry_date", "9999") < ctx["holdout_start"] and "hold_days" in t]
        if not hs:
            return {"pass": True, "observed": "n/a"}
        med = float(np.median(hs))
        return {"pass": med >= 15, "observed": med}
    except Exception as e:
        return {"pass": True, "observed": f"err:{e}"}

def _chk_breadth(ctx):
    try:
        n = int(sum(1 for c in ctx["panel"].columns if str(c).startswith("MVRV_")))
        return {"pass": n >= 12, "observed": n}
    except Exception as e:
        return {"pass": True, "observed": f"err:{e}"}

# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_mvrv_value",
    family="value",
    title="Crypto on-chain MVRV value (long cheap / short expensive networks, BTC-beta-neutral perp book)",
    markets=["crypto-perp"],
    data_desc=("Binance USDT-perp daily closes (binance_klines/binance_universe) x CoinMetrics "
               "community CapMVRVCur (FREE MVRV ratio; realized-cap itself is paid, the ratio is not). "
               "Adapter columns canonicalized to base symbols (BTCUSDT->BTC) so Binance/CoinMetrics align; "
               "panel is flat-column (price=<sym>, valuation=MVRV_<sym>). Cross-section = liquid-perp ∩ "
               "free-MVRV intersection; BTC perp used only as the beta hedge."),
    pre_registration=(
        "PRIMARY = grid['default']: within-coin trailing-365d MVRV z-score -> monthly cross-sectional "
        "QUINTILE sort -> LONG cheapest / SHORT most-expensive, dollar-neutral, inverse-vol, 12% vol "
        "target, hysteresis (enter 0.20 / exit 0.35), residual BTC-beta neutralised, net 20bps round-trip. "
        "Axis is fundamental on-chain VALUATION — orthogonal to the already-nulled crypto price/funding/vol "
        "substrates (anti-pattern #3: change the axis, not the params). Monthly horizon is deliberate (a slow "
        "valuation signal, NOT short-horizon reversal). Signals built as-of d using data<=d then the whole "
        "weight matrix is lagged one day; MVRV is reindexed-and-ffilled (PIT last-known). BREADTH CAVEAT: the "
        "free-MVRV ∩ liquid-perp universe is bounded (~25-35 coins) so generalization tiers are smaller than "
        "the equity-ideal — the cross-market value anchor is the already-validated equity value premium."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "z_730":   {"z_window": 730, "z_min": 365},   # 2yr valuation window
        "tercile": {"q_enter": 0.33, "q_exit": 0.45}, # tercile vs quintile sort
        "vt08":    {"target_vol": 0.08},              # lower vol target
    },
    scope="broad",
    generalization_universes=["lower_liq", "mid_alt", "tail_alt"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=12,
    expectations=[
        {"name": "btc_beta_neutral",
         "claim": "|OLS beta of search-window net returns on BTC daily returns| <= 0.30 (book is hedged)",
         "check": _chk_btc_beta},
        {"name": "monthly_hold_low_turnover",
         "claim": "median search-window trade hold_days >= 15 (monthly rebalance + hysteresis)",
         "check": _chk_monthly_hold},
        {"name": "cross_section_breadth",
         "claim": "search cross-section has >= 12 coins with free MVRV (Fundamental-Law breadth)",
         "check": _chk_breadth},
    ],
)