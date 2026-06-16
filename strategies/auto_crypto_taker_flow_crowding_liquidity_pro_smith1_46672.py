"""
Crucible strategy module
========================
Crypto taker-flow crowding -> liquidity-provision premium (cross-sectional, broad perp universe).

MECHANISM (a microstructure RISK premium, NOT a price-prediction edge):
  You are PAID to absorb crowded aggressive taker flow.  Coins under the heaviest
  *persistent aggressive SELLING* (forced / panic takers hitting the bid) trade cheap to
  their liquidity providers; coins under the heaviest *persistent aggressive BUYING*
  (crowded, fragile longs) are rich.  Provide liquidity to the crowd:
  LONG the under-bought, SHORT the over-bought, dollar-neutral, residual BTC-beta trimmed.

SIGNAL (pre-registered PRIMARY; no grid cherry-picking):
  TBI_t  = taker_buy_quote / quote_volume - 0.5      (daily taker-buy imbalance, in [-0.5,0.5])
  flow_t = EWMA_5d(TBI_t)                             (PERSISTENT crowding, not one-day noise)
  Each Friday, cross-sectionally percentile-rank flow across the perp cross-section:
      enter LONG  if rank <= 1/3   (under-bought -> provide liquidity to aggressive sellers)
      enter SHORT if rank >= 2/3   (over-bought  -> fade crowded longs)
      exit when rank leaves the loose 0.45 / 0.55 band (HYSTERESIS) AND >= 5d min-hold.
  Inverse-vol weight within each leg, dollar-neutral legs, 10% annualized vol target,
  residual BTC-beta trimmed with a small BTC-perp leg.  Net of ~20bps round-trip taker
  cost (= 10bps/side applied on turnover by net_of_cost).

  NOTE on sizing: the proposal's "equal-weight within legs" prose is superseded by the
  MANDATORY harness contract ("Inverse-vol size."); inverse-vol is the frozen design here.

HONEST HEADWIND: if the flow signal empirically collapses to price reversal it inherits the
  two prior crypto x-sec reversal nulls.  The standalone test + the `flow_not_reversal`
  soft-expectation are designed to falsify that, not paper over it.

LAG: every weight is decided from information available up to date d, then executed via
  W.shift(1) (a 1-day lag) BEFORE net_of_cost / trades_from_weights.  The lag is applied
  here explicitly -- nothing downstream re-lags.
"""

from sdk.harness import StrategySpec
from sdk.signal_kit import net_of_cost, trades_from_weights
from sdk.adapters import binance_universe, binance_klines
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- config
START      = "2019-01-01"
HOLDOUT    = "2022-01-01"
POOL_N     = 125            # liquidity-ranked perp pool to carve disjoint universes from
SEARCH_END = 50            # search cross-section = top-50 liquid perps (>=40 for breadth gate)
MIN_OBS    = 180           # drop barely-listed coins (need real history)
FIELDS     = ("close", "taker_buy_quote", "quote_volume", "volume")

# disjoint generalization universes (DIFFERENT, deeper-liquidity coins -> share NO
# cross-section tickers with the search top-50; BTC is appended only as hedge infra).
GEN_SLICES = {
    "perp_liq_t2": (50, 75),
    "perp_liq_t3": (75, 100),
    "perp_liq_t4": (100, 125),
}

# ----------------------------------------------------------------------------- helpers
def _pool(n=POOL_N):
    try:
        return list(binance_universe(n))
    except Exception:
        return list(binance_universe(75))


def _btc_symbol(cols):
    cols = list(cols)
    for c in cols:
        if str(c).upper() in ("BTC", "BTCUSDT", "BTCUSDTPERP", "BTC-USDT", "BTCUSD", "XBTUSDT"):
            return c
    for c in cols:
        if str(c).upper().startswith("BTC"):
            return c
    return None


def _load_klines(symbols, start=START):
    """field -> wide (date x symbol) DataFrame.  Robust to a few plausible adapter shapes."""
    symbols = list(dict.fromkeys(symbols))
    # 1) per-field wide panels (mirrors sep_panel's field= convention)
    try:
        out = {}
        for f in FIELDS:
            df = binance_klines(symbols, market="perp", field=f, start=start)
            if not isinstance(df, pd.DataFrame):
                raise TypeError
            out[f] = df
        return out
    except Exception:
        pass
    # 2) single frame -> dict / multiindex / long
    raw = binance_klines(symbols, market="perp", start=start)
    if isinstance(raw, dict):
        return {f: raw[f] for f in FIELDS}
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        if set(FIELDS).issubset(lvl0):
            return {f: raw[f] for f in FIELDS}
        return {f: raw.xs(f, axis=1, level=-1) for f in FIELDS}
    lc = {str(c).lower(): c for c in raw.columns}
    dcol = lc.get("date") or lc.get("timestamp") or lc.get("time") or lc.get("open_time")
    scol = lc.get("ticker") or lc.get("symbol") or lc.get("pair")
    out = {}
    for f in FIELDS:
        fcol = lc.get(f)
        w = raw.pivot_table(index=dcol, columns=scol, values=fcol)
        w.index = pd.to_datetime(w.index)
        out[f] = w
    return out


def _klines_panel(symbols, start=START):
    fld = _load_klines(symbols, start)
    panel = pd.concat({f: fld[f] for f in FIELDS}, axis=1)
    panel = panel.sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def _sector_map(symbols):
    """Liquidity-tier pseudo-sectors (binance_universe is liquidity-ranked) so the trade
    ledger has genuine cross-'sector' spread for the deployment gate."""
    syms = list(symbols)
    n = max(len(syms), 1)
    tiers = 5
    return {s: "liq_tier_%d" % (min(tiers - 1, int(i * tiers / n)) + 1) for i, s in enumerate(syms)}


def _rebal_dates(idx):
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period("W-FRI")).max()
    return pd.DatetimeIndex(sorted(pd.Series(last).values))


# ----------------------------------------------------------------------------- signal
def signal(panel,
           ewma_span=5,
           enter_q=1.0 / 3.0,
           exit_q=0.45,          # long loose-band upper bound (short uses 1-exit_q)
           min_hold=5,
           vol_lb=30,
           target_vol=0.10,
           beta_lb=60,
           beta_hedge=True,
           cost_bps=10.0,        # per-side on turnover ~= 20bps round-trip taker cost
           **kw):
    close = panel["close"].astype(float)
    tbq   = panel["taker_buy_quote"].astype(float)
    qv    = panel["quote_volume"].astype(float)

    keep = [c for c in close.columns if close[c].notna().sum() >= MIN_OBS]
    if len(keep) < 6:
        return pd.Series(dtype=float, name="taker_flow_crowding"), []
    close, tbq, qv = close[keep], tbq[keep], qv[keep]
    rets = close.pct_change()

    btc  = _btc_symbol(keep)
    cols = [c for c in keep if c != btc]          # tradeable cross-section (BTC = hedge only)

    # --- persistent aggressive-flow crowding signal ---
    tbi  = (tbq / qv).clip(0.0, 1.0) - 0.5
    tbi  = tbi.where(np.isfinite(tbi))
    flow = tbi.ewm(span=ewma_span, min_periods=ewma_span).mean()
    pr   = flow[cols].rank(axis=1, pct=True)       # cross-sectional percentile per date

    vol  = rets.rolling(vol_lb, min_periods=max(5, vol_lb // 2)).std()
    rdates = _rebal_dates(close.index)
    if len(rdates) == 0:
        return pd.Series(dtype=float, name="taker_flow_crowding"), []

    # --- weekly state machine: tercile membership + hysteresis + min-hold ---
    membership, entry_dt, W_rows = {}, {}, {}
    lo_exit, hi_enter, hi_exit = exit_q, 1.0 - enter_q, 1.0 - exit_q
    for d in rdates:
        prd = pr.loc[d] if d in pr.index else pd.Series(dtype=float)
        # exits (respect min-hold)
        for sym in list(membership):
            sd = membership[sym]
            p  = prd.get(sym, np.nan)
            held = (d - entry_dt[sym]).days
            drop = (not np.isfinite(p))
            if not drop and held >= min_hold:
                drop = (sd > 0 and p > lo_exit) or (sd < 0 and p < hi_exit)
            if drop:
                membership.pop(sym, None); entry_dt.pop(sym, None)
        # entries
        for sym in cols:
            if sym in membership:
                continue
            p = prd.get(sym, np.nan)
            if not np.isfinite(p):
                continue
            if p <= enter_q:
                membership[sym] = 1;  entry_dt[sym] = d
            elif p >= hi_enter:
                membership[sym] = -1; entry_dt[sym] = d
        # inverse-vol, dollar-neutral leg weights (each leg -> 0.5 gross)
        w  = pd.Series(0.0, index=cols)
        vd = vol.loc[d]
        longs  = [s for s, sd in membership.items() if sd > 0]
        shorts = [s for s, sd in membership.items() if sd < 0]
        for names, sign in ((longs, 1.0), (shorts, -1.0)):
            iv = {}
            for s in names:
                v = vd.get(s, np.nan)
                if np.isfinite(v) and v > 0:
                    iv[s] = 1.0 / v
            tot = sum(iv.values())
            if tot > 0:
                for s, val in iv.items():
                    w[s] = sign * (val / tot) * 0.5
        W_rows[d] = w

    W_rb = pd.DataFrame(W_rows).T
    W_rb.index = pd.DatetimeIndex(W_rb.index)
    W_rb = W_rb.sort_index()
    Wd   = W_rb.reindex(close.index).ffill().fillna(0.0)

    # --- 10% annualized vol target (computed at rebalance, held weekly) ---
    book = (Wd.shift(1) * rets[cols]).sum(axis=1)
    realized = book.rolling(vol_lb, min_periods=max(5, vol_lb // 2)).std() * np.sqrt(365.0)
    scale = (target_vol / realized).replace([np.inf, -np.inf], np.nan).clip(upper=5.0)
    scale = scale.reindex(W_rb.index).ffill().reindex(close.index).ffill().fillna(1.0)
    Wd = Wd.mul(scale, axis=0)

    # --- residual BTC-beta hedge (small BTC-perp leg, weekly held) ---
    Wfull = pd.DataFrame(0.0, index=close.index, columns=keep)
    Wfull[cols] = Wd[cols]
    if beta_hedge and btc is not None:
        sbook = (Wd.shift(1) * rets[cols]).sum(axis=1)
        bret  = rets[btc]
        cov   = sbook.rolling(beta_lb, min_periods=max(20, beta_lb // 2)).cov(bret)
        var   = bret.rolling(beta_lb, min_periods=max(20, beta_lb // 2)).var()
        beta  = (cov / var)
        beta  = beta.reindex(W_rb.index).ffill().reindex(close.index).ffill()
        beta  = beta.fillna(0.0).clip(-2.0, 2.0)
        Wfull[btc] = -beta

    # --- 1-day execution lag applied HERE; net-of-cost + contract trade ledger ---
    Wlag = Wfull.shift(1).fillna(0.0)
    daily = net_of_cost(Wlag, rets, cost_bps=cost_bps, name="taker_flow_crowding")
    sector_map = _sector_map(keep)
    trades = trades_from_weights(Wlag, rets, sector_map)
    return daily, trades


# ----------------------------------------------------------------------------- data
def load_data():
    pool = _pool(POOL_N)
    syms = pool[0:SEARCH_END]
    btc  = _btc_symbol(pool) or "BTCUSDT"
    use  = list(dict.fromkeys(list(syms) + [btc]))   # BTC reserved for the residual hedge
    return _klines_panel(use)


def load_gen_data(label):
    a, b = GEN_SLICES[label]
    pool = _pool(POOL_N)
    syms = pool[a:b]
    btc  = _btc_symbol(pool) or "BTCUSDT"
    use  = list(dict.fromkeys(list(syms) + [btc]))   # disjoint cross-section + hedge infra
    return _klines_panel(use)


# ----------------------------------------------------------------------------- soft checks
def _chk_dispersion(ctx):
    """Pre-reg: EWMA-TBI has genuine cross-sectional dispersion (not degenerate ~0.5)."""
    p  = ctx["panel"]; ho = pd.Timestamp(ctx["holdout_start"])
    tbi = (p["taker_buy_quote"].astype(float) / p["quote_volume"].astype(float)).clip(0, 1) - 0.5
    flow = tbi.where(np.isfinite(tbi)).ewm(span=5, min_periods=5).mean()
    flow = flow.loc[flow.index < ho]
    med = float(flow.std(axis=1).median())
    return {"pass": med > 0.01, "observed": round(med, 5)}


def _chk_not_reversal(ctx):
    """Pre-reg 'why-not-duplicate': flow is a DIFFERENT axis from price reversal -> the
    per-date cross-sectional rank-corr of flow vs trailing 5d return is only modest."""
    p  = ctx["panel"]; ho = pd.Timestamp(ctx["holdout_start"])
    close = p["close"].astype(float)
    tbi = (p["taker_buy_quote"].astype(float) / p["quote_volume"].astype(float)).clip(0, 1) - 0.5
    flow = tbi.where(np.isfinite(tbi)).ewm(span=5, min_periods=5).mean()
    trail = close.pct_change(5)
    flow = flow.loc[flow.index < ho]; trail = trail.loc[trail.index < ho]
    cors = flow.rank(axis=1).corrwith(trail.rank(axis=1), axis=1)
    med = float(cors.abs().median())
    return {"pass": (np.isfinite(med) and med < 0.6), "observed": round(med, 4)}


def _chk_beta(ctx):
    """Pre-reg / gate0: the BTC-perp hedge drives book beta_to_BTC below the 0.3 confound."""
    p = ctx["panel"]; ho = pd.Timestamp(ctx["holdout_start"]); book = ctx["search"]
    btc = _btc_symbol(p["close"].columns)
    if btc is None or book is None or len(book) == 0:
        return {"pass": False, "observed": "no_btc_or_returns"}
    bret = p["close"][btc].astype(float).pct_change()
    df = pd.concat([book.rename("b"), bret.rename("m")], axis=1).dropna()
    df = df.loc[df.index < ho]
    if len(df) < 60 or df["m"].var() == 0:
        return {"pass": False, "observed": "insufficient"}
    beta = float(np.cov(df["b"], df["m"])[0, 1] / np.var(df["m"]))
    return {"pass": abs(beta) < 0.3, "observed": round(beta, 3)}


def _chk_turnover(ctx):
    """Pre-reg: hysteresis + 5d min-hold SUPPRESS turnover -> fewer trades than the
    no-hysteresis baseline (one extra signal() call, search-window only)."""
    ho = pd.Timestamp(ctx["holdout_start"])
    _, nh = signal(ctx["panel"], exit_q=1.0 / 3.0, min_hold=1)   # bands collapse -> churn
    cnt = lambda tr: sum(1 for t in tr if pd.Timestamp(t["entry_date"]) < ho)
    nb, nn = cnt(ctx["trades"]), cnt(nh)
    if nn == 0:
        return {"pass": False, "observed": "no_baseline_trades"}
    ratio = nb / nn
    return {"pass": ratio <= 0.85, "observed": round(ratio, 3)}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_taker_flow_crowding",
    family="crypto_microstructure_flow",
    title="Crypto taker-flow crowding -> liquidity-provision premium (x-sec, broad perps)",
    markets=["crypto"],
    data_desc=(
        "Binance USDT perpetual-futures daily klines (close, volume, quote_volume, "
        "taker_buy_quote) for a weekly-resnapped liquidity-ranked cross-section (search = "
        "top-50 liquid perps; disjoint deeper-liquidity tiers for generalization). BTC perp "
        "is the residual-beta hedge only (excluded from the signal cross-section). "
        "taker_buy_quote/quote_volume is the catalog's deep-history aggressive-flow proxy. "
        "Owned/free public Binance API ($0)."
    ),
    pre_registration=(
        "PRIMARY (frozen): TBI = taker_buy_quote/quote_volume - 0.5; flow = EWMA_5d(TBI). "
        "Weekly (W-FRI) cross-sectional percentile rank of flow; LONG bottom tercile "
        "(under-bought -> provide liquidity to aggressive/panic sellers), SHORT top tercile "
        "(over-bought, crowded fragile longs); hysteresis enter 1/3, exit at 0.45/0.55 band; "
        "5-day min-hold; inverse-vol within legs (harness-mandatory sizing supersedes the "
        "proposal's equal-weight prose); dollar-neutral; 10% annualized vol target; residual "
        "BTC-beta trimmed via a small BTC-perp leg; net of ~20bps round-trip taker cost "
        "(10bps/side on turnover via net_of_cost). Signals are decided from info up to date d "
        "and executed at d+1 (W.shift(1) lag applied in-module). "
        "MECHANISM: a microstructure RISK premium (paid to bear inventory/crowding risk), "
        "distinct from statistical price mean-reversion (aggressive flow need not coincide "
        "with same-direction price moves when passive size absorbs it). "
        "SCOPE broad: liquidity-provision-to-crowded-flow is universal -> must GENERALISE to "
        "untouched, disjoint, deeper-liquidity perp tiers (>=60% OOS-positive on their "
        "holdouts) or it is an overfit single-tier outlier. The proposal's 2019-2022 vs "
        "2023-2026 time-half test is realised by the harness in-sample/holdout split "
        "(holdout 2022-01-01); it is not a separate machine universe. Cross-asset equity "
        "order-flow confirmation is DEFERRED (clean daily equity order-flow not owned). "
        "HEADWIND acknowledged: if flow empirically collapses to price reversal it inherits "
        "the two prior crypto x-sec reversal nulls; the standalone test and the "
        "flow_not_reversal soft-check are built to falsify, not hide, that. "
        "Standalone first per the 2026-06-08 don't-blend-a-0-Sharpe-leg lesson; any "
        "Boreas-trend tail-overlay or deribit_dvol VRP pairing is a future book needing its "
        "own fresh forward validation."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":        {},
        "ewma3":          {"ewma_span": 3},
        "ewma8":          {"ewma_span": 8},
        "tighter_bands":  {"exit_q": 0.40},
        "no_hysteresis":  {"exit_q": 1.0 / 3.0, "min_hold": 1},
    },
    scope="broad",
    generalization_universes=list(GEN_SLICES.keys()),
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=40,
    expectations=[
        {"name": "flow_dispersion",
         "claim": "EWMA-TBI shows genuine cross-sectional dispersion (median per-date XS std > 0.01), not a degenerate ~0.5 ratio",
         "check": _chk_dispersion},
        {"name": "flow_not_reversal",
         "claim": "Flow is a distinct data axis from price reversal: median |per-date XS rank-corr(flow, trailing-5d-return)| < 0.6",
         "check": _chk_not_reversal},
        {"name": "beta_hedged",
         "claim": "BTC-perp hedge drives search-window book |beta_to_BTC| < 0.3 (below the confound threshold)",
         "check": _chk_beta},
        {"name": "turnover_suppressed",
         "claim": "Hysteresis + 5d min-hold suppress turnover: default search-window trade count <= 85% of the no-hysteresis baseline",
         "check": _chk_turnover},
    ],
)