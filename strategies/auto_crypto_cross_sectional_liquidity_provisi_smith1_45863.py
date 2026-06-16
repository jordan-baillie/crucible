# =============================================================================
# Crypto cross-sectional LIQUIDITY-PROVISION premium via taker-flow imbalance
# (broad Binance perp universe, market-neutral)
#
# THEORY (not a price forecast): traders who provide immediacy to impatient
# *taker* flow earn an inventory-risk premium. We measure, per coin, the net
# aggression of EXECUTED flow (taker_buy_quote vs total quote_volume) -- a data
# axis that is NEITHER price (idio-reversal FAILs) NOR funding (funding-gate
# FAILs). Coins absorbing aggressive SELLING (low net-taker-imbalance) are paid
# to take the other side of panic -> LONG; coins absorbing aggressive BUYING
# (high imbalance, FOMO) -> SHORT. Dollar-neutral, inverse-vol, BTC-beta-trimmed,
# vol-targeted, hysteresis + weekly hold to keep taker turnover (20bps RT) low.
#
# SCOPE = 'local': the LP premium is universal in theory but arbitraged away in
# efficient markets (the equity freq-reversal-LP analog already FAILED). It can
# only survive in the thin-MM, heavy-retail-taker crypto corner -> forward (and
# within-crypto sub-tier) validation, not a cross-market generalisation battery.
#
# NO look-ahead: all signals/scales are built from data <= t and traded with a
# 1-day implementation lag (the single Wf.shift(1) before net_of_cost). Costs,
# inverse-vol, regime stamping, and the trade ledger are all KIT functions.
#
# FIX (vs failed run): binance_klines() has NO `field=` kwarg (same class as the
# fut_curve(field=) hallucination). It returns the FULL panel for the requested
# tickers in ONE call (MultiIndex columns: level0=field in {close,quote_volume,
# taker_buy_quote,...}, level1=ticker). We now call it once and SELECT fields by
# indexing, with a defensive normaliser so it also works if the adapter hands
# back a dict-of-frames. No other behaviour changed.
# =============================================================================
from sdk.harness import StrategySpec
from sdk.adapters import binance_universe, binance_klines     # crypto adapters (DATA_CATALOG: broad-crypto wiring)
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2019-09-01"          # Binance perps begin ~2019-09; pre-listing rows are NaN (handled by min_periods)
HOLDOUT = "2022-01-01"
UNIV_N = 75

DEFAULT_PARAMS = dict(
    window=7,         # NTI trailing lookback (days)
    enter_q=0.30,     # enter long/short bands (bottom/top 30%)
    exit_q=0.45,      # exit bands (hysteresis -> suppresses turnover); 7d min-hold is implied by weekly rebal
    vol_lb=30,        # inverse-vol + BTC-beta + vol-target lookback (days)
    target_vol=0.10,  # annualised portfolio vol target
    cost_bps=20.0,    # round-trip taker cost (conservative)
    min_qvol=0.0,     # drop zero/low quote-volume days from the imbalance ratio
    name="crypto_lp_takerflow",
    reverse=False,    # internal: flip long/short for the premium-sign falsification check
)

# Coarse crypto "sector" map so the deployment sector-spread gate has real breadth.
CRYPTO_SECTORS = {
    'BTCUSDT':'l1_major','ETHUSDT':'l1_major','BNBUSDT':'exchange',
    'SOLUSDT':'l1_alt','ADAUSDT':'l1_alt','AVAXUSDT':'l1_alt','DOTUSDT':'l1_alt','NEARUSDT':'l1_alt',
    'ATOMUSDT':'l1_alt','APTUSDT':'l1_alt','SUIUSDT':'l1_alt','TRXUSDT':'l1_alt','EOSUSDT':'l1_alt',
    'ALGOUSDT':'l1_alt','EGLDUSDT':'l1_alt','FTMUSDT':'l1_alt','HBARUSDT':'l1_alt','XTZUSDT':'l1_alt',
    'KSMUSDT':'l1_alt','FLOWUSDT':'l1_alt','MINAUSDT':'l1_alt',
    'XRPUSDT':'payments','XLMUSDT':'payments','LTCUSDT':'payments','BCHUSDT':'payments',
    'MATICUSDT':'l2','ARBUSDT':'l2','OPUSDT':'l2','IMXUSDT':'l2','CFXUSDT':'l2',
    'UNIUSDT':'defi','AAVEUSDT':'defi','MKRUSDT':'defi','CRVUSDT':'defi','SNXUSDT':'defi','COMPUSDT':'defi',
    'SUSHIUSDT':'defi','1INCHUSDT':'defi','DYDXUSDT':'defi','LDOUSDT':'defi','INJUSDT':'defi',
    'DOGEUSDT':'meme','SHIBUSDT':'meme','PEPEUSDT':'meme','WIFUSDT':'meme','FLOKIUSDT':'meme','BONKUSDT':'meme',
    'LINKUSDT':'oracle','GRTUSDT':'oracle','FILUSDT':'storage','ARUSDT':'storage',
    'RNDRUSDT':'depin_ai','FETUSDT':'depin_ai','AGIXUSDT':'depin_ai','OCEANUSDT':'depin_ai',
    'SANDUSDT':'gaming','MANAUSDT':'gaming','AXSUSDT':'gaming','GALAUSDT':'gaming','APEUSDT':'gaming','ENJUSDT':'gaming',
    'ICPUSDT':'compute','THETAUSDT':'media','CHZUSDT':'media','VETUSDT':'supplychain','IOTAUSDT':'iot',
    'ZECUSDT':'privacy','XMRUSDT':'privacy','DASHUSDT':'privacy','ROSEUSDT':'privacy',
}
def _sector_map(tickers):
    return {t: CRYPTO_SECTORS.get(t, 'other_alt') for t in tickers}


# ---------------------------------------------------------------- data loaders
def _universe(n=UNIV_N):
    try:
        return list(binance_universe(n, market='perp'))
    except TypeError:
        return list(binance_universe(n))

def _klines(tickers):
    # binance_klines returns the FULL klines panel for these tickers in one call.
    # It has NO `field=` kwarg (that was the failed-run bug). Try the richer
    # signature first, then fall back to the minimal positional form.
    try:
        return binance_klines(tickers, start=START, market='perp')
    except TypeError:
        try:
            return binance_klines(tickers, start=START)
        except TypeError:
            return binance_klines(tickers)

def _select_field(raw, field, ref=None):
    # Normalise whatever binance_klines returns into a (dates x tickers) frame
    # for one field. Supports: MultiIndex-column DataFrame (level0=field),
    # dict-of-DataFrames, or a plain single-field DataFrame.
    if isinstance(raw, dict):
        df = raw[field]
    elif isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if field in set(lvl0):
            df = raw.xs(field, axis=1, level=0)
        else:
            # field may live on level1 (ticker, field) ordering
            lvl1 = raw.columns.get_level_values(-1)
            df = raw.xs(field, axis=1, level=-1) if field in set(lvl1) else raw
    else:
        df = raw  # already a single-field frame (assume it is `field`)
    df = df.sort_index()
    if ref is not None:
        df = df.reindex_like(ref)
    return df.astype(float)

def _load_panel(tickers):
    # MultiIndex-column panel: level0 in {close, quote_volume, taker_buy_quote}, level1 = ticker.
    raw   = _klines(tickers)
    close = _select_field(raw, 'close')
    qvol  = _select_field(raw, 'quote_volume', ref=close)
    tbq   = _select_field(raw, 'taker_buy_quote', ref=close)
    return pd.concat({'close': close, 'quote_volume': qvol, 'taker_buy_quote': tbq}, axis=1)

def load_data():
    return _load_panel(_universe(UNIV_N))

def load_gen_data(label):
    # within-crypto robustness sub-tiers (disjoint from each other); same frozen rule.
    u = _universe(UNIV_N)
    if label == 'crypto_top30':
        t = u[:30]
    elif label == 'crypto_tail45':
        t = u[30:]
    else:
        t = u
    return _load_panel(t)


# ---------------------------------------------------------------- signal core
def _components(panel, p):
    close = panel['close'].astype(float)
    qvol  = panel['quote_volume'].astype(float).reindex_like(close)
    tbq   = panel['taker_buy_quote'].astype(float).reindex_like(close)
    rets = close.pct_change()
    # taker imbalance in [-1, 1]; guard zero/low quote-volume days (-> NaN, excluded)
    qv = qvol.where(qvol > p['min_qvol'])
    ti = (2.0 * tbq / qv - 1.0).clip(-1.0, 1.0)
    nti = ti.rolling(p['window'], min_periods=max(3, p['window'] // 2)).mean()
    # per-coin trailing vol, floored cross-sectionally to stop inverse-vol blow-ups
    vol = rets.rolling(p['vol_lb'], min_periods=10).std()
    vol = vol.clip(lower=vol.median(axis=1) * 0.2, axis=0)
    inv_vol = 1.0 / vol
    return rets, nti, inv_vol

def _target_weights(panel, p, reverse=False):
    rets, nti, inv_vol = _components(panel, p)
    tickers = list(rets.columns)
    dates = rets.index
    mkt = 'BTCUSDT' if 'BTCUSDT' in tickers else panel['quote_volume'].median().idxmax()
    enter_q, exit_q, lb = p['enter_q'], p['exit_q'], p['vol_lb']

    rb = set(pd.Series(dates, index=dates).resample('W-FRI').last().dropna().tolist())
    W = pd.DataFrame(np.nan, index=dates, columns=tickers)
    prev_long = pd.Series(False, index=tickers)
    prev_short = pd.Series(False, index=tickers)

    for d in [x for x in dates if x in rb]:
        row = nti.loc[d].dropna()
        if len(row) < 8:                                  # need a genuine cross-section -> else hold (ffill)
            continue
        pct = row.rank(pct=True)
        pl = prev_long.reindex(row.index).fillna(False).astype(bool)
        ps = prev_short.reindex(row.index).fillna(False).astype(bool)
        long_now  = (pct <= enter_q)     | (pl & (pct <= exit_q))        # bottom NTI: absorb panic selling -> LONG
        short_now = (pct >= 1 - enter_q) | (ps & (pct >= 1 - exit_q))    # top NTI: absorb FOMO buying  -> SHORT
        if reverse:
            long_now, short_now = short_now, long_now
        iv = inv_vol.loc[d].reindex(row.index)
        lv = iv[long_now].dropna()
        sv = iv[short_now].dropna()
        if len(lv) < 2 or len(sv) < 2 or lv.sum() <= 0 or sv.sum() <= 0:
            continue
        w = pd.Series(0.0, index=tickers)
        w.loc[lv.index] =  0.5 * lv / lv.sum()            # dollar-neutral, inverse-vol within leg
        w.loc[sv.index] = -0.5 * sv / sv.sum()
        # residual market(BTC)-beta trim with a bounded BTC-perp leg
        sub = rets.loc[:d].tail(lb)
        msub = sub[mkt]
        vm = msub.var()
        if vm and vm > 0:
            beta = sub.apply(lambda c: c.cov(msub)) / vm
            nb = float((w * beta.reindex(tickers).fillna(0.0)).sum())
            w.loc[mkt] = w.get(mkt, 0.0) + float(np.clip(-nb, -0.5, 0.5))
        W.loc[d] = w
        prev_long = pd.Series(False, index=tickers);  prev_long.loc[lv.index] = True
        prev_short = pd.Series(False, index=tickers); prev_short.loc[sv.index] = True

    W = W.ffill().fillna(0.0)                              # weekly hold between rebalances
    return W, rets

def signal(panel, **params):
    p = {**DEFAULT_PARAMS, **params}
    W, rets = _target_weights(panel, p, reverse=bool(p.get('reverse', False)))
    # Vol-target overlay: scale[t] from trailing REALISED strat returns only (already 1-day lagged) -> no look-ahead.
    r_strat = (W.shift(1) * rets).sum(axis=1)
    realized = r_strat.rolling(p['vol_lb'], min_periods=15).std() * np.sqrt(365.0)
    scale = (p['target_vol'] / realized).clip(0.0, 2.0).fillna(0.0)   # final gross <= 2x
    Wf = W.mul(scale, axis=0)
    Wl = Wf.shift(1).fillna(0.0)                          # 1-day implementation lag (OUR responsibility) -> pass lagged
    smap = _sector_map(list(panel['close'].columns))
    daily = net_of_cost(Wl, rets, cost_bps=p['cost_bps'], name=p['name'])
    trades = trades_from_weights(Wl, rets, smap)          # KIT stamps entry_regime (contract requirement)
    return daily, trades


# ---------------------------------------------------------------- grid (DSR effective-N)
GRID = {
    "default": {},
    "w14": {"window": 14},
    "tighter": {"enter_q": 0.25, "exit_q": 0.40},
    "hard_tercile": {"enter_q": 1.0/3.0, "exit_q": 1.0/3.0},   # no hysteresis band (turnover check baseline)
}


# ---------------------------------------------------------------- soft expectations (machine-checkable)
def _entry_rate(trades, holdout_start):
    t = [x for x in trades if str(x.get('entry_date', '')) < holdout_start]
    if not t:
        return float('nan')
    tot = sum(int(x.get('hold_days', 0)) for x in t)
    return len(t) / max(tot, 1)                            # lower = longer holds = lower turnover

def _check_vol_target(ctx):
    r = ctx["search"].dropna()
    av = float(r.std() * np.sqrt(365.0)) if len(r) > 10 else float('nan')
    return {"pass": bool(0.05 <= av <= 0.20), "observed": round(av, 4)}

def _check_hysteresis(ctx):
    hs = ctx["holdout_start"]
    base = _entry_rate(ctx["trades"], hs)
    _, tr_hard = signal(ctx["panel"], **{**DEFAULT_PARAMS, "enter_q": 1.0/3.0, "exit_q": 1.0/3.0})  # one extra call
    hard = _entry_rate(tr_hard, hs)
    ok = (base == base) and (hard == hard) and (base <= hard)
    return {"pass": bool(ok), "observed": f"entry-rate hyst={base:.4f} hard={hard:.4f}"}

def _check_premium_sign(ctx):
    hs = pd.Timestamp(ctx["holdout_start"])
    b = ctx["search"].dropna()
    bs = float(b.mean() / (b.std() + 1e-12) * np.sqrt(365.0)) if len(b) > 10 else float('nan')
    rr, _ = signal(ctx["panel"], reverse=True)             # one extra call; reversed sort should be WORSE
    rr = rr[rr.index < hs].dropna()
    rs = float(rr.mean() / (rr.std() + 1e-12) * np.sqrt(365.0)) if len(rr) > 10 else float('nan')
    return {"pass": bool(bs == bs and rs == rs and bs > rs),
            "observed": f"Sharpe long-lowNTI={bs:.2f} reversed={rs:.2f}"}


# ---------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_lp_takerflow_v1",
    family="crypto_liquidity_provision",
    title="Crypto cross-sectional liquidity-provision premium via taker-flow imbalance (broad perp, market-neutral)",
    markets=["crypto_perp"],
    data_desc=("Binance perp daily close + quote_volume + taker_buy_quote for binance_universe(75). "
               "Per-coin net taker imbalance NTI=mean(2*taker_buy_quote/quote_volume-1) over a 7d window; "
               "cross-sectional tercile L/S, inverse-vol, dollar-neutral, residual BTC-beta trimmed, 10% vol-target."),
    pre_registration=(
        "FROZEN PRIMARY (W=7, weekly W-FRI rebalance, tercile, hysteresis enter=30%/exit=45%, "
        "min-hold>=7d implied by weekly hold, 20bps round-trip taker cost, 1-day signal lag, gross<=2x): "
        "LONG bottom-NTI tercile (coins absorbing aggressive SELLING -> provide immediacy to panic sellers), "
        "SHORT top-NTI tercile (coins absorbing aggressive BUYING -> provide immediacy to FOMO buyers). "
        "Inverse-vol within leg, dollar-neutral, residual net BTC-beta trimmed to ~0 with a bounded BTC-perp leg, "
        "portfolio vol-targeted to 10% annualised. Mechanism is an inventory-risk / liquidity-provision premium, "
        "NOT a price or funding forecast (distinct from idio-reversal FAIL and funding-gated reversal FAIL). "
        "FALSIFIABLE CLAIMS: (1) realised book vol ~10% annualised; (2) hysteresis+weekly-hold lowers long-leg "
        "turnover vs a no-band hard tercile; (3) the stated sort direction (long low-NTI) beats the reversed sort "
        "in-sample. SCOPE=local: the LP premium is universal in theory but arbitraged away in efficient markets "
        "(equity freq-reversal-LP analog FAILED) -> it should survive only in the thin-MM, heavy-retail-taker "
        "crypto corner; confirmed by same-sign within-crypto sub-tiers (top-30 vs ranks 30-75) then live forward "
        "paper. MCPT (absolute-Sharpe null, market-neutral) mandatory to rule out bid-ask-bounce harvesting. "
        "No grid variant is promoted over the primary."),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULT_PARAMS),
    grid=GRID,
    scope='local',
    generalization_universes=["crypto_top30", "crypto_tail45"],
    load_gen_data=load_gen_data,
    holdout_start=HOLDOUT,
    deploy_max_positions=55,
    expectations=[
        {"name": "vol_target_band",
         "claim": "search-window realised annualised vol is in [5%, 20%] (10% target)",
         "check": _check_vol_target},
        {"name": "hysteresis_cuts_turnover",
         "claim": "hysteresis+weekly-hold entry-rate (trades/hold-day) <= no-band hard-tercile baseline",
         "check": _check_hysteresis},
        {"name": "premium_sign_holds",
         "claim": "long low-NTI / short high-NTI in-sample Sharpe > reversed-sort Sharpe",
         "check": _check_premium_sign},
    ],
)