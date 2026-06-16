# crypto_xs_downside_risk.py
# Crucible strategy module — crypto cross-sectional DOWNSIDE-RISK (co-skewness / downside-beta)
# premium, market-neutral L/S on the broad USDT-perp universe.
#
# MECHANISM (defensive / low-risk anomaly, Harvey-Siddique 2000, Ang-Chen-Xing 2006):
#   crash-prone coins (high downside-beta, very negative co-skewness) are over-bid by lottery-seeking
#   retail and EARN LESS risk-adjusted; the defensive (low-downside-risk) names out-perform.
#   -> LONG lowest-downside-risk tercile, SHORT highest.  Dollar- AND beta-neutral so we isolate the
#   ASYMMETRIC crash-co-movement premium (the part NOT explained by symmetric crypto beta).
#
# NO external side effects (pure functions + SPEC).  All data via OWNED/FREE adapters ($0).
# Only novel code = the signal; costs/ledger/regime-stamping/z-scoring all use the kit.

from sdk.adapters import sep_panel, yf_panel, binance_universe
try:
    from sdk.adapters import binance_klines
except Exception:               # binance_universe may itself return a panel -> klines not required
    binance_klines = None
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
from sdk.harness import StrategySpec
import numpy as np, pandas as pd

SPEC_ID = "crypto_xs_downside_risk"

DEFAULTS = dict(
    lookback=180,        # trailing window (days) for downside-beta / co-skewness
    min_obs=150,         # min valid obs in window for a coin to be scored
    rebalance=7,         # weekly (crypto trades 365d/yr -> 7 rows == 7 calendar days; gives 7d min-hold)
    vol_target=0.10,     # ~10% annualised residual-book vol
    cost_bps=20.0,       # 20bps per turnover unit (conservative crypto taker round-trip)
    min_names=30,        # min cross-section size to rebalance
    entry_q=0.33,        # tercile entry band
    exit_q=0.40,         # hysteresis exit band (cuts boundary churn)
    max_leverage=3.0,
    coskew_weight=0.5,   # blend: (1-w)*z(downside-beta) + w*z(-co-skewness)
)

# ---- crypto "sector" map (asset-class proxy so the trade-ledger sector gate has spread) -------------
CRYPTO_MAP = {
    "BTC": "L1", "ETH": "L1", "SOL": "L1", "ADA": "L1", "AVAX": "L1", "DOT": "L1", "TRX": "L1",
    "ATOM": "L1", "NEAR": "L1", "APT": "L1", "SUI": "L1", "SEI": "L1", "FTM": "L1", "ALGO": "L1",
    "HBAR": "L1", "EOS": "L1", "XTZ": "L1", "ETC": "L1", "KSM": "L1", "FLOW": "L1", "EGLD": "L1",
    "ZIL": "L1", "TON": "L1", "TIA": "Infra", "FIL": "Infra", "ICP": "Infra", "GRT": "Infra",
    "VET": "Infra", "THETA": "Infra", "ROSE": "Infra", "BAT": "Infra", "QNT": "Infra",
    "MATIC": "L2", "ARB": "L2", "OP": "L2", "STX": "L2", "STRK": "L2", "IMX": "L2",
    "UNI": "DeFi", "AAVE": "DeFi", "MKR": "DeFi", "INJ": "DeFi", "CRV": "DeFi", "SNX": "DeFi",
    "COMP": "DeFi", "LDO": "DeFi", "RUNE": "DeFi", "DYDX": "DeFi", "GMX": "DeFi", "SUSHI": "DeFi",
    "1INCH": "DeFi", "KAVA": "DeFi", "ENA": "DeFi", "JUP": "DeFi", "JTO": "DeFi", "ONDO": "DeFi",
    "LINK": "Oracle", "PYTH": "Oracle", "BNB": "Exchange",
    "XRP": "Payment", "LTC": "Payment", "XLM": "Payment", "BCH": "Payment",
    "DOGE": "Meme", "SHIB": "Meme", "PEPE": "Meme", "WIF": "Meme", "FLOKI": "Meme", "BONK": "Meme",
    "ORDI": "Meme", "SAND": "Gaming", "MANA": "Gaming", "AXS": "Gaming", "GALA": "Gaming",
    "CHZ": "Gaming", "ENJ": "Gaming", "FET": "AI", "RNDR": "AI", "AGIX": "AI", "WLD": "AI",
    "ZEC": "Privacy", "DASH": "Privacy", "XMR": "Privacy",
}

# ---- cross-asset ETF generalisation universe -------------------------------------------------------
ETF_LIST = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "EWZ", "FXI", "INDA",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
            "TLT", "IEF", "SHY", "AGG", "BND", "TIP", "LQD", "HYG", "EMB",
            "GLD", "SLV", "USO", "DBC", "DBA", "UNG", "GDX", "VNQ", "IYR"]
ETF_SECTORS = {
    "SPY": "EquityUS", "QQQ": "EquityUS", "IWM": "EquityUS",
    "EFA": "EquityIntl", "VGK": "EquityIntl", "EWJ": "EquityIntl",
    "EEM": "EquityEM", "EWZ": "EquityEM", "FXI": "EquityEM", "INDA": "EquityEM",
    "XLK": "SectorEq", "XLF": "SectorEq", "XLE": "SectorEq", "XLV": "SectorEq", "XLI": "SectorEq",
    "XLP": "SectorEq", "XLY": "SectorEq", "XLU": "SectorEq", "XLB": "SectorEq",
    "TLT": "RatesBond", "IEF": "RatesBond", "SHY": "RatesBond", "AGG": "RatesBond",
    "BND": "RatesBond", "TIP": "RatesBond",
    "LQD": "CreditBond", "HYG": "CreditBond", "EMB": "CreditBond",
    "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity", "DBC": "Commodity",
    "DBA": "Commodity", "UNG": "Commodity", "GDX": "Commodity",
    "VNQ": "RealEstate", "IYR": "RealEstate",
}


# ============================ helpers (non-signal plumbing) =========================================
def _to_close_panel(obj):
    """Coerce an adapter return into a (dates x symbols) close-price DataFrame."""
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    df = obj
    if isinstance(df.columns, pd.MultiIndex):
        for f in ("closeadj", "close", "adj close", "adj_close"):
            for lvl in range(df.columns.nlevels):
                vals = [str(v).lower() for v in df.columns.get_level_values(lvl)]
                if f in vals:
                    sel = [str(v).lower() == f for v in df.columns.get_level_values(lvl)]
                    sub = df.loc[:, sel].copy()
                    sub.columns = sub.columns.droplevel(lvl)
                    return sub
        return df.droplevel(list(range(df.columns.nlevels - 1)), axis=1)
    return df


def _crypto_sectors(cols):
    out = {}
    for c in cols:
        b = str(c).upper()
        for suf in ("USDT", "USDC", "BUSD", "USD"):
            if b.endswith(suf):
                b = b[:-len(suf)]
                break
        b = b.replace("PERP", "").replace("/", "").replace(":", "").replace("-", "")
        b = b.lstrip("0123456789")
        out[c] = CRYPTO_MAP.get(b, "Altcoin")
    return out


def _scores(rw, mw):
    """Per-coin trailing-window stats vs equal-weight market factor (Series indexed by columns)."""
    nobs = rw.notna().sum()
    mu = mw.mean(); em = mw - mu
    varm = (em ** 2).mean(); sigm = mw.std()
    ei = rw.sub(rw.mean()); sig = rw.std()
    beta = ei.mul(em, axis=0).mean() / varm
    cosk = ei.mul(em ** 2, axis=0).mean() / (sig * (sigm ** 2))      # Harvey-Siddique co-skewness
    dm = mw < mu                                                      # market down-state
    rwd, mwd = rw.loc[dm], mw.loc[dm]
    emd = mwd - mwd.mean(); vmd = (emd ** 2).mean()
    dbeta = rwd.sub(rwd.mean()).mul(emd, axis=0).mean() / vmd        # Ang-Chen-Xing downside beta
    return dbeta, cosk, beta, sig, nobs


def _xsz(s):
    """Cross-sectional winsorised z of one cross-section, via the kit (no temporal axis -> no lookahead)."""
    return xs_zscore(s.to_frame().T).iloc[0]


def _residualize(w, beta):
    """Orthogonalise weights against [1, beta] -> dollar-neutral AND beta-neutral (isolates asymmetry)."""
    X = np.column_stack([np.ones_like(beta), beta])
    try:
        coef, *_ = np.linalg.lstsq(X, w, rcond=None)
        return w - X @ coef
    except Exception:
        return w - w.mean()


# ================================ data loaders =====================================================
def load_data():
    try:
        out = binance_universe(75, market='perp')
    except TypeError:
        out = binance_universe(75)
    # binance_universe may return a price panel directly OR a list of symbols
    if isinstance(out, (pd.DataFrame, pd.Series)):
        px = _to_close_panel(out)
    else:
        if binance_klines is None:
            raise RuntimeError("binance_universe returned symbols but binance_klines is unavailable")
        syms = list(out)
        try:
            raw = binance_klines(syms, start="2019-09-01")
        except TypeError:
            raw = binance_klines(syms)
        px = _to_close_panel(raw)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().dropna(how='all').dropna(axis=1, how='all')
    px.attrs["sector_map"] = _crypto_sectors(list(px.columns))
    return px


def load_gen_data(label):
    """Panel for ONE generalisation universe — same shape as load_data() (dates x tickers, close)."""
    if label == "us_small_cap":
        tk, smap = sector_universe(marketcap='Small', top_n_per_sector=20)
        px = sep_panel(tk, start="2016-01-01")
    elif label == "us_mid_cap":
        tk, smap = sector_universe(marketcap='Mid', top_n_per_sector=20)
        px = sep_panel(tk, start="2016-01-01")
    elif label == "cross_asset_etf":
        smap = dict(ETF_SECTORS)
        px = yf_panel(ETF_LIST, start="2010-01-01")
    else:
        raise ValueError(f"unknown generalization universe: {label}")
    px = _to_close_panel(px)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().dropna(how='all').dropna(axis=1, how='all')
    px.attrs["sector_map"] = {t: smap.get(t, "NA") for t in px.columns}
    return px


# ================================== the signal =====================================================
def signal(panel, **params):
    cfg = {**DEFAULTS, **params}
    lb, min_obs, reb = int(cfg["lookback"]), int(cfg["min_obs"]), int(cfg["rebalance"])
    vt, cost, min_names = float(cfg["vol_target"]), float(cfg["cost_bps"]), int(cfg["min_names"])
    eq, xq = float(cfg["entry_q"]), float(cfg["exit_q"])
    maxlev, cw = float(cfg["max_leverage"]), float(cfg["coskew_weight"])

    px = panel.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    rets = px.pct_change().clip(-0.75, 0.75)        # cap data-glitch jumps; daily close-to-close
    mkt = rets.mean(axis=1, skipna=True)            # PRE-REGISTERED market factor: equal-weight universe

    smap = dict(panel.attrs.get("sector_map") or {})
    if not smap:
        smap = _crypto_sectors(list(px.columns))

    dates, cols = rets.index, rets.columns
    yrs = max((dates[-1] - dates[0]).days / 365.25, 1e-6)
    annf = len(dates) / yrs                          # obs/yr -> ~365 crypto, ~252 equities (auto)

    wdict, prev_long, prev_short = {}, set(), set()

    for i in range(lb, len(dates), reb):
        d = dates[i]
        w = rets.iloc[i - lb + 1:i + 1]             # trailing window, info through day d (same-day weights)
        mw = mkt.loc[w.index]
        dbeta, cosk, beta, sig, nobs = _scores(w, mw)
        ok = (nobs >= min_obs) & np.isfinite(dbeta) & np.isfinite(cosk) & np.isfinite(beta) & (sig > 0)
        okb = ok.fillna(False)
        names = [c for c in cols if bool(okb.get(c, False))]
        if len(names) < min_names:
            continue                                # hold previous (ffilled below)

        risk = (1 - cw) * _xsz(dbeta[names]) + cw * _xsz(-cosk[names])   # higher = more downside risk
        risk = risk.dropna()
        if len(risk) < min_names:
            continue
        names = list(risk.index)
        rk = risk.rank(pct=True)

        long_set = set(rk.index[rk <= eq]) | (prev_long & set(rk.index[rk <= xq]))
        short_set = set(rk.index[rk >= 1 - eq]) | (prev_short & set(rk.index[rk >= 1 - xq]))
        long_set -= short_set
        long_names = [n for n in names if n in long_set]
        short_names = [n for n in names if n in short_set]
        prev_long, prev_short = set(long_names), set(short_names)
        if len(long_names) < 3 or len(short_names) < 3:
            continue

        inv = (1.0 / sig)                           # equal-RISK (inverse-vol) within each leg
        wl = inv.reindex(long_names); wl = wl / wl.sum()
        ws = inv.reindex(short_names); ws = -ws / ws.sum()
        active = long_names + short_names
        wv = pd.Series(0.0, index=active)
        wv.loc[long_names] = wl.values
        wv.loc[short_names] = ws.values

        wp = _residualize(wv.values, beta.reindex(active).astype(float).values)   # $- & beta-neutral
        port = (w[active] * pd.Series(wp, index=active)).sum(axis=1)              # window book returns
        rv = port.std() * np.sqrt(annf)
        scale = 0.0 if (not np.isfinite(rv) or rv <= 0) else min(vt / rv, maxlev)

        row = pd.Series(0.0, index=cols)
        row.loc[active] = wp * scale
        wdict[d] = row

    if wdict:
        Wreb = pd.DataFrame(wdict).T.reindex(columns=cols).astype(float)
        W = Wreb.reindex(dates).ffill().fillna(0.0)
    else:
        W = pd.DataFrame(0.0, index=dates, columns=cols)

    # weights are built with info THROUGH day d (same-day) -> LAG 1 day to avoid look-ahead.
    Wlag = W.shift(1).fillna(0.0)
    daily = net_of_cost(Wlag, rets, cost_bps=cost, name=SPEC_ID)
    trades = trades_from_weights(Wlag, rets, smap)
    return daily, trades


# ============================ soft expectations (machine-checkable) =================================
def _ann_sharpe(s, hs):
    s = s.copy(); s.index = pd.to_datetime(s.index)
    s = s[s.index < pd.Timestamp(hs)].dropna()
    if len(s) < 30 or float(s.std()) == 0.0:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(365.0))


def _check_market_neutral(ctx):
    """CLAIM: beta-neutralised L/S book has low correlation to the equal-weight crypto market."""
    panel, sr = ctx["panel"], ctx["search"]
    hs = pd.Timestamp(ctx["holdout_start"])
    px = panel.copy(); px.index = pd.to_datetime(px.index); px = px.sort_index()
    mkt = px.pct_change().clip(-0.75, 0.75).mean(axis=1, skipna=True)
    sr = sr.copy(); sr.index = pd.to_datetime(sr.index)
    df = pd.concat([sr[sr.index < hs], mkt], axis=1).dropna()
    if len(df) < 30:
        return {"pass": False, "observed": "insufficient overlap"}
    rho = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return {"pass": abs(rho) <= 0.35, "observed": round(rho, 3)}


def _check_both_legs_positive(ctx):
    """CLAIM: BOTH single-signal sorts (downside-beta, co-skewness) earn positive search-window returns."""
    grid = ctx.get("grid", {})
    db, ck = grid.get("db_only"), grid.get("cosk_only")
    if db is None or ck is None:
        return {"pass": False, "observed": "grid variants missing"}
    sdb, sck = _ann_sharpe(db, ctx["holdout_start"]), _ann_sharpe(ck, ctx["holdout_start"])
    ok = bool(np.isfinite(sdb) and np.isfinite(sck) and sdb > 0 and sck > 0)
    return {"pass": ok, "observed": f"db_sharpe={sdb:.2f}, cosk_sharpe={sck:.2f}"}


# ===================================== the SPEC ====================================================
SPEC = StrategySpec(
    id=SPEC_ID,
    family="defensive_downside_risk",
    title="Crypto cross-sectional downside-risk (co-skewness / downside-beta) market-neutral L/S",
    markets=["crypto"],
    data_desc=("Broad Binance USDT-perp cross-section (~75 most-liquid names, delisting-inclusive via "
               "binance_universe). Daily close-to-close returns; equal-weight universe as the market "
               "factor. No fundamentals."),
    pre_registration=(
        "Defensive low-risk anomaly (Harvey-Siddique 2000 co-skewness; Ang-Chen-Xing 2006 downside "
        "beta). Crash-prone coins (high downside-beta / very negative co-skewness) are over-bid by "
        "lottery-seeking retail and underperform risk-adjusted. Score = blend of cross-sectional z of "
        "downside-beta and z of (-co-skewness) over a 180d trailing window vs the equal-weight market "
        "factor. LONG lowest-risk tercile, SHORT highest, inverse-vol within leg, then residualise the "
        "weight vector against [1, full-sample-window beta] so the book is dollar- AND beta-neutral "
        "(isolating the ASYMMETRIC crash-co-movement premium, not symmetric beta). Weekly rebalance "
        "with tercile-entry / 0.40 hysteresis-exit to cut churn; 20bps round-trip cost; ~10% vol "
        "target, 3x leverage cap. Signals use info through day d and are LAGGED 1 day (W.shift(1)) "
        "before returns/ledger. BROAD scope: a universal downside-risk premium should also appear in "
        "untouched markets (US small/mid-cap equities, cross-asset ETFs) -> stage-2 generalisation. "
        "Expected: market-neutral (|corr to crypto market| <= 0.35) and both single-signal legs "
        "positive standalone in the search window."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "db_only": {"coskew_weight": 0.0},
        "cosk_only": {"coskew_weight": 1.0},
        "lb_120": {"lookback": 120, "min_obs": 100},
        "lb_252": {"lookback": 252, "min_obs": 210},
    },
    scope="broad",
    generalization_universes=["us_small_cap", "us_mid_cap", "cross_asset_etf"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "market_neutral",
         "claim": "search-window |corr| of book returns to equal-weight crypto market <= 0.35",
         "check": _check_market_neutral},
        {"name": "both_legs_positive",
         "claim": "downside-beta-only AND co-skewness-only sorts each earn positive search Sharpe",
         "check": _check_both_legs_positive},
    ],
)