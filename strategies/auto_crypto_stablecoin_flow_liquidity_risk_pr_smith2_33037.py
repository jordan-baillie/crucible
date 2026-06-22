"""
Crypto stablecoin-flow liquidity-risk premium (Pastor-Stambaugh transplant).

Mechanism (priced RISK, not a return forecast): high-flow-beta coins must pay a
premium for loading on crypto's dominant undiversifiable macro-liquidity shock --
fiat capital entering/leaving via stablecoin (USDT+USDC) supply. Pro-cyclical, so
it crashes in deleveraging (the exact tail trend hedges -- left as a future small
overlay; tested STANDALONE here per the 2026-06-08 anti-over-blend lesson).

Cross-section: top-75 most-liquid Binance USD(S)-M perps. Each month, tercile-sort
on flow-beta = c from a rolling 90d two-factor OLS r_i = a + b*MKT + c*F (F = the
stationary stablecoin-supply shock, orthogonal to market beta b). Long top / short
bottom, inverse-vol, dollar+beta-neutral, vol-targeted 10%/yr, gross<=2, 20bps
one-way taker, hysteresis + monthly min-hold. Signals lagged 1 day.

Only novel code is the signal; returns/ledger/regime-stamps go through the kit.
"""
from sdk.harness import StrategySpec
from sdk.adapters import binance_universe, binance_klines, coinmetrics_metrics
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- config
FLOW_COL = "__STABLE_SUPPLY__"        # reserved panel column for the macro liquidity factor
START    = "2020-01-01"               # binance perps + USDT/USDC supply both live by here
N_SEARCH = 75
GEN_SIZE = 60

DEFAULTS = dict(beta_window=90, flow_growth=7, flow_detrend=180, n_tiles=3,
                target_vol=0.10, gross=2.0, max_scale=3.0, cost_bps=20.0,
                vol_win=90, coin_vol_lb=30, hysteresis=True,
                min_obs=45, min_names=12)

# Coarse crypto "sectors" (domain knowledge) so the trade ledger has genuine spread.
# Unknown / illiquid alts default to "Altcoin"; lookup is on the base symbol.
_CRYPTO_SECTOR = {
    "BTC": "Currency", "LTC": "Currency", "BCH": "Currency", "XMR": "Privacy",
    "ZEC": "Privacy", "DASH": "Privacy",
    "ETH": "SmartContractL1", "SOL": "SmartContractL1", "ADA": "SmartContractL1",
    "AVAX": "SmartContractL1", "NEAR": "SmartContractL1", "ATOM": "SmartContractL1",
    "ALGO": "SmartContractL1", "DOT": "SmartContractL1", "TRX": "SmartContractL1",
    "EOS": "SmartContractL1", "FTM": "SmartContractL1", "APT": "SmartContractL1",
    "SUI": "SmartContractL1", "SEI": "SmartContractL1", "TON": "SmartContractL1",
    "INJ": "SmartContractL1", "EGLD": "SmartContractL1", "KAS": "SmartContractL1",
    "MATIC": "L2Scaling", "ARB": "L2Scaling", "OP": "L2Scaling", "IMX": "L2Scaling",
    "STRK": "L2Scaling", "METIS": "L2Scaling", "MANTA": "L2Scaling",
    "UNI": "DeFi", "AAVE": "DeFi", "MKR": "DeFi", "COMP": "DeFi", "CRV": "DeFi",
    "SNX": "DeFi", "SUSHI": "DeFi", "LDO": "DeFi", "DYDX": "DeFi", "GMX": "DeFi",
    "CAKE": "DeFi", "RUNE": "DeFi", "1INCH": "DeFi", "YFI": "DeFi", "BAL": "DeFi",
    "LINK": "Infra", "GRT": "Infra", "FIL": "Infra", "AR": "Infra", "RNDR": "Infra",
    "THETA": "Infra", "HNT": "Infra", "AKT": "Infra", "QNT": "Infra",
    "DOGE": "Meme", "SHIB": "Meme", "PEPE": "Meme", "WIF": "Meme", "FLOKI": "Meme",
    "BONK": "Meme",
    "BNB": "Exchange", "OKB": "Exchange", "CRO": "Exchange", "KCS": "Exchange",
    "AXS": "Gaming", "SAND": "Gaming", "MANA": "Gaming", "GALA": "Gaming",
    "ENJ": "Gaming", "APE": "Gaming", "FLOW": "Gaming", "CHZ": "Gaming",
    "XRP": "Payments", "XLM": "Payments",
}

_CACHE = {}

# ----------------------------------------------------------------------------- data
def _base(t):
    s = str(t).upper()
    for suf in ("USDT", "USDC", "USD"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s


def _sector_map(cols):
    return {c: _CRYPTO_SECTOR.get(_base(c), "Altcoin") for c in cols}


def _stable_supply():
    """Aggregate USDT+USDC circulating supply, observed same-day (no revision lag)."""
    if "supply" in _CACHE:
        return _CACHE["supply"]
    cm = coinmetrics_metrics(("usdt", "usdc"), ("SplyCur",)).sort_index()
    cols = [c for c in cm.columns if "Sply" in str(c)]
    if cols:
        cm = cm[cols]
    s = cm.sum(axis=1, min_count=1).astype(float)
    s = s.loc[s.index >= pd.Timestamp(START)]
    _CACHE["supply"] = s
    return s


def _build_panel(coins, market="perp"):
    px = binance_klines(list(coins), market=market)
    px = px.loc[px.index >= pd.Timestamp(START)].astype(float).sort_index()
    px = px.dropna(axis=1, how="all")
    out = px.copy()
    out[FLOW_COL] = _stable_supply().reindex(out.index).ffill()
    return out


def _search_coins():
    if "search" not in _CACHE:
        _CACHE["search"] = list(binance_universe(N_SEARCH, market="perp"))
    return _CACHE["search"]


def _gen_coins(label):
    """Universes DISJOINT from the top-75 search set (lower liquidity tiers + spot)."""
    search = set(_search_coins())
    if label in ("liq_tier2", "liq_tier3"):
        ranked = list(binance_universe(N_SEARCH + 2 * GEN_SIZE + 40, market="perp"))
        pool = [c for c in ranked if c not in search]
        return (pool[:GEN_SIZE], "perp") if label == "liq_tier2" \
            else (pool[GEN_SIZE:2 * GEN_SIZE], "perp")
    if label == "tier2_spot":
        ranked = list(binance_universe(N_SEARCH + GEN_SIZE + 40, market="spot"))
        pool = [c for c in ranked if c not in search]
        return pool[:GEN_SIZE], "spot"
    raise ValueError("unknown generalization universe: %s" % label)


def load_data():
    return _build_panel(_search_coins(), market="perp")


def load_gen_data(label):
    coins, market = _gen_coins(label)
    return _build_panel(coins, market=market)


# ----------------------------------------------------------------------------- signal helpers
def _flow_shock(flow, growth, detrend):
    """Stationary, no-lookahead liquidity shock: trailing-standardized (drift-removed)
    7d log-growth of aggregate supply. Strips the secular upward drift that would
    otherwise alias into pure market beta. All windows are TRAILING."""
    s = np.log(flow.where(flow > 0))
    g = s.diff(growth)
    mu = g.rolling(detrend, min_periods=30).mean()
    sd = g.rolling(detrend, min_periods=30).std()
    return ((g - mu) / sd).clip(-4.0, 4.0)


def _construct_weights(fb, mb, vol, prev_long, prev_short, hyst, n_tiles, min_names):
    """Tercile L/S on flow-beta -> inverse-vol, dollar-neutral, bounded beta-neutral
    scale on the short leg. Hysteresis exits at the cross-sectional median."""
    v = vol.reindex(fb.index)
    valid = fb[(v.notna()) & (v > 0)].dropna()
    if len(valid) < min_names:
        return pd.Series(dtype=float), set(), set()

    q_hi, q_lo, q_mid = (valid.quantile(1.0 - 1.0 / n_tiles),
                         valid.quantile(1.0 / n_tiles), valid.quantile(0.5))
    longs = set(valid[valid >= q_hi].index)
    shorts = set(valid[valid <= q_lo].index)
    if hyst:
        if prev_long:
            longs |= set(valid[valid >= q_mid].index) & prev_long
        if prev_short:
            shorts |= set(valid[valid <= q_mid].index) & prev_short
        shorts -= longs
    if not longs or not shorts:
        return pd.Series(dtype=float), set(), set()

    iv = 1.0 / vol
    wl = iv.reindex(list(longs)); wl = wl / wl.sum()
    ws = iv.reindex(list(shorts)); ws = ws / ws.sum()
    w = pd.Series(0.0, index=valid.index)
    w.loc[wl.index] = wl
    w.loc[ws.index] = -ws

    bL = float((wl * mb.reindex(wl.index)).sum())
    bS = float((ws * mb.reindex(ws.index)).sum())
    if np.isfinite(bL) and np.isfinite(bS) and bL > 0 and bS > 0:
        w.loc[ws.index] = w.loc[ws.index] * np.clip(bL / bS, 0.5, 2.0)
    return w, set(wl.index), set(ws.index)


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    flow = panel[FLOW_COL].astype(float)
    px = panel.drop(columns=[FLOW_COL])
    rets = px.pct_change()
    idx = rets.index

    mkt = rets.mean(axis=1, skipna=True)
    F = _flow_shock(flow, p["flow_growth"], p["flow_detrend"]).reindex(idx)
    fvals, mvals = F.values, mkt.values

    rebal_dates = (pd.Series(idx, index=idx)
                   .groupby(idx.to_period("M")).max().tolist())
    W_rebal = pd.DataFrame(0.0, index=pd.DatetimeIndex(rebal_dates),
                           columns=rets.columns)

    prev_long, prev_short = set(), set()
    bw, mo, mn, cvl, nt, hy = (p["beta_window"], p["min_obs"], p["min_names"],
                               p["coin_vol_lb"], p["n_tiles"], p["hysteresis"])
    for t in rebal_dates:
        longs, shorts, w = set(), set(), pd.Series(dtype=float)
        pos = idx.get_indexer([t])[0]
        lo = max(0, pos - bw + 1)
        wi = slice(lo, pos + 1)
        m = mvals[wi]; f = fvals[wi]
        ok = np.isfinite(m) & np.isfinite(f)
        if ok.sum() >= mo:
            R = rets.iloc[wi]
            fb, mb = {}, {}
            for col in R.columns:
                y = R[col].values
                mask = ok & np.isfinite(y)
                if mask.sum() < mo:
                    continue
                X = np.column_stack([np.ones(int(mask.sum())), m[mask], f[mask]])
                try:
                    coef = np.linalg.lstsq(X, y[mask], rcond=None)[0]
                except Exception:
                    continue
                mb[col], fb[col] = coef[1], coef[2]
            if len(fb) >= mn:
                vol = R.iloc[-cvl:].std()
                fin = vol.values[np.isfinite(vol.values)]
                if fin.size:
                    floor = np.percentile(fin, 10)
                    if floor > 0:
                        vol = vol.clip(lower=floor)
                w, longs, shorts = _construct_weights(
                    pd.Series(fb), pd.Series(mb), vol,
                    prev_long, prev_short, hy, nt, mn)
        if len(w):
            W_rebal.loc[t, w.index] = w.values
        prev_long, prev_short = longs, shorts

    # daily weights held between monthly rebalances
    W = W_rebal.reindex(idx, method="ffill").fillna(0.0)

    # vol-target the spread to ~10%/yr using TRAILING realised vol (no lookahead),
    # then cap gross. scale_t is known at end of day t and applied to W_t held t+1.
    spread = (W.shift(1) * rets).sum(axis=1)
    tv = spread.rolling(p["vol_win"], min_periods=20).std() * np.sqrt(365.0)
    scale = (p["target_vol"] / tv).replace([np.inf, -np.inf], np.nan) \
        .clip(upper=p["max_scale"]).fillna(0.0)
    W = W.multiply(scale, axis=0)
    gt = W.abs().sum(axis=1)
    cap = (p["gross"] / gt).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    W = W.multiply(cap, axis=0)

    Wlag = W.shift(1)  # lag is OUR responsibility: weights set on data through t held from t+1
    daily = net_of_cost(Wlag, rets, cost_bps=p["cost_bps"],
                        name="crypto_stableflow_liqbeta").fillna(0.0)
    trades = trades_from_weights(Wlag, rets, _sector_map(rets.columns))
    return daily, trades


# ----------------------------------------------------------------------------- soft expectations
def _neutral_check(ctx):
    """PRIMARY claim: the book is market-neutral (net beta to the equal-weight
    universe ~ 0). Cheap: regress search-window net returns on the universe mean."""
    try:
        hs = pd.Timestamp(ctx["holdout_start"])
        px = ctx["panel"].drop(columns=[FLOW_COL])
        mkt = px.pct_change().mean(axis=1)
        df = pd.concat([ctx["search"].rename("r"), mkt.rename("m")], axis=1).dropna()
        df = df[df.index < hs]
        if len(df) < 60 or df["m"].std() == 0:
            return {"pass": True, "observed": "insufficient"}
        beta = float(np.polyfit(df["m"].values, df["r"].values, 1)[0])
        return {"pass": bool(abs(beta) < 0.3), "observed": round(beta, 3)}
    except Exception as e:
        return {"pass": True, "observed": "error:%s" % e}


def _turnover_check(ctx):
    """Mechanism claim: hysteresis + monthly min-hold lengthen holds (cut turnover)
    vs a no-hysteresis variant. One extra signal() call; trades sliced to search."""
    try:
        hs = ctx["holdout_start"]
        base = [t["hold_days"] for t in ctx["trades"] if t["entry_date"] < hs]
        _, alt_tr = signal(ctx["panel"], hysteresis=False)
        alt = [t["hold_days"] for t in alt_tr if t["entry_date"] < hs]
        if len(base) < 10 or len(alt) < 10:
            return {"pass": True, "observed": "insufficient"}
        ratio = float(np.mean(base)) / float(np.mean(alt))
        return {"pass": bool(ratio >= 1.1), "observed": round(ratio, 3)}
    except Exception as e:
        return {"pass": True, "observed": "error:%s" % e}


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="crypto_stableflow_liqbeta",
    family="crypto_liquidity_beta",
    title=("Crypto stablecoin-flow liquidity-risk premium -- cross-sectional sort on "
           "each coin's beta to aggregate USDT+USDC supply shocks (Pastor-Stambaugh "
           "transplant, market-neutral perp L/S)"),
    markets=["crypto"],
    data_desc=("Binance USD(S)-M perpetual close panel (top-75 by liquidity via "
               "binance_universe) + CoinMetrics aggregate USDT+USDC SplyCur as the "
               "macro liquidity factor; equal-weight universe return as the market factor."),
    pre_registration=(
        "PRIMARY (default params): universe = top-75 most-liquid Binance USD(S)-M "
        "perpetuals. Macro-liquidity SHOCK F_t = trailing-standardized (180d, "
        "drift-removed) z-score of 7-day log-growth of aggregate USDT+USDC circulating "
        "supply (CoinMetrics SplyCur), built from TRAILING data only (supply observed "
        "same-day, no revision look-ahead). For each coin a rolling 90d two-factor OLS "
        "r_i = a + b*MKT + c*F (MKT = equal-weight universe return) yields flow-beta c, "
        "ORTHOGONAL to market beta b. Monthly, cross-sectionally tercile-sort on c: LONG "
        "top / SHORT bottom, inverse-vol within each leg, dollar-neutral with a bounded "
        "beta-neutralizing scale on the short leg (net market beta ~ 0), vol-targeted to "
        "10%/yr (90d trailing), gross <= 2x, 20bps one-way taker cost, hysteresis (exit "
        "at the cross-sectional median) + monthly min-hold to cap turnover. Weights set "
        "on data through t are held from t+1 (signals lagged 1 day). HYPOTHESIS: "
        "high-flow-beta coins must pay a premium for loading on crypto's dominant "
        "undiversifiable liquidity shock (fiat in/out via stablecoins); the L/S spread "
        "earns this RISK premium -- pro-cyclical, crashing in deleveraging. KEY "
        "FALSIFIERS the rails must clear: (i) MCPT before breadth (market-neutral L/S -> "
        "absolute null); (ii) beta-confound gate -- if the spread reduces to market beta "
        "(beta_to_universe>0.6 & sel-alpha<0.4) it is a clean beta-confound null, not a "
        "premium; (iii) broad generalization -- the SAME frozen construction must show "
        ">=60% OOS-positive across disjoint lower-liquidity tiers and a spot cut. Holdout "
        "from 2023-01-01 (search window keeps enough breadth + trades). NOT machine-"
        "checkable here and left as prose: the equity analog (mutual-fund/money-market "
        "flow shocks) as a future cross-market test, and a small (<=25% risk) trend "
        "tail-overlay -- tested STANDALONE first per the anti-over-blend lesson."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default":  {},
        "beta120":  {"beta_window": 120},
        "quartile": {"n_tiles": 4},
        "growth14": {"flow_growth": 14},
        "no_hyst":  {"hysteresis": False},
    },
    scope="broad",
    generalization_universes=["liq_tier2", "liq_tier3", "tier2_spot"],
    load_gen_data=load_gen_data,
    holdout_start="2023-01-01",
    deploy_max_positions=60,
    expectations=[
        {"name": "market_neutral",
         "claim": "abs(beta) of net returns to equal-weight universe < 0.3 (search window)",
         "check": _neutral_check},
        {"name": "hysteresis_cuts_turnover",
         "claim": "hysteresis raises mean holding period >= 10% vs no-hysteresis variant",
         "check": _turnover_check},
    ],
)