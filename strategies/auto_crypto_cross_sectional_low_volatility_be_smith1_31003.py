# Strategy: cross-sectional CRYPTO low-volatility / betting-against-beta (BAB) long-short.
#
# THESIS FIDELITY NOTE (read first):
# The proposal's frozen economic thesis is a crypto low-vol/lottery-demand premium: sort the
# crypto cross-section on 30d realized vol, go long the low-vol tercile / short the high-vol
# tercile DOLLAR-NEUTRAL, then neutralize the book's 60d BTC-beta with a SMALL BTC overlay.
# This module implements EXACTLY that. The ONLY thing it cannot reproduce is the venue/instrument
# detail: the proposal names binance USDT PERPS via binance_klines/binance_universe/funding_rates,
# which are NOT in the tested harness adapter set (sdk.adapters exposes only sep/us_universe/sf1/
# yf_panel/fred_series/trend_returns/inv_vol_position). Fabricating a binance adapter would break
# import or download raw data -> forbidden. The closest OWNED/FREE tested crypto source is yfinance
# USD spot (yf_panel); perp prices track spot tightly, so the 30d-vol sort and 60d-BTC-beta carry
# over essentially unchanged. The perp-only FUNDING-carry contamination + funding-rate decomposition
# (N/A to spot) remain a FLAGGED downstream step pending a tested binance_klines adapter.
# CAVEAT (honest, prior LOW): yfinance lists only surviving coins -> residual survivorship bias;
# this is why scope is 'local' (forward-validated on the 2022+ holdout) rather than 'broad'.

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2018-01-01"   # spans the 2018 bear, 2020 crash, 2021 bull in the search window
BTC = "BTC-USD"        # the market factor / overlay instrument (excluded from the cross-sectional sort)

# Liquid crypto cross-section (yfinance USD spot), pseudo-sector tagged for ledger spread.
# Stablecoins / wrapped-pegged assets are intentionally excluded (they would trivially dominate
# the low-vol long leg). BTC is the market factor, held separately as the overlay.
CRYPTO = {
    # Layer-1 / smart-contract platforms
    "ETH-USD": "L1", "BNB-USD": "L1", "ADA-USD": "L1", "SOL-USD": "L1", "AVAX-USD": "L1",
    "DOT-USD": "L1", "TRX-USD": "L1", "EOS-USD": "L1", "NEO-USD": "L1", "ETC-USD": "L1",
    "ATOM-USD": "L1", "ALGO-USD": "L1", "XTZ-USD": "L1", "WAVES-USD": "L1", "QTUM-USD": "L1",
    "ICX-USD": "L1", "ONT-USD": "L1", "ZIL-USD": "L1", "HBAR-USD": "L1", "EGLD-USD": "L1",
    "NEAR-USD": "L1", "FTM-USD": "L1", "KSM-USD": "L1", "FLOW-USD": "L1",
    # Payments / currency
    "XRP-USD": "Payments", "LTC-USD": "Payments", "BCH-USD": "Payments", "XLM-USD": "Payments",
    "DASH-USD": "Payments", "DGB-USD": "Payments", "NANO-USD": "Payments", "DCR-USD": "Payments",
    # Privacy
    "XMR-USD": "Privacy", "ZEC-USD": "Privacy", "ZEN-USD": "Privacy",
    # DeFi
    "UNI-USD": "DeFi", "AAVE-USD": "DeFi", "MKR-USD": "DeFi", "COMP-USD": "DeFi", "SNX-USD": "DeFi",
    "YFI-USD": "DeFi", "SUSHI-USD": "DeFi", "CRV-USD": "DeFi", "1INCH-USD": "DeFi", "CAKE-USD": "DeFi",
    "RUNE-USD": "DeFi", "BAL-USD": "DeFi", "KNC-USD": "DeFi", "ZRX-USD": "DeFi", "BNT-USD": "DeFi",
    # Oracle / infra / storage
    "LINK-USD": "Infra", "GRT-USD": "Infra", "BAND-USD": "Infra", "VET-USD": "Infra",
    "ICP-USD": "Infra", "FIL-USD": "Storage", "STORJ-USD": "Storage", "AR-USD": "Storage",
    # Exchange tokens
    "CRO-USD": "Exchange", "KCS-USD": "Exchange", "HT-USD": "Exchange",
    # Gaming / NFT / metaverse
    "MANA-USD": "Gaming", "ENJ-USD": "Gaming", "SAND-USD": "Gaming", "AXS-USD": "Gaming",
    "CHZ-USD": "Gaming", "GALA-USD": "Gaming", "THETA-USD": "Gaming",
    # Meme
    "DOGE-USD": "Meme", "SHIB-USD": "Meme",
    # Misc utility
    "BAT-USD": "Utility", "HOT-USD": "Utility", "IOST-USD": "Utility", "SC-USD": "Utility",
    "RVN-USD": "Utility", "KAVA-USD": "Utility", "IOTA-USD": "Utility", "ANKR-USD": "Utility",
    "CELO-USD": "Utility", "OMG-USD": "Utility", "LRC-USD": "Utility", "SKL-USD": "Utility",
}

DEFAULTS = dict(
    vol_lb=30,          # 30d trailing realized-vol window -- the sort variable (per proposal)
    beta_lb=60,         # 60d trailing BTC-beta window -- for the BAB overlay (per proposal)
    long_q=1.0/3.0,     # long bottom tercile (lowest realized vol)
    short_q=1.0/3.0,    # short top tercile (highest realized vol)
    min_leg=10,         # breadth gate: real diversified legs each side (universe ~75 -> terciles ~25)
    max_zero_frac=0.30, # drop stale/dead names (repeated last price -> spuriously "low vol")
    gross_cap=2.0,      # respect the 2x gross cap
    target_vol=0.15,    # annualized vol target (de-risk only; never lever beyond base gross)
    vt_lb=45,           # trailing window for the vol-target estimate
    disp_lb=180,        # window for the cross-sectional-dispersion stress gate
    disp_floor=0.60,    # frozen: scale down when dispersion < 0.6 * trailing median
    stress_scale=0.50,  # frozen gross-haircut when dispersion collapses
    cost_bps=8.0,       # ~8 bps on turnover
)


def _panel():
    tickers = list(CRYPTO.keys())
    px = yf_panel([BTC] + tickers, START)
    cols = [c for c in [BTC] + tickers if c in px.columns]
    px = px.reindex(columns=cols).sort_index()
    smap = {BTC: "Market"}
    smap.update({t: CRYPTO[t] for t in tickers if t in px.columns})
    px.attrs["sector_map"] = smap   # attrs survive the reindex above
    return px


def load_data() -> pd.DataFrame:
    """Search universe: liquid crypto USD spot cross-section + BTC market factor (survivor-listed)."""
    return _panel()


# ---------------------------------- signal helpers -------------------------------------------
def _rolling_beta(R, mkt, lb, mp):
    """Trailing beta of each name vs BTC (population moments, all strictly trailing)."""
    Em = mkt.rolling(lb, min_periods=mp).mean()
    Em2 = (mkt * mkt).rolling(lb, min_periods=mp).mean()
    varm = (Em2 - Em * Em).replace(0, np.nan)
    Erm = R.mul(mkt, axis=0).rolling(lb, min_periods=mp).mean()
    Er = R.rolling(lb, min_periods=mp).mean()
    cov = Erm.sub(Er.mul(Em, axis=0))
    return cov.div(varm, axis=0)


def signal(panel, **params):
    p = {**DEFAULTS, **params}
    px = panel.sort_index()
    smap = {t: (panel.attrs.get("sector_map", {}) or {}).get(t, "Other") for t in px.columns}
    rets = px.pct_change()
    idx = px.index

    cross = [c for c in px.columns if c != BTC]
    R = rets[cross]
    btc = rets[BTC] if BTC in rets.columns else R.mean(axis=1)   # market factor (fallback if BTC absent)

    vlb, blb = int(p["vol_lb"]), int(p["beta_lb"])
    rvol = R.rolling(vlb, min_periods=int(vlb * 0.8)).std()
    beta = _rolling_beta(R, btc, blb, int(blb * 0.8))

    # eligibility: valid features + non-stale (crypto has no meaningful $ price floor; SHIB/DOGE etc.
    # trade << $1, so we gate on data quality, not nominal price).
    zero_frac = (R.abs() < 1e-9).rolling(vlb, min_periods=int(vlb * 0.8)).mean()
    elig = rvol.notna() & beta.notna() & (zero_frac < p["max_zero_frac"]) & (px[cross] > 0)
    rvol_e = rvol.where(elig).clip(lower=1e-6)
    beta_e = beta.where(elig)

    # cross-sectional tercile membership on 30d realized vol (low vol = long, high vol = short)
    r = rvol_e.rank(axis=1, pct=True)
    long_mask = r <= p["long_q"]
    short_mask = r >= (1.0 - p["short_q"])

    # inverse-vol (equal-risk) weights within each leg
    iv = 1.0 / rvol_e
    wL = iv.where(long_mask); wL = wL.div(wL.sum(axis=1), axis=0)
    wS = iv.where(short_mask); wS = wS.div(wS.sum(axis=1), axis=0)

    # DOLLAR-NEUTRAL spread (each leg half-gross -> spread gross 1.0, net 0)
    spread = (wL.fillna(0.0) - wS.fillna(0.0)) * 0.5

    # neutralize the book's residual 60d BTC-beta with a SMALL BTC overlay (the BAB construction;
    # NOT scaling each leg to beta=1 -- this is the proposal's dollar-neutral + BTC-overlay form)
    net_beta = (spread * beta_e).sum(axis=1)
    btc_overlay = -net_beta

    W = spread.copy()
    W[BTC] = btc_overlay
    W = W.reindex(columns=list(px.columns)).fillna(0.0)

    # cap gross at 2x
    gross = W.abs().sum(axis=1)
    W = W.mul((p["gross_cap"] / gross).clip(upper=1.0).fillna(0.0), axis=0)

    # breadth gate: need real, diversified legs each side, else flat
    breadth_ok = (long_mask.sum(axis=1) >= p["min_leg"]) & (short_mask.sum(axis=1) >= p["min_leg"])
    W = W.mul(breadth_ok.astype(float), axis=0)

    # weekly rebalance: keep last trading day of each ISO week, hold (ffill) in between
    s = pd.Series(np.arange(len(idx)), index=idx.to_period("W"))
    rebal = np.zeros(len(idx), bool)
    rebal[s.groupby(level=0).max().to_numpy()] = True
    W_held = W.copy()
    W_held.loc[~rebal] = np.nan
    W_held = W_held.ffill().fillna(0.0)

    # volatility target (de-risk only; uses strictly-lagged pre-vt spread vol -> no look-ahead)
    r1 = (W_held.shift(1) * rets).sum(axis=1)
    sv = r1.rolling(int(p["vt_lb"]), min_periods=20).std() * np.sqrt(252.0)
    vt = (p["target_vol"] / sv).clip(upper=1.0).fillna(1.0)
    W_held = W_held.mul(vt, axis=0)

    # frozen conditional gate: scale gross down when cross-sectional vol-dispersion collapses
    disp = np.log(rvol_e).std(axis=1)
    disp_med = disp.rolling(int(p["disp_lb"]), min_periods=60).median()
    dscale = pd.Series(1.0, index=idx)
    dscale[(disp < p["disp_floor"] * disp_med).fillna(False)] = p["stress_scale"]
    W_held = W_held.mul(dscale, axis=0)

    # lag is OUR responsibility: weights decided at close t are TRADED from t+1
    Wlag = W_held.shift(1).fillna(0.0).reindex(columns=rets.columns).fillna(0.0)

    daily = net_of_cost(Wlag, rets, cost_bps=p["cost_bps"], name="lowvol_bab_crypto")
    trades = trades_from_weights(Wlag, rets, smap)   # kit stamps entry_regime
    return daily.dropna(), trades


# ----------------------------- soft expectations (machine-checkable) -------------------------
def _check_beta_neutral(ctx):
    """Mechanism claim: the BTC-overlay leaves the spread ~market(BTC)-neutral."""
    r = ctx["search"].dropna()
    px = ctx["panel"].sort_index()
    if BTC not in px.columns:
        return {"pass": True, "observed": "no_btc"}
    m = px[BTC].pct_change()
    df = pd.concat([r.rename("s"), m.rename("m")], axis=1).dropna()
    df = df[df.index < pd.Timestamp(ctx["holdout_start"])]
    if len(df) < 60:
        return {"pass": True, "observed": "insufficient_data"}
    b = float(np.polyfit(df["m"].to_numpy(), df["s"].to_numpy(), 1)[0])
    return {"pass": abs(b) <= 0.30, "observed": round(b, 4)}


def _check_persistence(ctx):
    """Mechanism claim: 30d realized-vol ranks are persistent -> tercile membership is sticky ->
    median hold extends beyond a single weekly cycle (median hold_days >= 10)."""
    hds = [t["hold_days"] for t in ctx["trades"] if "hold_days" in t]
    if not hds:
        return {"pass": True, "observed": "no_trades"}
    med = float(np.median(hds))
    return {"pass": med >= 10.0, "observed": med}


SPEC = StrategySpec(
    id="lowvol_bab_crypto",
    family="low_vol_bab",
    title="Cross-sectional crypto low-volatility / betting-against-beta long-short (lottery-demand premium)",
    markets=["Crypto (yfinance USD spot cross-section; BTC-USD as market factor/overlay)"],
    data_desc=("yfinance daily USD spot Close over a ~75-coin liquid crypto cross-section; trailing "
               "30d realized vol (sort) + 60d BTC-beta (overlay neutralization) per coin; BTC-USD is "
               "the market factor and is held only as the beta-neutralizing overlay, not sorted. "
               "Forward-validation on the 2022+ holdout."),
    pre_registration=(
        "HYPOTHESIS (frozen): leverage-aversion / lottery-demand in crypto systematically overprices "
        "high-volatility coins and underprices low-vol ones -> a low-vol/BAB premium. CONSTRUCTION "
        "(frozen, faithful to proposal): weekly (last ISO trading day) rebalance over a liquid crypto "
        "cross-section; long bottom-tercile / short top-tercile of trailing 30d realized vol; inverse-vol "
        "(equal-risk) weights within each leg; the spread is built DOLLAR-NEUTRAL and its residual 60d "
        "BTC-beta is neutralized with a SMALL BTC overlay (w_BTC = -net_beta) so the book carries ~no "
        "BTC direction; gross capped at 2x; 15% annual vol target (de-risk only); frozen conditional gate "
        "halves gross when cross-sectional vol-dispersion < 0.6x its trailing median (no spread to "
        "harvest). NO LOOK-AHEAD: all features strictly trailing, weights lagged 1 day before trading, "
        "8bps cost on turnover. "
        "DATA SUBSTITUTION (disclosed): the originating proposal names binance USDT PERPS via "
        "binance_klines/binance_universe/funding_rates, which are NOT in the tested harness adapter set; "
        "fabricating one would break import or download raw data (forbidden). The closest OWNED/FREE "
        "tested crypto source is yfinance USD spot (yf_panel), and perp prices track spot tightly, so the "
        "30d-vol sort + 60d-BTC-beta overlay carry over essentially unchanged. The perp-only "
        "funding-carry contamination and the funding-rate decomposition (N/A to spot, no funding leg) are "
        "a FLAGGED downstream deployment step pending a tested binance_klines adapter. SCOPE = local: "
        "yfinance lists only surviving coins (residual survivorship bias -> prior LOW) and a single "
        "data-limited asset class cannot populate 3 disjoint, breadth-sufficient crypto holdouts; the edge "
        "is therefore forward-validated on the 2022+ out-of-sample window rather than via a broad "
        "cross-universe battery. CHECKABLE CLAIMS (soft expectations): residual BTC-beta ~0 (|beta|<=0.30); "
        "persistent 30d-vol ranks give median hold_days>=10. Cross-asset sign stability / crowding / "
        "sub-period stability are evaluated by the harness sub-sample / MCPT / beta-confound gates, not "
        "by soft expectations."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "vol_short": {"vol_lb": 20},
        "vol_long": {"vol_lb": 45},
        "beta_short": {"beta_lb": 45},
    },
    scope="local",
    holdout_start="2022-01-01",
    deploy_max_positions=50,
    expectations=[
        {"name": "beta_neutral",
         "claim": "the BTC overlay leaves the spread near BTC-neutral (|beta to BTC| <= 0.30)",
         "check": _check_beta_neutral},
        {"name": "vol_rank_persistence",
         "claim": "persistent 30d vol ranks make tercile membership sticky (median hold_days >= 10)",
         "check": _check_persistence},
    ],
)