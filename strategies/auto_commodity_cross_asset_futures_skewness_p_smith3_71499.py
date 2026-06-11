"""
fut_skew_premium_v1 — Cross-sectional SKEWNESS premium in futures.

Mechanism (lottery-aversion risk premium, Fernandez-Perez et al.):
participants overpay for positively-skewed "lottery" contracts and shun
negatively-skewed ones. We get PAID to hold negative skew and short
positive skew. Frozen construction (pre-registered, no tuning after the
fact): each month-end, trailing-252d realized skew of daily returns of the
21-market Boreas continuous-futures universe; LONG the most-negative-skew
tercile, SHORT the most-positive-skew tercile; inverse-60d-vol weights
within each leg, legs dollar-balanced (market-neutral -> absolute MCPT
null); whole book scaled to 10% ann. vol using TRAILING vol only, gross
capped at 2x; monthly rebalance; 3bps futures cost on turnover.

Tested STANDALONE per the 2026-06-08 anti-dilution lesson — NO trend blend
in this module. scope='broad': lottery aversion is a universal mechanism,
so a stage-1 pass must generalise to three universes sharing ZERO tickers
with the search universe (extended commodities; international financials;
crypto spot cross-section as the out-of-family clientele check).

NO look-ahead: signal at t uses returns through t; weights are shift(1)
before net_of_cost / trades_from_weights (the lag is taken HERE, in this
module). Vol-target scale at t uses returns through t, applied to the
weight effective for t+1's return.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2000-01-01"

# ---- Search universe: the 21 Boreas cross-asset futures markets ----------
UNIVERSE = {
    # equity index
    "ES=F": "equity_index", "NQ=F": "equity_index",
    "YM=F": "equity_index", "RTY=F": "equity_index",
    # rates
    "ZN=F": "rates", "ZB=F": "rates", "ZF=F": "rates",
    # FX
    "6E=F": "fx", "6B=F": "fx", "6J=F": "fx", "6A=F": "fx", "6C=F": "fx",
    # energy
    "CL=F": "energy", "NG=F": "energy", "HO=F": "energy",
    # metals
    "GC=F": "metals", "SI=F": "metals", "HG=F": "metals",
    # grains
    "ZC=F": "grains", "ZS=F": "grains", "ZW=F": "grains",
}

# ---- Generalization universes: DISJOINT (zero ticker overlap) ------------
GEN_UNIVERSES = {
    # Untouched commodity futures (softs / livestock / products / minor
    # grains / PGMs) — literature says the premium is strongest here.
    "ext_commodities": {
        "SB=F": "softs", "KC=F": "softs", "CC=F": "softs", "CT=F": "softs",
        "OJ=F": "softs",
        "LE=F": "livestock", "HE=F": "livestock", "GF=F": "livestock",
        "ZL=F": "grains_ext", "ZM=F": "grains_ext", "ZO=F": "grains_ext",
        "KE=F": "grains_ext",
        "PL=F": "metals_pgm", "PA=F": "metals_pgm",
        "RB=F": "energy_ext",
    },
    # Untouched financials: CHF/NZD/MXN futures + international equity
    # indices (yfinance indices are the sanctioned free source here).
    "ext_financials": {
        "6S=F": "fx_ext", "6N=F": "fx_ext", "6M=F": "fx_ext",
        "^GDAXI": "intl_equity", "^FTSE": "intl_equity", "^N225": "intl_equity",
        "^HSI": "intl_equity", "^STOXX50E": "intl_equity", "^AXJO": "intl_equity",
        "^GSPTSE": "intl_equity", "^BVSP": "intl_equity", "^MXX": "intl_equity",
        "^KS11": "intl_equity", "^FCHI": "intl_equity",
    },
    # Out-of-family check: crypto cross-section (young market, lottery
    # clientele plausibly strong).
    "crypto_xs": {
        "BTC-USD": "crypto_major", "ETH-USD": "crypto_major",
        "BNB-USD": "crypto_major", "XRP-USD": "crypto_alt",
        "ADA-USD": "crypto_alt", "SOL-USD": "crypto_alt",
        "DOGE-USD": "crypto_alt", "LTC-USD": "crypto_alt",
        "BCH-USD": "crypto_alt", "LINK-USD": "crypto_alt",
        "DOT-USD": "crypto_alt", "AVAX-USD": "crypto_alt",
        "ATOM-USD": "crypto_alt", "XLM-USD": "crypto_alt",
        "TRX-USD": "crypto_alt", "MATIC-USD": "crypto_alt",
    },
}

# one flat ticker -> sector map covering search + gen universes
SECTOR_MAP = dict(UNIVERSE)
for _m in GEN_UNIVERSES.values():
    SECTOR_MAP.update(_m)


def load_data() -> pd.DataFrame:
    """Continuous front-contract close panel for the 21 Boreas markets."""
    return yf_panel(sorted(UNIVERSE), start=START)


def load_gen_data(label) -> pd.DataFrame:
    """Panel for ONE generalization universe (same shape as load_data())."""
    return yf_panel(sorted(GEN_UNIVERSES[label]), start=START)


def signal(panel, skew_lb=252, vol_lb=60, q=1.0 / 3.0,
           target_vol=0.10, cost_bps=3.0, min_names=9):
    """
    Long negative-skew tercile / short positive-skew tercile, inverse-vol
    within legs, legs balanced, trailing-vol-targeted, monthly rebalance.
    """
    panel = panel.sort_index()
    rets = panel.pct_change(fill_method=None)

    # trailing realized skew & vol — both use data through t only
    skew = rets.rolling(skew_lb, min_periods=int(skew_lb * 0.8)).skew()
    vol = rets.rolling(vol_lb, min_periods=int(vol_lb * 0.8)).std()

    # month-end (last trading day of each month) rebalance dates
    me_dates = rets.groupby(rets.index.to_period("M")).tail(1).index

    Wt = pd.DataFrame(np.nan, index=me_dates, columns=rets.columns)
    for d in me_dates:
        s = skew.loc[d].dropna()
        v = vol.loc[d].reindex(s.index)
        valid = s.index[v.notna() & (v > 0)]
        if len(valid) < min_names:
            continue  # cross-section too thin -> stay flat
        s, v = s.loc[valid], v.loc[valid]
        k = max(int(round(len(valid) * q)), 2)
        longs = s.nsmallest(k).index   # most NEGATIVE skew -> long
        shorts = s.nlargest(k).index   # most POSITIVE skew -> short
        iv_l = 1.0 / v.loc[longs]
        iv_s = 1.0 / v.loc[shorts]
        row = pd.Series(0.0, index=rets.columns)
        row.loc[longs] = iv_l / iv_l.sum()    # long leg sums to +1
        row.loc[shorts] = -iv_s / iv_s.sum()  # short leg sums to -1
        Wt.loc[d] = row

    # hold weights between rebalances; raw gross = 2 (1 long + 1 short)
    W_raw = Wt.reindex(rets.index).ffill().fillna(0.0)

    # vol-target scale from TRAILING realized book vol (data through t,
    # applied to the weight that earns t+1's return); cap scale at 1.0 so
    # gross never exceeds 2x; flat until enough history.
    r_unscaled = (W_raw.shift(1) * rets).sum(axis=1)
    ann_vol = r_unscaled.rolling(vol_lb, min_periods=vol_lb).std() * np.sqrt(252)
    scale = (target_vol / ann_vol).clip(upper=1.0).replace(
        [np.inf, -np.inf], 0.0).fillna(0.0)
    W = W_raw.mul(scale, axis=0)

    # THE LAG: weights formed at close t earn returns from t+1
    W_lag = W.shift(1)

    daily = net_of_cost(W_lag, rets, cost_bps=cost_bps,
                        name="fut_skew_premium")
    sector_map = {t: SECTOR_MAP.get(t, "other") for t in rets.columns}
    trades = trades_from_weights(W_lag, rets, sector_map)
    return daily, trades


SPEC = StrategySpec(
    id="fut_skew_premium_v1",
    family="skewness_premium",
    title=("Cross-asset futures skewness premium — long negative-skew / "
           "short positive-skew terciles, market-neutral, 10% vol target"),
    markets=["futures"],
    data_desc=("yfinance continuous front-contract closes for the 21-market "
               "Boreas cross-asset universe (equity index, rates, FX, "
               "commodities), 2000->present; gen universes: disjoint "
               "extended commodities, international financials, crypto "
               "cross-section. $0 / free."),
    pre_registration=(
        "Lottery-aversion premium: holders of negatively-skewed futures are "
        "compensated; positively-skewed 'lottery' contracts are overpriced. "
        "FROZEN: month-end trailing-252d daily-return skew, long bottom "
        "tercile / short top tercile, inverse-60d-vol within legs, legs "
        "dollar-balanced (market-neutral -> absolute MCPT null), trailing-"
        "vol-targeted to 10% ann (gross <= 2x), monthly rebalance, 3bps "
        "futures cost on turnover. Standalone — no trend blend (2026-06-08 "
        "anti-dilution lesson); trend may be considered later only as a "
        "sized tail-overlay if it cuts DD without Sharpe drag. Broad scope: "
        "a stage-1 pass must show OOS-positive holdout returns on >=60% of "
        "three ticker-disjoint universes or the candidate is rejected."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "skew126": {"skew_lb": 126},     # shorter skew window
        "quartile": {"q": 0.25},         # tighter tails
        "vol8": {"target_vol": 0.08},    # more conservative targeting
    },
    scope="broad",
    generalization_universes=["ext_commodities", "ext_financials", "crypto_xs"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=14,  # tercile of 21 -> ~7 long + ~7 short
)