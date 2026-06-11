"""
Cross-sectional skewness premium in futures.

Mechanism (FROZEN, pre-registered): lottery-aversion risk premium. Investors overpay
for positive-skew (lottery) payoffs and demand compensation to hold negative-skew
assets — the holder of negative skew is PAID to bear crash risk. We harvest it
cross-sectionally: LONG the most-negative-skew tercile, SHORT the most-positive-skew
tercile, in the same 21-market continuous-futures universe as the validated trend leg.

Spec (no tuning of window/terciles — primary is exactly this):
  - rolling 252d skew of daily returns (min 200 obs)
  - monthly rebalance, cross-sectional tercile sort
  - inverse 63d-vol weights WITHIN each leg; the two legs VOL-MATCHED to each other
    (each leg scaled by the inverse of its trailing realized leg vol, so long and
    short legs contribute equal ex-ante risk — NOT equal notional)
  - whole book scaled to 10% annualized vol target, leverage capped at 2x gross
  - PER-MARKET futures costs (Boreas cost model): each market charged its own
    spread bps on turnover (NG is not ES) PLUS a roll drag = spread crossed once
    per roll, rolls-per-year by contract type (crypto spot: no roll).
  - ALL signals/vols/leverage trailing-only; the final weight matrix is
    shift(1)-lagged exactly once before P&L (the lag is taken HERE, once).

scope='broad': lottery aversion should appear WITHIN asset classes. Generalization
universes are ticker-DISJOINT from the 21-market search book:
  commods_alt    — commodities NOT in the search universe (metals/meats/softs/oilseed products)
  financials_alt — equity index / rates / FX NOT in the search universe
  crypto         — crypto cross-section (youngest, least-arbitraged corner)
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

# ----------------------------------------------------------------------------- universes
# Search universe: the 21-market cross-asset futures book (same coverage as trend leg).
SEARCH = {
    # equity index
    "ES=F": "equity_index", "NQ=F": "equity_index", "YM=F": "equity_index",
    # rates
    "ZN=F": "rates", "ZB=F": "rates", "ZF=F": "rates",
    # metals
    "GC=F": "metals", "SI=F": "metals", "HG=F": "metals",
    # energy
    "CL=F": "energy", "NG=F": "energy", "RB=F": "energy",
    # grains / softs
    "ZC=F": "grains", "ZS=F": "grains", "ZW=F": "grains",
    "KC=F": "softs", "SB=F": "softs",
    # FX
    "6E=F": "fx", "6J=F": "fx", "6B=F": "fx", "6A=F": "fx",
}

# Generalization universes — share NO tickers with SEARCH.
GEN = {
    "commods_alt": {
        "PL=F": "metals", "PA=F": "metals",
        "HO=F": "energy",
        "LE=F": "meats", "HE=F": "meats", "GF=F": "meats",
        "ZM=F": "oilseed", "ZL=F": "oilseed", "ZO=F": "grains",
        "CC=F": "softs", "CT=F": "softs", "OJ=F": "softs",
    },
    "financials_alt": {
        "RTY=F": "equity_index", "NKD=F": "equity_index",
        "^FTSE": "equity_index", "^GDAXI": "equity_index", "^STOXX50E": "equity_index",
        "ZT=F": "rates", "UB=F": "rates",
        "6C=F": "fx", "6S=F": "fx", "6N=F": "fx",
    },
    "crypto": {
        "BTC-USD": "crypto", "ETH-USD": "crypto", "XRP-USD": "crypto",
        "LTC-USD": "crypto", "ADA-USD": "crypto", "BNB-USD": "crypto",
        "DOGE-USD": "crypto", "SOL-USD": "crypto", "DOT-USD": "crypto",
        "LINK-USD": "crypto", "BCH-USD": "crypto", "XLM-USD": "crypto",
    },
}

# One global sector map covering every ticker any panel can contain — the trade
# ledger labeller needs it regardless of which universe signal() is run on.
SECTOR_MAP = dict(SEARCH)
for _m in GEN.values():
    SECTOR_MAP.update(_m)

# ------------------------------------------------------------------- Boreas cost model
# PER-MARKET half-spread-crossing cost in bps of notional, charged on turnover.
# Liquid financials are ~1bp; energies/softs/meats are far wider. Crypto spot ~10bp.
SPREAD_BPS = {
    "ES=F": 0.5, "NQ=F": 1.0, "YM=F": 1.0, "RTY=F": 1.5, "NKD=F": 2.0,
    "^FTSE": 2.0, "^GDAXI": 2.0, "^STOXX50E": 2.0,
    "ZN=F": 0.8, "ZB=F": 1.0, "ZF=F": 0.8, "ZT=F": 0.8, "UB=F": 1.5,
    "GC=F": 1.5, "SI=F": 3.0, "HG=F": 3.0, "PL=F": 5.0, "PA=F": 8.0,
    "CL=F": 2.0, "NG=F": 6.0, "RB=F": 5.0, "HO=F": 4.0,
    "ZC=F": 3.0, "ZS=F": 3.0, "ZW=F": 4.0, "ZO=F": 8.0,
    "ZM=F": 4.0, "ZL=F": 4.0,
    "KC=F": 5.0, "SB=F": 5.0, "CC=F": 6.0, "CT=F": 6.0, "OJ=F": 10.0,
    "LE=F": 6.0, "HE=F": 8.0, "GF=F": 10.0,
    "6E=F": 0.8, "6J=F": 1.0, "6B=F": 1.0, "6A=F": 1.0,
    "6C=F": 1.0, "6S=F": 1.5, "6N=F": 1.5,
}
DEFAULT_SPREAD_BPS = 10.0  # crypto + anything unmapped: conservative

# Rolls per year by sector (a roll crosses the spread once on full position notional).
ROLLS_PER_YEAR = {
    "equity_index": 4, "rates": 4, "fx": 4,            # quarterly cycles
    "metals": 6, "energy": 12, "grains": 5, "softs": 5,
    "meats": 6, "oilseed": 6,
    "crypto": 0,                                        # spot: no roll
}


def _per_market_costs(W_lag: pd.DataFrame) -> pd.Series:
    """Daily cost drag (in return units) from per-market spreads on turnover
    plus per-market roll drag on held gross exposure. Trailing/known-only:
    depends solely on the already-lagged weight matrix."""
    spread = pd.Series(
        {c: SPREAD_BPS.get(c, DEFAULT_SPREAD_BPS) for c in W_lag.columns}
    ) / 1e4
    rolls = pd.Series(
        {c: ROLLS_PER_YEAR.get(SECTOR_MAP.get(c, "crypto"), 6) for c in W_lag.columns}
    )
    turnover = W_lag.diff().abs()
    turnover.iloc[0] = W_lag.iloc[0].abs()
    trade_cost = (turnover * spread).sum(axis=1)
    # roll: cross the spread `rolls` times/year on |position|, amortized daily
    roll_cost = (W_lag.abs() * (spread * rolls / 252.0)).sum(axis=1)
    return trade_cost + roll_cost


START = "2000-01-01"


def load_data() -> pd.DataFrame:
    panel = yf_panel(list(SEARCH.keys()), start=START)
    return panel.ffill(limit=5)


def load_gen_data(label) -> pd.DataFrame:
    panel = yf_panel(list(GEN[label].keys()), start=START)
    return panel.ffill(limit=5)


# ----------------------------------------------------------------------------- signal
def signal(panel, skew_window=252, min_obs=200, vol_lb=63,
           target_vol=0.10, gross_cap=2.0, **_):
    """
    Long bottom-skew tercile / short top-skew tercile, monthly rebalance,
    inverse-vol within legs, legs VOL-MATCHED (scaled by inverse trailing leg vol
    so each contributes equal ex-ante risk), 10% vol target, 2x gross cap,
    per-market spread + roll costs (Boreas model).

    Lookahead discipline: skew/vols/leg-vol-matching/leverage at date t use data
    through t only; the FULL weight matrix is shift(1)-lagged exactly once below.
    """
    rets = panel.pct_change(fill_method=None)

    skew = rets.rolling(skew_window, min_periods=min_obs).skew()
    vol = rets.rolling(vol_lb, min_periods=40).std() * np.sqrt(252)

    # monthly rebalance: first trading day of each month
    month = pd.Series(panel.index.month, index=panel.index)
    is_reb = (month != month.shift(1)).values
    dates = panel.index

    w_cur = pd.Series(0.0, index=panel.columns)
    rows = []
    for i, dt in enumerate(dates):
        if is_reb[i]:
            s = skew.loc[dt].dropna()
            v = vol.loc[dt].reindex(s.index)
            s = s[(v > 0) & v.notna()]
            if len(s) >= 6:
                n = max(2, len(s) // 3)
                longs = s.nsmallest(n).index   # most NEGATIVE skew -> earn the premium
                shorts = s.nlargest(n).index   # lottery names -> short
                iv_l = 1.0 / vol.loc[dt, longs]
                iv_s = 1.0 / vol.loc[dt, shorts]
                wl = iv_l / iv_l.sum()          # each leg internally sums to 1
                ws = iv_s / iv_s.sum()
                # VOL-MATCH the legs: trailing realized vol of each leg's portfolio
                # with TODAY's weights over data through dt only.
                hist = rets.iloc[max(0, i - vol_lb + 1): i + 1]
                vl = (hist[longs] @ wl).std() * np.sqrt(252)
                vs = (hist[shorts] @ ws).std() * np.sqrt(252)
                if np.isfinite(vl) and np.isfinite(vs) and vl > 0 and vs > 0:
                    w = pd.Series(0.0, index=panel.columns)
                    w[longs] = wl / vl          # equal-risk legs, NOT equal-notional
                    w[shorts] = -ws / vs
                    w = w / w.abs().sum()       # normalize gross to 1 pre-leverage
                    w_cur = w
            # else: hold previous book (insufficient cross-section this month)
        rows.append(w_cur)
    W = pd.DataFrame(rows, index=dates, columns=panel.columns)

    # trailing vol-target: realized vol of the (lagged) raw book, scaled to target,
    # capped at gross_cap. Trailing-only -> safe once W is shifted below.
    raw = (W.shift(1) * rets).sum(axis=1)
    port_vol = raw.rolling(vol_lb, min_periods=40).std() * np.sqrt(252)
    lev = (target_vol / port_vol.replace(0.0, np.nan)).clip(upper=gross_cap)
    W_scaled = W.mul(lev, axis=0)

    # THE lag: weights built with data through t are applied to t+1 returns.
    W_lag = W_scaled.shift(1).fillna(0.0)

    # gross P&L via the kit (cost_bps=0), then subtract the PER-MARKET Boreas
    # spread + roll cost model — a flat bps would mis-price NG vs ES.
    gross = net_of_cost(W_lag, rets, cost_bps=0.0, name="futures_xs_skew")
    costs = _per_market_costs(W_lag).reindex(gross.index).fillna(0.0)
    daily = (gross - costs).rename("futures_xs_skew")

    trades = trades_from_weights(W_lag, rets, SECTOR_MAP)
    return daily, trades


# ----------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="futures_xs_skew_premium",
    family="higher_moment",
    title="Cross-sectional skewness premium in futures (long negative-skew / short lottery-skew)",
    markets=["futures"],
    data_desc=("yfinance continuous futures, 21-market cross-asset universe "
               "(equity index, rates, metals, energy, grains, softs, FX); "
               "gen universes: disjoint alt-commodities, alt-financials, crypto spot cross-section"),
    pre_registration=(
        "FROZEN PRIMARY: 252d rolling skew (min 200 obs), monthly rebalance, "
        "long bottom-skew tercile / short top-skew tercile, inverse-63d-vol within "
        "legs, legs VOL-MATCHED (each leg scaled by inverse trailing realized leg "
        "vol for equal ex-ante risk contribution), 10% ann vol target trailing-"
        "estimated, 2x gross cap, PER-MARKET futures spread + roll costs per the "
        "Boreas cost model (market-specific spread bps on turnover + spread crossed "
        "rolls-per-year on held notional; crypto spot rolls=0), weights lagged 1 day. "
        "Hypothesis: lottery-aversion premium — negative-skew assets carry positive "
        "compensation; the book is pro-cyclical (paid in calm, hurt in crashes). "
        "Tested STANDALONE; any trend pairing is a separate sized tail-overlay "
        "decision, NOT a reflexive blend. Broad scope: premium must appear "
        "directionally in disjoint commodity-only, financial-only, and crypto "
        "cross-sections or it is a non-generalizing outlier."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "skew_180d": {"skew_window": 180, "min_obs": 150},
        "skew_378d": {"skew_window": 378, "min_obs": 300},
        "vol_8pct": {"target_vol": 0.08},
    },
    scope="broad",
    generalization_universes=["commods_alt", "financials_alt", "crypto"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=14,  # ~7 per side at tercile width
)