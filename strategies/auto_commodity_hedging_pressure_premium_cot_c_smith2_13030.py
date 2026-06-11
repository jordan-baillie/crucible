"""
Commodity hedging-pressure premium — traded via its PRICE FOOTPRINT (positioning proxy), L/S futures.

MECHANISM (Keynes normal backwardation / Basu-Miffre hedging pressure): speculators are paid an
insurance premium for absorbing commercial hedgers' net positions. Commercials are systematically
CONTRARIAN and speculators trend-following (Kang-Rouwenhorst-Tang; Moskowitz-Ooi-Pedersen): when a
market has rallied over the past year, producers hedge harder -> commercials most net SHORT and
speculators net long. The hedging-pressure premium accrues to whoever holds the commercials'
opposite side, i.e. LONG markets where commercials are most net short.

SANDBOX FIX (this revision): the previous version downloaded CFTC COT archives via
urllib.request — a SANDBOX VIOLATION (the harness owns ALL I/O; data must come from sdk.adapters
only). ALL download code (urllib/zipfile/io/time) is REMOVED. Instead we trade the documented
PRICE FOOTPRINT of hedging pressure: trailing 12-month return is a strong positive proxy for
commercial net-short positioning (the literature above reports correlations >0.6 between
commercial net-short and trailing returns). LONG the highest trailing-return tercile
(commercials most net short -> we provide the insurance) and SHORT the lowest tercile
(commercials most net long). All data via yf_panel (FREE continuous futures) — no other I/O.

POINT-IN-TIME DISCIPLINE: the proxy uses only past closes (fully observable same-day); weights
are shifted ONE day before costing/ledger (the standard execution lag — applied explicitly below).

COSTS: 5bps per side on all rebalance turnover via net_of_cost, PLUS an explicit amortized ROLL
cost: each held contract rolls on a sector-specific calendar (energy ~monthly, metals/livestock
~bi-monthly, grains/softs/fx/rates ~quarterly); a roll is 2 sides of turnover (close old + open
new) at the same bps rate, charged daily as gross_exposure * 2 / roll_interval_days * cost_bps.

FROZEN SPEC: NO grid search over lookbacks or quantiles — the 252d lookback and tercile cut are
FIXED. The only grid entry besides "default" is a pure cost-stress (doubled bps), which searches
nothing.

SCOPE='broad': the mechanism is universal wherever commercials hedge. Generalization universes
are fully DISJOINT from the search universe (share no tickers):
  - softs_livestock : CT, SB, KC, CC, LE, HE   (commodity sectors held out of search)
  - fx              : 6E, 6J, 6B, 6A, 6C, 6S   (financial futures — same hedger/speculator split)
  - rates           : ZT, ZF, ZN, ZB           (Treasury futures)

FROZEN PRIMARY: weekly tercile L/S on the 12-month-return positioning proxy, inverse-vol within
legs, 10% vol target, weekly (Friday) rebalance.
STANDALONE test — no trend blend (per the credit-carry dilution lesson).
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

PX_START = "1999-06-01"

# ---------------------------------------------------------------------------
# Universes: yfinance continuous ticker -> sector
# ---------------------------------------------------------------------------
SEARCH_UNIVERSE = {
    "CL=F": "energy",   # WTI crude
    "HO=F": "energy",   # heating oil
    "RB=F": "energy",   # RBOB gasoline
    "NG=F": "energy",   # natural gas
    "GC=F": "metals",   # gold
    "SI=F": "metals",   # silver
    "HG=F": "metals",   # copper
    "PL=F": "metals",   # platinum
    "ZW=F": "grains",   # CBOT wheat
    "ZC=F": "grains",   # corn
    "ZS=F": "grains",   # soybeans
    "ZL=F": "grains",   # soybean oil
    "ZM=F": "grains",   # soybean meal
}

GEN_UNIVERSES = {
    "softs_livestock": {
        "CT=F": "softs",      # cotton
        "SB=F": "softs",      # sugar 11
        "KC=F": "softs",      # coffee C
        "CC=F": "softs",      # cocoa
        "LE=F": "livestock",  # live cattle
        "HE=F": "livestock",  # lean hogs
    },
    "fx": {
        "6E=F": "fx",  # euro
        "6J=F": "fx",  # yen
        "6B=F": "fx",  # pound
        "6A=F": "fx",  # aussie
        "6C=F": "fx",  # cad
        "6S=F": "fx",  # chf
    },
    "rates": {
        "ZT=F": "rates",  # 2y note
        "ZF=F": "rates",  # 5y note
        "ZN=F": "rates",  # 10y note
        "ZB=F": "rates",  # 30y bond
    },
}

# global ticker -> sector map (signal() only sees the panel, so look sectors up here)
SECTORS = dict(SEARCH_UNIVERSE)
for _u in GEN_UNIVERSES.values():
    SECTORS.update(_u)

# sector -> approximate roll interval in trading days (close old + open new = 2 sides per roll)
ROLL_DAYS = {
    "energy": 21,      # monthly contract cycle
    "metals": 42,      # ~bi-monthly active months
    "grains": 63,      # ~quarterly active months
    "softs": 63,
    "livestock": 42,
    "fx": 63,          # quarterly IMM cycle
    "rates": 63,       # quarterly cycle
    "other": 63,
}


# ---------------------------------------------------------------------------
# Data (adapters ONLY — the harness owns all I/O)
# ---------------------------------------------------------------------------
def _build_panel(universe):
    tickers = list(universe)
    px = yf_panel(tickers, start=PX_START)
    return px.reindex(columns=tickers)


def load_data():
    return _build_panel(SEARCH_UNIVERSE)


def load_gen_data(label):
    return _build_panel(GEN_UNIVERSES[label])


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def signal(panel, mom_lb=252, long_q=2.0 / 3.0, short_q=1.0 / 3.0,
           target_vol=0.10, vol_lb=63, cost_bps=5.0, max_lev=3.0, min_names=4):
    px = panel
    rets = px.pct_change(fill_method=None)
    rets_f = rets.fillna(0.0)

    # positioning proxy: trailing 12m return ~ commercial net-SHORT pressure
    hp_proxy = px.pct_change(mom_lb, fill_method=None)

    # tradeable = proxy defined AND enough price history for vol sizing
    vol = rets.rolling(vol_lb).std() * np.sqrt(252)
    valid = hp_proxy.notna() & vol.notna() & (vol > 0)
    enough = valid.sum(axis=1) >= min_names

    ranks = hp_proxy.where(valid).rank(axis=1, pct=True)
    long_m = (ranks >= long_q) & valid   # commercials most net SHORT -> we provide insurance, LONG
    short_m = (ranks <= short_q) & valid  # commercials most net LONG -> SHORT

    iv = (1.0 / vol).where(valid)
    wl = (iv * long_m).div((iv * long_m).sum(axis=1), axis=0).fillna(0.0) * 0.5
    ws = (iv * short_m).div((iv * short_m).sum(axis=1), axis=0).fillna(0.0) * 0.5
    W_raw = (wl - ws).where(enough, 0.0)

    # weekly rebalance: snapshot Fridays, hold through the week
    fridays = rets.index[rets.index.weekday == 4]
    W_raw = W_raw.reindex(fridays).reindex(rets.index, method="ffill").fillna(0.0)

    # vol-target the book using trailing realized vol of the (already-lagged) unscaled book
    book = (W_raw.shift(1) * rets_f).sum(axis=1)
    realized = book.rolling(vol_lb).std() * np.sqrt(252)
    lev = (target_vol / realized.replace(0, np.nan)).clip(upper=max_lev)
    lev = lev.reindex(fridays).reindex(rets.index, method="ffill").fillna(0.0)
    W = W_raw.mul(lev, axis=0)

    # 1-day execution lag is applied HERE before costing/ledger (our responsibility per contract)
    Wl = W.shift(1).fillna(0.0)

    # rebalance turnover cost: cost_bps per side via net_of_cost
    daily = net_of_cost(Wl, rets_f, cost_bps=cost_bps, name="hp_footprint_ls")

    # explicit ROLL cost: each held name rolls on its sector calendar; a roll is 2 sides of
    # turnover at cost_bps, amortized daily over the roll interval
    roll_int = pd.Series({t: float(ROLL_DAYS.get(SECTORS.get(t, "other"), 63))
                          for t in px.columns})
    roll_turnover_daily = Wl.abs().mul(2.0 / roll_int, axis=1).sum(axis=1)
    roll_drag = roll_turnover_daily * (cost_bps / 1e4)
    daily = (daily - roll_drag.reindex(daily.index).fillna(0.0)).rename("hp_footprint_ls")

    sector_map = {t: SECTORS.get(t, "other") for t in px.columns}
    trades = trades_from_weights(Wl, rets_f, sector_map)

    # trim the leading dead zone before proxy/vol history exists
    live = Wl.abs().sum(axis=1) > 0
    if live.any():
        daily = daily.loc[live.idxmax():]
    return daily.dropna(), trades


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
SPEC = StrategySpec(
    id="hedging_pressure_footprint_ls_v2",
    family="futures_positioning_carry",
    title="Hedging-pressure premium via positioning footprint (12m-return proxy, tercile L/S)",
    markets=["futures"],
    data_desc=("yfinance continuous-contract closes only (sdk.adapters.yf_panel; no external "
               "I/O). Positioning proxy = trailing 252d return, the documented price footprint "
               "of commercial net-short hedging pressure (commercials contrarian, speculators "
               "trend-following). 13-market commodity search universe (energy/metals/grains)."),
    pre_registration=(
        "FROZEN PRIMARY: weekly cross-sectional tercile L/S on the hedging-pressure footprint "
        "proxy = trailing 252d return. LONG highest tercile (commercials most net short — we "
        "provide insurance), SHORT lowest tercile (commercials most net long). Inverse-vol "
        "within legs (0.5 gross per side), 10% ann vol target (cap 3x), Friday rebalance. "
        "COSTS: 5bps per side on all rebalance turnover PLUS explicit roll cost (2 sides per "
        "roll at 5bps, sector roll calendar: energy 21d, metals/livestock 42d, "
        "grains/softs/fx/rates 63d, amortized daily on gross exposure). Signal uses only past "
        "closes; weights shifted +1d for execution. NO grid search over lookbacks or quantiles "
        "— the 252d lookback and tercile cut are FIXED; the only non-default grid entry is a "
        "pure cost-stress (doubled bps). STANDALONE test — no trend blend (credit-carry "
        "dilution lesson); trend overlay only as a future, separately registered variant if "
        "standalone passes. Broad-scope falsification: same frozen signal must be OOS-positive "
        "on >=60% of {softs_livestock, fx, rates} — all ticker-disjoint from the search "
        "universe. MCPT absolute null applies (skew-FAIL lesson)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "cost_stress": {"cost_bps": 10.0},
    },
    scope="broad",
    generalization_universes=["softs_livestock", "fx", "rates"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
)