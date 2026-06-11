"""
Cross-sectional annual seasonality premium in futures (Keloharju-Linnainmaa-Nyberg).

FIX vs. failed run: the previous module fetched CFTC COT archives with urllib.request —
a SANDBOX VIOLATION (the harness owns ALL I/O; data must come via sdk.adapters only).
No adapter exposes CFTC positioning, so the hedging-pressure hypothesis is UNTESTABLE on
these rails. Rather than smuggle I/O, this module is honestly RE-REGISTERED as a fresh
hypothesis on the same futures universes, computable purely from sanctioned yf_panel
closes (yfinance is the sanctioned source for futures, not US stocks).

MECHANISM (frozen, pre-registered): annual return seasonality — a market's average return
in the SAME CALENDAR MONTH over prior years predicts its cross-sectional return this month
(Keloharju, Linnainmaa, Nyberg, JF 2016: documented across equities, commodities, country
indices). In commodities the economic driver is real: harvest cycles (grains/softs),
heating/driving demand (energy), seasonal hedging flow. Universal mechanism -> scope
'broad': must generalize to untouched ags/softs; fx and rates/livestock are weaker-prior
same-mechanism checks. This axis (calendar) is orthogonal to every prior failed
price-derived sort here (skew, basis carry, value, momentum).

NO-LOOKAHEAD DESIGN:
  * Month m's seasonal score uses ONLY same-calendar-month returns from PRIOR years
    (shift(1) within month-of-year group before the rolling mean), stamped at month START.
  * All weights additionally shift(1)-lagged before net_of_cost / trades_from_weights.

Costs: harness-standard 8bps on turnover via net_of_cost. Standalone book only — no
reflexive trend blend (2026-06-08 lesson); any tail-overlay is a separate later decision.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2000-01-01"

# ---- universes (search vs. DISJOINT confirmation sets) -----------------------------------
PRIMARY = ["CL", "NG", "HO", "RB", "GC", "SI", "HG", "PL"]  # energy+metals (micro-deployable)
GEN_UNIVERSES = {
    "ags_softs": ["ZC", "ZW", "ZS", "ZL", "SB", "KC", "CT", "CC"],
    "fx": ["6E", "6J", "6B", "6C", "6A", "6S"],
    "rates_livestock": ["ZN", "ZB", "ZF", "ZT", "LE", "HE"],
}
SECTORS = {
    "CL": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
    "ZC": "grains", "ZW": "grains", "ZS": "grains", "ZL": "grains",
    "SB": "softs", "KC": "softs", "CT": "softs", "CC": "softs",
    "6E": "fx", "6J": "fx", "6B": "fx", "6C": "fx", "6A": "fx", "6S": "fx",
    "ZN": "rates", "ZB": "rates", "ZF": "rates", "ZT": "rates",
    "LE": "livestock", "HE": "livestock",
}


# ---- panel builders (adapter-only I/O) ----------------------------------------------------
def _build_panel(tickers):
    """Daily continuous-futures close panel via the sanctioned yf_panel adapter."""
    px = yf_panel([t + "=F" for t in tickers], start=START)
    px.columns = [str(c).replace("=F", "") for c in px.columns]
    keep = [t for t in tickers if t in px.columns]
    px = px[keep].dropna(how="all")
    px.index.name = "date"
    return px


def load_data() -> pd.DataFrame:
    return _build_panel(PRIMARY)


def load_gen_data(label) -> pd.DataFrame:
    return _build_panel(GEN_UNIVERSES[label])


# ---- signal -------------------------------------------------------------------------------
def signal(panel, lb_years=10, min_years=3, tercile=1.0 / 3.0, vol_lb=63,
           target_vol=0.10, max_w=0.60, min_names=5, cost_bps=8.0):
    """
    Seasonal score for month m = mean same-calendar-month return over the prior lb_years
    years (>=min_years required; current month EXCLUDED by construction). Cross-sectional
    tercile sort: LONG seasonally strongest, SHORT seasonally weakest. Weekly Monday
    rebalance (sign changes only monthly; sizing refreshes weekly), inverse-vol sized to a
    10% annualized book target, weights shift(1)-lagged, 8bps on turnover.
    """
    px = panel
    rets = px.pct_change(fill_method=None)

    # monthly returns -> trailing same-calendar-month mean from PRIOR years only
    m_ret = px.resample("ME").last().pct_change(fill_method=None)
    seas = pd.DataFrame(index=m_ret.index, columns=m_ret.columns, dtype=float)
    for c in m_ret.columns:
        s = m_ret[c]
        seas[c] = s.groupby(s.index.month).transform(
            lambda x: x.shift(1).rolling(lb_years, min_periods=min_years).mean())
    # knowable at month START (built from prior-year months only) -> stamp at month start
    seas.index = seas.index.to_period("M").to_timestamp()
    z = seas.reindex(rets.index, method="ffill")
    z = z.where(px.notna())  # only rank markets actually trading that day

    n = z.count(axis=1)
    rk = z.rank(axis=1, method="first")
    frac = rk.sub(1).div((n - 1).clip(lower=1), axis=0)
    sgn = frac.ge(1.0 - tercile).astype(float) - frac.le(tercile).astype(float)
    sgn = sgn.where(z.notna(), 0.0)
    sgn.loc[n < min_names] = 0.0  # minimum cross-sectional breadth

    # ---- weekly (Monday) rebalance, inverse-vol sizing ------------------------------------
    mondays = rets.index[rets.index.dayofweek == 0]
    held = sgn.reindex(mondays).fillna(0.0)
    vol = rets.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252.0)
    vol_w = vol.reindex(mondays)
    n_act = held.abs().sum(axis=1).clip(lower=1.0)
    w = held.div(vol_w).mul(target_vol).div(np.sqrt(n_act), axis=0)
    w = w.replace([np.inf, -np.inf], np.nan).clip(-max_w, max_w)
    w = w.where(held != 0, 0.0).fillna(0.0)

    # hold weekly weights through the week on the daily grid
    W = w.reindex(rets.index).ffill().fillna(0.0)

    W_lag = W.shift(1)  # OUR explicit 1-day execution lag
    daily = net_of_cost(W_lag, rets, cost_bps=cost_bps, name="futures_seasonality_xs")
    trades = trades_from_weights(W_lag, rets, SECTORS)
    return daily.dropna(), trades


# ---- spec ---------------------------------------------------------------------------------
SPEC = StrategySpec(
    id="futures_seasonality_xs_v1",
    family="futures_seasonality",
    title="Annual seasonality premium in futures (same-calendar-month XS sort, KLN 2016)",
    markets=["futures"],
    data_desc=("yfinance continuous-futures closes via yf_panel adapter only (sanctioned "
               "yfinance use: futures, not US stocks). No external I/O in module."),
    pre_registration=(
        "FROZEN: seasonal score for month m = mean same-calendar-month return over prior "
        "10 years (>=3 obs required, current month excluded by construction, stamped at "
        "month start); LONG top tercile / SHORT bottom tercile cross-sectionally; "
        "inverse-vol sizing, 10% vol target, weekly Monday rebalance, 8bps turnover costs, "
        "1-day weight lag. SEARCH universe = energy+metals (8 mkts). DIRECTIONAL "
        "EXPECTATIONS (not tuning knobs): (a) mechanism is universal (KLN 2016, across "
        "asset classes) -> must generalize to untouched ags/softs where harvest seasonality "
        "is strongest; fx and rates/livestock are weaker-prior same-mechanism checks; "
        "(b) effect should NOT be explained by momentum (different horizon structure). "
        "STANDALONE test only — any trend tail-overlay is a separate later decision, never "
        "a reflexive 50/50. Book is ~market-neutral: absolute Sharpe MCPT null applies. "
        "Deployment book = CME micro subset (MCL/MGC/SIL/MHG/MNG) of the same frozen signal."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "quartile_book": {"tercile": 0.25},
        "short_memory": {"lb_years": 5},
        "strict_history": {"min_years": 5},
    },
    scope="broad",
    generalization_universes=list(GEN_UNIVERSES),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
)