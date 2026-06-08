"""Agent-proposed strategy: Commodity(energy)-carry + Trend two-premium book.

Carry leg  = ENERGY term-structure carry (roll-yield / backwardation risk premium, pro-cyclical).
             Per commodity, carry_i = spot_i / front_future_i - 1  (>0 = backwardation = positive roll
             yield). Each rebalance, rank the energy complex cross-sectionally, go long the backwardated
             (above-median) names / short the contango (below-median) names, inverse-vol sized, monthly,
             holding the FRONT continuous future. SAME frozen 8bps micro-cost model as Boreas.
Trend leg  = the FROZEN validated Boreas 21-market TSMOM (1/3/12m sign blend, inverse-vol, weekly) —
             crisis-alpha hedge. Vol-matched 50/50 against the carry leg.

GATE-0 (honest scoping). A free, long-history, daily front-vs-deferred term structure is reliably
obtainable through the sanctioned adapters ONLY for the ENERGY complex: yfinance front futures
(CL/BZ/NG/HO=F, 2005/2007+) paired with FRED hub SPOT (DCOILWTICO, DCOILBRENTEU, DHHNGSP, DDFUELUSGULF
diesel/No.2 ~ heating-oil proxy) to form the short-end basis. Metals/ags deferred curves are NOT free
through yf_panel/fred_series, so per the proposal's own contingency the carry leg is scoped to an
energy sleeve; the combined book's cross-sector diversification + deployment-sanity sector spread is
carried by the 21-market trend leg (Equity/Rates/Commod/FX). This DOES de-risk the validated book's
flagged weakness (crypto regime-concentration) by replacing the crypto carry leg with a non-crypto,
physically-driven (storage / convenience-yield) carry premium.

Pre-registered PRIMARY metric = the combined book's MAR / max-drawdown and leg correlation on the
write-once holdout via the standard CPCV/DSR/PBO/FDR rails. The standalone legs are diagnostics only.
The harness owns ALL rails; this module only produces (daily_returns, trades). FROZEN.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, trend_returns, inv_vol_position

COST = 8.0 / 1e4  # 8bps per unit turnover (same frozen micro-cost as Boreas)

# Energy sleeve: name -> (yfinance front future, FRED daily hub spot). Units align per name
# (WTI/Brent $/bbl, NatGas $/MMBtu, HeatingOil $/gal) so the basis ratio is dimensionless + comparable.
FUT_TICKERS = {"CL=F": "WTI", "BZ=F": "Brent", "NG=F": "NatGas", "HO=F": "HeatingOil"}
SPOT_IDS = {"DCOILWTICO": "WTI", "DCOILBRENTEU": "Brent", "DHHNGSP": "NatGas", "DDFUELUSGULF": "HeatingOil"}
START = "2005-01-01"


def load_data() -> pd.DataFrame:
    """Panel = energy FRONT futures (return + curve front point) + FRED hub SPOT (curve near point).
    Columns prefixed FUT_/SPOT_; signal() splits them. Trend leg is loaded internally (frozen)."""
    fut = yf_panel(list(FUT_TICKERS), start=START).rename(columns=FUT_TICKERS)
    spot = fred_series(SPOT_IDS, start=START)  # columns already named WTI/Brent/NatGas/HeatingOil
    fut.columns = [f"FUT_{c}" for c in fut.columns]
    spot.columns = [f"SPOT_{c}" for c in spot.columns]
    df = pd.concat([fut, spot], axis=1).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df.ffill(limit=3)


def _vol_scale(r, tgt=0.10, ann=252):
    v = float(pd.Series(r).std() * np.sqrt(ann))
    return r * (tgt / v) if v > 0 else r


def _sign_run_trades(pos: pd.DataFrame, rets: pd.DataFrame, sector: str, lo) -> list:
    """One trade per held-position run per name (for deployment-sanity), within the overlap window."""
    trades = []
    for nm in pos.columns:
        s = np.sign(pos[nm]).fillna(0.0)
        cur, ent = 0.0, None
        for dt, sg in s.items():
            if sg != cur:
                if cur != 0.0 and ent is not None and dt >= lo:
                    seg = pos[nm].loc[ent:dt]
                    rseg = rets[nm].loc[ent:dt].fillna(0.0)
                    trades.append({"ticker": f"{nm}-carry", "sector": sector,
                                   "entry_date": str(ent.date()), "exit_date": str(dt.date()),
                                   "hold_days": int(len(seg)),
                                   "position_value": float(abs(seg).mean() if len(seg) else 0.0),
                                   "pnl": float((seg.fillna(0) * rseg).sum())})
                cur, ent = sg, dt
    return trades


def signal(panel, blend=0.5, carry_vol=0.10, max_pos=2.0, vol_lb=60, carry_rebalance="ME"):
    """(daily_returns, trades) for the vol-matched commodity-carry + trend book. Causal: 1-day lag."""
    fut = panel[[c for c in panel.columns if c.startswith("FUT_")]].rename(columns=lambda c: c[4:])
    spot = panel[[c for c in panel.columns if c.startswith("SPOT_")]].rename(columns=lambda c: c[5:])
    names = [n for n in FUT_TICKERS.values() if n in fut.columns and n in spot.columns]
    fut, spot = fut[names], spot[names]
    rets = fut.pct_change()

    # --- ENERGY CARRY signal: short-end basis (backwardation = positive carry), cross-sectional L/S ---
    basis = (spot / fut) - 1.0                       # >0 backwardated (earn roll), <0 contango
    med = basis.median(axis=1)
    sig = np.sign(basis.sub(med, axis=0))            # +1 above-median carry / -1 below (terciles->halves at N=4)
    sig = sig.where(basis.notna(), 0.0).fillna(0.0)
    # inverse-vol sizing + monthly rebalance + 1d lag (no look-ahead) via the shared building block
    pos = inv_vol_position(sig, rets, target_vol=carry_vol, vol_lb=vol_lb,
                           max_pos=max_pos, rebalance=carry_rebalance)
    gross = (pos * rets).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(0.0)
    carry = (gross - turn * COST).dropna()
    carry.index = pd.to_datetime(carry.index).normalize()
    carry.name = "carry"

    # --- TREND leg (frozen Boreas 21-market TSMOM) ---
    trend, ttrades = trend_returns()
    trend = pd.Series(trend).copy(); trend.index = pd.to_datetime(trend.index).normalize()
    trend.name = "trend"

    # --- align overlap, vol-match each leg, blend 50/50 ---
    df = pd.concat([carry, trend], axis=1).dropna()
    combo = blend * _vol_scale(df["carry"]) + (1 - blend) * _vol_scale(df["trend"])
    combo = combo.dropna(); combo.name = "commod_carry_trend"

    # --- trades = trend sign-runs (4 sectors) + carry sign-runs (Energy), within the overlap ---
    lo = df.index.min()
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    trades += _sign_run_trades(pos.loc[pos.index >= lo], rets, "Energy", lo)
    return combo, trades


SPEC = StrategySpec(
    id="commod-carry-trend-book",
    family="commod_carry_trend_combo",
    title="Commodity(energy)-carry + Trend two-premium book (non-crypto carry leg)",
    markets=["futures"],
    data_desc=("FREE: energy front futures CL/BZ/NG/HO=F (yfinance, 2005/2007+) + FRED hub spot "
               "DCOILWTICO/DCOILBRENTEU/DHHNGSP/DDFUELUSGULF for the short-end basis; Boreas 21-market "
               "trend (yfinance). Gate-0: only energy term structure is free via the sanctioned adapters "
               "-> carry scoped to an energy sleeve; trend leg supplies cross-sector breadth."),
    pre_registration=(
        "Carry leg = ENERGY term-structure carry. carry_i = spot_i/front_i - 1 (>0 backwardation = "
        "positive roll yield). DATA LIMITATION (frozen, documented BEFORE the verdict): free sources "
        "yield a front-vs-physical-hub-SPOT basis (a short-end curve PROXY), not the front-vs-deferred "
        "slope; WTI hub spot ~ the front contract so WTI carries little signal, and other hubs carry a "
        "structural offset -- this is the honest free-data ceiling, not a tuned choice. "
        "Weekly/monthly cross-sectional rank of {WTI,Brent,NatGas,HeatingOil}; "
        "long above-median carry / short below (tercile->half at N=4), inverse-vol to "
        "carry_vol target, MONTHLY rebalance, hold the front continuous future, 8bps on turnover, "
        "signals lagged 1 day (no look-ahead). HeatingOil uses Gulf No.2 diesel spot as the basis proxy. "
        "Trend leg = FROZEN Boreas 21-market TSMOM. Vol-matched 50/50. FROZEN. PRIMARY pre-registered "
        "metric = combined book MAR / max-drawdown + leg correlation on the write-once 2022+ holdout via "
        "CPCV/DSR/PBO/FDR; standalone legs are diagnostics only. HYPOTHESIS: a non-crypto (storage / "
        "convenience-yield) carry premium with corr(carry,trend)<~0 lets the crisis-alpha trend leg cut "
        "the carry drawdown at little Sharpe cost, de-risking the book's crypto regime-concentration."),
    load_data=load_data, signal=signal,
    default_params={},
    grid={
        "default": {},
        "blend_carry": {"blend": 0.65},
        "blend_trend": {"blend": 0.35},
        "carryvol_low": {"carry_vol": 0.07},
        "reb_weekly": {"carry_rebalance": "W-FRI"},
        "vollb_120": {"vol_lb": 120},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=25,  # 4 energy carry names + 21 trend markets
)
