"""Agent-proposed strategy: Commodity term-structure carry + Trend two-premium book.

PREMIUM. A COMBINATION of two complementary premia (the one validated STRUCTURE, new carry leg):
  CARRY leg = commodity term-structure / hedging-pressure premium (Keynesian normal backwardation /
              convenience yield). You are paid to provide price insurance to producers/hedgers: a
              long position in backwardated curves earns a positive roll yield as the future
              converges UP to spot. Cross-sectional: long the most-backwardated commodities, short
              the most-contango'd. NON-crypto -> de-risks the book's crypto regime-concentration and
              does NOT depend on the 2026-08-28 crypto-carry forward verdict.
  TREND leg = the FROZEN, validated Boreas 21-market cross-asset TSMOM (1/3/12m sign blend, inverse
              vol, weekly) -- the crisis-alpha hedge. corr(carry, trend) ~ 0; trend pays the big
              directional moves that crush a short-insurance carry book.

GATE-0 (honest free-data scoping, run BEFORE the build via the two sanctioned adapters):
  Daily, long-history, front-vs-near term structure is FREE only for the ENERGY complex -- yfinance
  front continuous futures paired with FRED daily physical-hub spot:
      WTI        CL=F  / DCOILWTICO    (Cushing  -- SAME hub, clean basis)
      Brent      BZ=F  / DCOILBRENTEU  (Europe   -- SAME hub, clean basis)
      NatGas     NG=F  / DHHNGSP       (HenryHub -- SAME hub, clean basis)
      HeatingOil HO=F  / DDFUELUSGULF  (ULSD/diesel proxy, Gulf hub -- documented location proxy)
      Gasoline   RB=F  / DGASUSGULF    (RBOB/Gulf proxy            -- documented location proxy)
  FRED publishes NO clean daily metals/ags spot (gold/silver IDs 404; copper is monthly), and the
  sanctioned adapters expose no free deferred-contract / CFTC-COT feed -> the carry leg is SCOPED to
  this 5-name ENERGY sleeve (>=5 markets, full 2005/2006+ history, >=2 stress episodes 2008/2020/2022).
  Cross-sector breadth + deployment-sanity sector spread are supplied by the 21-market TREND leg.
  This is the honest free-data CEILING (disclosed before the verdict), not a tuned choice.

The harness owns ALL rails (split / CPCV / DSR / PBO / FDR / write-once holdout / deployment-sanity /
verdict / wiki / alert). This module only produces (daily_returns, trades). FROZEN.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel, fred_series, trend_returns, inv_vol_position

COST = 8.0 / 1e4  # 8bps per unit turnover (same frozen micro-cost as Boreas)

# Energy sleeve: name -> (yfinance front future, FRED daily physical-hub spot). Basis is a ratio so the
# WTI/Brent/NatGas pairs are clean same-hub roll yields; HeatingOil/Gasoline are documented hub proxies.
FUT_TICKERS = {"CL=F": "WTI", "BZ=F": "Brent", "NG=F": "NatGas", "HO=F": "HeatingOil", "RB=F": "Gasoline"}
SPOT_IDS = {"DCOILWTICO": "WTI", "DCOILBRENTEU": "Brent", "DHHNGSP": "NatGas",
            "DDFUELUSGULF": "HeatingOil", "DGASUSGULF": "Gasoline"}
START = "2005-01-01"


def load_data() -> pd.DataFrame:
    """Panel = energy FRONT futures (FUT_*) + FRED hub SPOT (SPOT_*); signal() splits by prefix.
    Trend leg is loaded internally (frozen). No side effects beyond FREE adapter reads."""
    fut = yf_panel(list(FUT_TICKERS), start=START).rename(columns=FUT_TICKERS)
    spot = fred_series(SPOT_IDS, start=START)  # columns already WTI/Brent/NatGas/HeatingOil/Gasoline
    fut.columns = [f"FUT_{c}" for c in fut.columns]
    spot.columns = [f"SPOT_{c}" for c in spot.columns]
    df = pd.concat([fut, spot], axis=1).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df.ffill(limit=3)


def _vol_scale(r, tgt=0.10, ann=252):
    v = float(pd.Series(r).std() * np.sqrt(ann))
    return r * (tgt / v) if v > 0 else r


def _tercile_ls(roll: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional carry portfolio: +1 to the most-backwardated third, -1 to the most-contango'd
    third (k = round(n/3) per side), 0 to the middle. Dollar-neutral by construction (equal counts)."""
    valid = roll.notna()
    n = valid.sum(axis=1)
    k = np.maximum(1, (n / 3.0).round()).astype(int)          # names per side
    rk_hi = roll.rank(axis=1, ascending=False)                # 1 = highest carry (most backwardated)
    rk_lo = roll.rank(axis=1, ascending=True)                 # 1 = lowest carry (most contango)
    sig = pd.DataFrame(0.0, index=roll.index, columns=roll.columns)
    sig = sig.mask(rk_hi.le(k, axis=0), 1.0).mask(rk_lo.le(k, axis=0), -1.0)
    sig = sig.where(valid, 0.0)
    sig = sig.where(n >= 4, 0.0)                              # need a full cross-section to rank
    return sig.fillna(0.0)


def _sign_run_trades(pos: pd.DataFrame, rets: pd.DataFrame, sector: str, lo) -> list:
    """One trade per held-position run per name (deployment-sanity), inside the overlap window."""
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


def signal(panel, blend=0.5, carry_vol=0.10, max_pos=2.0, vol_lb=60, carry_rebalance="W-FRI"):
    """(daily_returns, trades) for the vol-matched commodity-term-carry + trend book. Causal: 1d lag."""
    fut = panel[[c for c in panel.columns if c.startswith("FUT_")]].rename(columns=lambda c: c[4:])
    spot = panel[[c for c in panel.columns if c.startswith("SPOT_")]].rename(columns=lambda c: c[5:])
    names = [n for n in FUT_TICKERS.values() if n in fut.columns and n in spot.columns]
    fut, spot = fut[names], spot[names]
    rets = fut.pct_change()

    # --- CARRY signal: annualized roll yield proxy = spot/front - 1 (>0 backwardation = earn roll).
    #     Cross-sectional rank is invariant to a common annualization constant, so we rank the basis. ---
    roll = (spot / fut) - 1.0
    sig = _tercile_ls(roll)                                   # long top third / short bottom third
    # inverse-vol sizing within each side + weekly hold + 1-day lag (no look-ahead) via shared block
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

    # --- align overlap, ex-post vol-match each sleeve to ~10%, blend 50/50 (the validated structure) ---
    df = pd.concat([carry, trend], axis=1).dropna()
    combo = blend * _vol_scale(df["carry"]) + (1 - blend) * _vol_scale(df["trend"])
    combo = combo.dropna(); combo.name = "commod_term_carry_trend"

    # --- trades = trend sign-runs (4 sectors) + carry sign-runs (Energy), within the overlap ---
    lo = df.index.min()
    trades = [t for t in ttrades if pd.Timestamp(t["entry_date"]) >= lo]
    trades += _sign_run_trades(pos.loc[pos.index >= lo], rets, "Energy", lo)
    return combo, trades


SPEC = StrategySpec(
    id="commod-term-carry-trend-book",
    family="commod_carry_trend_combo",
    title="Commodity term-structure carry + Trend two-premium book (non-crypto carry leg)",
    markets=["futures"],
    data_desc=("FREE: energy front futures CL/BZ/NG/HO/RB=F (yfinance 2005/2006+) + FRED daily hub spot "
               "DCOILWTICO/DCOILBRENTEU/DHHNGSP/DDFUELUSGULF/DGASUSGULF for the front-vs-near basis; "
               "Boreas 21-market trend (yfinance). Gate-0: free daily term structure is energy-only via "
               "the sanctioned adapters -> carry scoped to a 5-name energy sleeve; trend supplies breadth."),
    pre_registration=(
        "COMBINATION test re-using the ONE validated structure (carry+trend) with a NEW non-crypto carry "
        "leg. CARRY = commodity term-structure / hedging-pressure premium: roll-yield proxy "
        "carry_i = spot_i/front_i - 1 (>0 backwardation = positive roll). FREE-DATA CEILING (frozen, "
        "disclosed BEFORE the verdict): clean same-hub basis for WTI/Brent/NatGas; HeatingOil/Gasoline use "
        "Gulf ULSD/RBOB spot as a documented location proxy; metals/ags daily spot are NOT free so the leg "
        "is energy-scoped. Each rebalance: cross-sectional rank the 5-name complex, long the most-"
        "backwardated third / short the most-contango'd third (k=round(n/3) per side), inverse-vol within "
        "each side, dollar-neutral, hold the FRONT continuous future, WEEKLY rebalance, 8bps on turnover, "
        "signals lagged 1 day (no look-ahead). TREND = FROZEN Boreas 21-market TSMOM. Each sleeve ex-post "
        "vol-matched to ~10% and summed 50/50; no extra leverage. PRIMARY pre-registered metric = combined-"
        "book net Sharpe AND max-drawdown reduction vs carry-alone on the write-once 2022+ holdout via "
        "CPCV/DSR/PBO/FDR; standalone legs are diagnostics only. HYPOTHESIS: a physical (convenience-yield/"
        "hedging-pressure) carry premium with corr(carry,trend)~0 lets the crisis-alpha trend leg cut the "
        "carry drawdown at little Sharpe cost -- an independent, non-crypto-concentrated carry source that "
        "de-risks the book ahead of the 2026-08-28 crypto-carry forward verdict."),
    load_data=load_data, signal=signal,
    default_params={},
    grid={                                   # pre-declared design alternatives (honest DSR search burden)
        "default": {},
        "blend_carry": {"blend": 0.65},
        "blend_trend": {"blend": 0.35},
        "carryvol_low": {"carry_vol": 0.07},
        "reb_monthly": {"carry_rebalance": "ME"},
        "vollb_120": {"vol_lb": 120},
    },
    holdout_start="2022-01-01",
    deploy_max_positions=26,   # 5 energy carry names + 21 trend markets
)