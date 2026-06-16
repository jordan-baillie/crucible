"""
Illiquidity-Premium x Trend two-premium book - NATIVE commodity-futures sub-sleeve variant.

FAITHFUL TO PROPOSAL: the trend sleeve's commodity slice is built on the owned 17-root
Databento futures complex (PA dropped, ~91%% rank-2 coverage) using WITHIN-CONTRACT returns
(NEVER differencing close_1 across a roll), CONDITIONED on the term-structure / storage /
positioning trio, REPLACING the DBC ETF proxy. SPY/EFA/TLT/GLD remain liquid-ETF trend legs;
the Amihud illiquidity leg is unchanged. DBC is retained ONLY as an optional grid arm for the
DBC-vs-native diagnostic the proposal pre-registers.

ADAPTER INTERFACE ASSUMPTIONS (reconcile with the harness if the names/columns differ):
  fut_curve(root, start) -> DataFrame indexed by trade date with columns
      {'close_1','close_2'} (front/second nearby settle), optionally {'contract_1','ret_1'}.
      Within-contract return uses 'ret_1' if present, else close_1.pct_change() with roll-day
      jumps masked via 'contract_1' (NEVER diff close_1 across a roll).
  eia_series(code, start) / usda_nass(root, start) -> Series indexed by the PIT RELEASE date
      (value as of public release, not the reference week) -> seasonal anomaly z-score.
  cot_positioning(root, start) -> DataFrame indexed by Friday RELEASE date with {'comm_net','oi'}
      -> capped, non-gating secondary hedging-pressure tilt.
All conditioning series are reindexed to trade dates with ffill (known only AFTER release) so
there is no look-ahead. Roots with missing curves/conditioning degrade gracefully (term
structure alone, or skipped), matching 'standard size otherwise / PA dropped'.

NO LOOK-AHEAD: every weight matrix is resampled to a weekly (W-FRI) hold then explicitly
.shift(1) before net_of_cost / trades_from_weights consume it. Costs: 8 bps on turnover.
"""
from sdk.harness import StrategySpec
from sdk.adapters import (sep_panel, yf_panel, fut_curve, eia_series, usda_nass,
                          cot_positioning)
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2010-01-01"
ETF_TREND = ["SPY", "EFA", "TLT", "GLD"]          # liquid cross-asset trend legs (DBC removed)
TREND_ALL = ETF_TREND + ["DBC"]                    # DBC loaded only for the diagnostic arm
COMM_ROOTS = ["CL", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
              "ZC", "ZS", "ZW", "ZL", "ZM", "LE", "HE", "GF"]   # PA dropped
METALS = {"GC", "SI", "HG", "PL"}                  # no public storage series -> TS + COT only
EIA_CODE = {"CL": "WCESTUS1", "NG": "NW2_EPG0_SWO_R48_BCF",
            "HO": "WDISTUS1", "RB": "WGTSTUS1"}    # PIT weekly stocks (release-dated)
DBC_WEIGHT_SHARE = 0.2     # DBC was 1 of 5 ~equal trend instruments -> commodity-slice share

_SECTOR_MAP, _CURVES, _INVZ, _COT = {}, {}, {}, {}


# ----------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    global _SECTOR_MAP, _CURVES, _INVZ, _COT
    tickers, sector_map = sector_universe(marketcap="Small", top_n_per_sector=100)
    _SECTOR_MAP = sector_map
    close = sep_panel(tickers, START, field="closeadj")        # survivorship-clean, owned
    vol = sep_panel(tickers, START, field="volume")
    etf = yf_panel(TREND_ALL, START).reindex(close.index).ffill(limit=3)
    panel = pd.concat({"close": close, "volume": vol, "trend": etf}, axis=1)
    panel.attrs["sector_map"] = sector_map

    curves, invz, cot = {}, {}, {}
    for r in COMM_ROOTS:
        try:
            curves[r] = fut_curve(r, START)
        except Exception:
            continue
        try:
            if r in EIA_CODE:
                invz[r] = _seasonal_z(eia_series(EIA_CODE[r], START))
            elif r not in METALS:
                invz[r] = _seasonal_z(usda_nass(r, START))
        except Exception:
            pass
        try:
            cot[r] = cot_positioning(r, START)
        except Exception:
            pass
    _CURVES, _INVZ, _COT = curves, invz, cot
    panel.attrs["curves"], panel.attrs["invz"], panel.attrs["cot"] = curves, invz, cot
    return panel


def load_gen_data(label) -> pd.DataFrame:
    """scope='local' -> stage-2 battery not run; provided for interface completeness only."""
    return load_data()


# ----------------------------------------------------------------- shared helpers
def _weekly_lag(W, idx):
    """Weekly (W-FRI) rebalance, hold between, then EXPLICIT 1-day lag. Returned matrix is
    already lagged -> net_of_cost / trades_from_weights consume it directly (no extra shift)."""
    return (W.resample("W-FRI").last()
             .reindex(idx, method="ffill")
             .shift(1)
             .fillna(0.0))


def _seasonal_z(s):
    """PIT seasonal-anomaly z of an inventory level indexed by RELEASE date (past-only)."""
    s = pd.Series(s).sort_index()
    anom = s - s.rolling(252, min_periods=60).mean()
    z = anom / anom.rolling(252, min_periods=60).std()
    return z.replace([np.inf, -np.inf], np.nan)


def _within_contract_returns(curve):
    """NEVER diff close_1 across a roll. Prefer adapter 'ret_1'; else mask roll-day jumps."""
    if "ret_1" in curve.columns:
        return curve["ret_1"]
    r = curve["close_1"].pct_change()
    if "contract_1" in curve.columns:
        roll = curve["contract_1"].ne(curve["contract_1"].shift(1))
        r = r.mask(roll, 0.0)
    return r


# ----------------------------------------------------------------- weight construction
def _amihud_weights(close, volume, illiq_lb, vol_lb):
    rets = close.pct_change()
    dvol = (close * volume).replace(0.0, np.nan)
    illiq = (rets.abs() / dvol)                                 # Amihud price-impact, daily
    illiq_ma = illiq.rolling(illiq_lb, min_periods=illiq_lb // 2).mean()
    illiq_ma = illiq_ma.where(illiq_ma > 0)
    z = xs_zscore(np.log(illiq_ma))                             # +z = illiquid -> LONG
    vol_d = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()
    raw = (z / vol_d).replace([np.inf, -np.inf], np.nan)        # inverse-vol size
    raw = raw.sub(raw.mean(axis=1), axis=0)                     # dollar-neutral L/S
    Wt = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0)  # gross = 1
    return _weekly_lag(Wt, close.index), rets


def _etf_trend_returns(etf, mom_lb, vol_lb, name):
    rets = etf.pct_change()
    sig = np.sign(etf.pct_change(mom_lb))                       # canonical TS-momentum sign
    vol_d = rets.rolling(vol_lb, min_periods=vol_lb // 2).std()
    raw = (sig / vol_d).replace([np.inf, -np.inf], np.nan)      # inverse-vol size
    Wt = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    return net_of_cost(_weekly_lag(Wt, etf.index), rets, cost_bps=8.0, name=name)


def _commodity_sleeve(idx, mom_lb, vol_lb):
    """Native roll-aware commodity-futures trend, trio-conditioned, REPLACING DBC."""
    weights, comm_rets = {}, {}
    for r, curve in _CURVES.items():
        if "close_1" not in curve.columns:
            continue
        wret = _within_contract_returns(curve).reindex(idx).fillna(0.0)
        # canonical trend on WITHIN-CONTRACT cumulative return (roll-safe)
        cum = (1.0 + wret).cumprod()
        trend_sign = np.sign(cum.pct_change(mom_lb)).fillna(0.0)
        # term-structure carry: contango (slope>0) bearish, backwardation bullish
        if "close_2" in curve.columns:
            slope = (curve["close_2"] / curve["close_1"] - 1.0).reindex(idx).ffill()
            carry_ts = -np.sign(slope).fillna(0.0)
        else:
            carry_ts = pd.Series(0.0, index=idx)
        # storage carry: high inventory (vs seasonal norm) bearish, low bullish (PIT release)
        if r in _INVZ:
            inv = _INVZ[r].reindex(idx).ffill()
            carry_inv = -np.sign(inv).fillna(0.0)
        else:
            carry_inv = pd.Series(0.0, index=idx)
        carry_sign = np.sign(carry_ts + carry_inv)
        # TRIO CONFIRMATION: take the trend position only if carry/storage AGREES in sign
        pos = trend_sign.where(trend_sign == carry_sign, 0.0).fillna(0.0)
        # COT hedging-pressure: capped, non-gating secondary tilt (Friday release)
        if r in _COT and {"comm_net", "oi"}.issubset(_COT[r].columns):
            c = _COT[r]
            hp = c["comm_net"] / c["oi"].replace(0.0, np.nan)
            hpz = ((hp - hp.rolling(104, min_periods=26).mean())
                   / hp.rolling(104, min_periods=26).std()).reindex(idx).ffill()
            tilt = (1.0 + 0.2 * np.tanh(hpz.fillna(0.0))).clip(0.8, 1.2)
            pos = pos * tilt
        vol_d = wret.rolling(vol_lb, min_periods=vol_lb // 2).std()
        weights[r] = (pos / vol_d).replace([np.inf, -np.inf], np.nan)
        comm_rets[r] = wret
    if not weights:
        return pd.Series(0.0, index=idx, name="commodity_native")
    W = pd.DataFrame(weights).reindex(idx)
    R = pd.DataFrame(comm_rets).reindex(idx).fillna(0.0)
    W = W.div(W.abs().sum(axis=1).replace(0.0, np.nan), axis=0)   # gross = 1
    W = _weekly_lag(W, idx)
    return net_of_cost(W, R, cost_bps=8.0, name="commodity_native")


# ------------------------------------------------------------------------------ signal
def signal(panel, **params):
    illiq_lb = int(params.get("illiq_lb", 60))
    vol_lb = int(params.get("vol_lb", 60))
    mom_lb = int(params.get("mom_lb", 126))
    trend_risk = float(params.get("trend_risk", 0.25))
    use_dbc_proxy = bool(params.get("use_dbc_proxy", False))

    close = panel["close"].dropna(how="all", axis=1)
    volume = panel["volume"].reindex(columns=close.columns)
    etf_all = panel["trend"].dropna(how="all", axis=1)
    etf = etf_all[[c for c in ETF_TREND if c in etf_all.columns]]
    idx = close.index

    # --- LEG A: Amihud illiquidity (the alpha book the ledger/regime gates judge) ---
    W_amihud, eq_rets = _amihud_weights(close, volume, illiq_lb, vol_lb)
    a_rets = net_of_cost(W_amihud, eq_rets, cost_bps=8.0, name="amihud")   # W already lagged
    trades = trades_from_weights(W_amihud, eq_rets, _SECTOR_MAP)           # kit stamps regime

    book = a_rets.copy()
    if trend_risk > 0:
        # --- LEG B: cross-asset trend crisis-alpha sleeve ---
        etf_t = _etf_trend_returns(etf, mom_lb, vol_lb, "trend_etf").reindex(idx).fillna(0.0)
        if use_dbc_proxy and "DBC" in etf_all.columns:
            comm_t = _etf_trend_returns(etf_all[["DBC"]], mom_lb, vol_lb,
                                        "trend_dbc").reindex(idx).fillna(0.0)
        else:
            comm_t = _commodity_sleeve(idx, mom_lb, vol_lb).reindex(idx).fillna(0.0)
        # commodity slice INHERITS DBC's weight share inside the trend sleeve
        t_rets = (1.0 - DBC_WEIGHT_SHARE) * etf_t + DBC_WEIGHT_SHARE * comm_t
        df = pd.concat([a_rets, t_rets], axis=1).reindex(idx).fillna(0.0)
        a, t = df.iloc[:, 0], df.iloc[:, 1]
        av = a.rolling(60, min_periods=20).std().shift(1)
        tv = t.rolling(60, min_periods=20).std().shift(1)
        t_matched = (t * (av / tv)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        book = a + trend_risk * t_matched

    book = book.reindex(idx).fillna(0.0)
    book.name = "amihud_illiq_x_trend_native_commodity_book"
    return book, trades


# ------------------------------------------------------------- pre-registered checks
def _sharpe(r):
    r = r.dropna(); sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0


def _maxdd(r):
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _legs(ctx):
    g = ctx.get("grid", {})
    return g.get("default"), g.get("amihud_only"), g.get("dbc_proxy")


def check_leg_corr(ctx):
    """Amihud vs trend-sleeve leg correlation <= +0.1 (complementarity preserved)."""
    full, am, _ = _legs(ctx)
    if full is None or am is None:
        return {"pass": False, "observed": "grid_missing"}
    df = pd.concat([full, am], axis=1).dropna()
    trend_leg = df.iloc[:, 0] - df.iloc[:, 1]
    c = float(df.iloc[:, 1].corr(trend_leg))
    return {"pass": bool(c <= 0.1), "observed": round(c, 4)}


def check_maxdd_reduced(ctx):
    """combined MaxDD reduced >= 20% vs standalone Amihud."""
    full, am, _ = _legs(ctx)
    if full is None or am is None:
        return {"pass": False, "observed": "grid_missing"}
    dc, da = _maxdd(full), _maxdd(am)
    red = (1.0 - abs(dc) / abs(da)) if da != 0 else 0.0
    return {"pass": bool(abs(dc) <= 0.8 * abs(da)), "observed": round(float(red), 4)}


def check_sharpe_preserved(ctx):
    """Sharpe degradation from adding the hedge sleeve <= 10%."""
    full, am, _ = _legs(ctx)
    if full is None or am is None:
        return {"pass": False, "observed": "grid_missing"}
    sc, sa = _sharpe(full), _sharpe(am)
    deg = (1.0 - sc / sa) if sa > 0 else 1.0
    return {"pass": bool(deg <= 0.10), "observed": round(float(deg), 4)}


def check_native_not_worse_than_dbc(ctx):
    """ON-THESIS test: the NATIVE roll-aware commodity sub-sleeve must NOT degrade the
    trend-sleeve Sharpe vs the DBC-proxy version (improvement expected). delta >= -0.05."""
    full, am, dbc = _legs(ctx)
    if full is None or am is None or dbc is None:
        return {"pass": False, "observed": "grid_missing"}
    df = pd.concat([am, full, dbc], axis=1).dropna()
    trend_native = df.iloc[:, 1] - df.iloc[:, 0]
    trend_dbc = df.iloc[:, 2] - df.iloc[:, 0]
    delta = _sharpe(trend_native) - _sharpe(trend_dbc)
    return {"pass": bool(delta >= -0.05), "observed": round(float(delta), 4)}


# -------------------------------------------------------------------------------- spec
SPEC = StrategySpec(
    id="amihud_illiq_x_trend_native_commodity_v1",
    family="illiquidity_premium_x_trend",
    title=("Illiquidity-Premium x Trend Two-Premium Book - NATIVE roll-aware commodity-futures "
           "sub-sleeve (trio-conditioned) REPLACING the DBC ETF proxy"),
    markets=["US small-cap equities (Amihud illiquidity)",
             "cross-asset trend: {SPY,EFA,TLT,GLD} ETFs + native commodity-futures sub-sleeve "
             "over {CL,NG,HO,RB,GC,SI,HG,PL,ZC,ZS,ZW,ZL,ZM,LE,HE,GF}"],
    data_desc=("Sharadar SEP closeadj+volume on survivorship-clean small-caps (Amihud, owned); "
               "yfinance daily closes for SPY/EFA/TLT/GLD (free); owned Databento fut_curve for the "
               "17-root commodity complex (within-contract returns, term structure); eia_series / "
               "usda_nass storage (PIT release-dated); cot_positioning hedging-pressure. $0 "
               "incremental data (all owned/free)."),
    pre_registration=(
        "Two-premium book: (A) VALIDATED Amihud illiquidity premium - long the most-illiquid "
        "small-caps, short the liquid names, inverse-vol sized, dollar-neutral, weekly rebalanced. "
        "(B) canonical TS-trend crisis-alpha sleeve = liquid {SPY,EFA,TLT,GLD} legs PLUS a NATIVE "
        "commodity-futures sub-sleeve that REPLACES DBC. The native sub-sleeve computes the canonical "
        "trend on WITHIN-CONTRACT returns (never diffing close_1 across a roll), then applies a TRIO "
        "CONFIRMATION FILTER: take the trend position only if term-structure (close_2/close_1 slope) "
        "and storage (eia/usda seasonal anomaly, conditioned on PIT RELEASE date) AGREE in sign; COT "
        "hedging-pressure is a capped, non-gating secondary tilt. The sub-sleeve inherits DBC's weight "
        "share inside the trend sleeve; trend sleeve at 25% of book risk, vol-matched to Amihud. "
        "Signals are weekly-held and explicitly .shift(1) lagged; 8 bps turnover cost; conditioning "
        "series are release-dated/ffilled (no look-ahead). SCOPE='local': both premia are settled "
        "standalone; open claims are book-level. PRE-REGISTERED success: combined MaxDD reduced >=20% "
        "vs standalone Amihud, Sharpe degradation <=10%, Amihud-vs-trend leg correlation <=+0.1, AND "
        "the native sub-sleeve must NOT degrade trend-sleeve Sharpe vs the DBC-proxy arm (improvement "
        "expected). The 'dbc_proxy' grid arm exists ONLY for that diagnostic comparison."),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                          # primary: NATIVE commodity sub-sleeve
        "amihud_only": {"trend_risk": 0.0},     # standalone alpha leg
        "dbc_proxy": {"use_dbc_proxy": True},    # DBC-ETF slice (native-vs-DBC diagnostic)
        "illiq_lb_120": {"illiq_lb": 120},       # declared search variants -> honest DSR eff-N
        "mom_lb_252": {"mom_lb": 252},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=10,
    expectations=[
        {"name": "leg_corr_low",
         "claim": "Amihud vs trend-sleeve leg correlation <= +0.1",
         "check": check_leg_corr},
        {"name": "maxdd_reduced_20pct",
         "claim": "combined book MaxDD reduced >= 20% vs standalone Amihud",
         "check": check_maxdd_reduced},
        {"name": "sharpe_preserved",
         "claim": "combined-book Sharpe degradation vs standalone Amihud <= 10%",
         "check": check_sharpe_preserved},
        {"name": "native_not_worse_than_dbc",
         "claim": "native commodity sub-sleeve does not degrade trend-sleeve Sharpe vs DBC "
                  "proxy (delta_Sharpe >= -0.05) - validates the DBC->native replacement",
         "check": check_native_not_worse_than_dbc},
    ],
)