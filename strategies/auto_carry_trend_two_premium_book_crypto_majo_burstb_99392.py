"""
Carry x Trend Two-Premium Book -- CRYPTO-MAJORS DEPLOYABLE PORT
==============================================================
Ports the portfolio's ONE confirmed edge (the COMBINATION of opposite-tail premia:
pro-cyclical carry hedged by crisis-alpha trend) out of the physics-dead retail-equity
domain (small-cap illiquidity short = un-borrowable => un-fundable) into the one domain
where the short leg is genuinely deployable: liquid crypto-major PERPS, where shorts trade
freely with no stock-borrow wall.

LEG 1 (pro-cyclical) -- cross-sectional funding-DISPERSION carry: each day go long the
   most-NEGATIVE-funding majors (you are paid to be long) and short the most-POSITIVE-funding
   majors (paid to be short), equal-weight dollar-neutral; gross SCALED by realized
   cross-sectional funding dispersion vs its own trailing reference, so when dispersion -> 0
   (carry has no fuel, e.g. the 2025-26 compressed regime) the leg shrinks toward flat and
   trend dominates.  This is RELATIVE-spread carry, NOT the DORMANT delta-neutral single-coin
   carry (which compressed to ~0 and is intentionally not re-proposed).  CRITICAL: the funding
   cashflow is COLLECTED in the P&L (-w*f per position), because the premium IS the funding,
   not a price prediction.
LEG 2 (defensive)    -- canonical frozen time-series-trend rule (sign-average of 21/63/126/252d),
   inverse-vol sized, evaluated DAILY so it stays responsive to fast crashes.

Costs: a CONSERVATIVE fixed 20bps round-trip taker (real major-perp taker ~6-10bps).  No look-
ahead: every signal uses trailing windows ONLY and the executed weight matrix is held.shift(1)
(day-t weights earn day-(t+1) returns; net_of_cost receives the ALREADY-LAGGED matrix).  Perp
funding accrues on the held (already-lagged) position over the holding day, so it is added with
no further shift.  Daily funding uses resample-mean (conservatively ~1 payment/day, never
overstating the premium); if the adapter is unavailable the carry leg degrades to flat and the
book reduces to the deployable TREND-ONLY baseline (a pre-registered acceptable null).
"""
import numpy as np, pandas as pd
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import net_of_cost, trades_from_weights

# Guarded crypto-funding adapter (referenced in markets/crypto.md / DATA_CATALOG.md).
# If absent -> carry leg flat -> book = deployable trend-only (pre-registered acceptable null).
try:
    from sdk.adapters import funding_rates as _funding_rates
except Exception:
    _funding_rates = None

# ---------------- pre-registered FROZEN universe & constants (single spec, no tuning) -----------
MAJORS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
          "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD"]
SECTOR_MAP = {
    "BTC-USD": "store-of-value", "ETH-USD": "smart-contract-L1", "SOL-USD": "smart-contract-L1",
    "BNB-USD": "exchange-L1", "XRP-USD": "payments", "ADA-USD": "smart-contract-L1",
    "DOGE-USD": "payments", "AVAX-USD": "smart-contract-L1", "LINK-USD": "oracle-defi",
    "LTC-USD": "payments",
}
START      = "2019-01-01"
FUND_LB    = 7                       # trailing-day funding mean
TREND_LBS  = (21, 63, 126, 252)      # canonical multi-lookback sign-average
VOL_LB     = 60                      # trailing realized-vol window
TARGET_VOL = 0.20                    # whole-book annualized vol cap
GROSS_CAP  = 2.0                     # hard gross cap (<=2x)
TREND_RISK = 0.50                    # crisis-alpha hedge sized at half the carry leg's risk
COST_BPS   = 20.0                    # conservative round-trip taker assumption
REL_BAND   = 0.20                    # no-trade band: relative drift threshold
ABS_FLOOR  = 0.01                    # ~$50 on a $5k book: suppress sub-floor noise rebalances
EPS        = 1e-9


# --------------------------------------- data plumbing -----------------------------------------
def _to_fund_series(obj):
    """Normalize whatever funding_rates returns (Series / wide-1col / long DataFrame) to a Series."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0].copy()
        else:
            col = next((c for c in ["fundingRate", "funding_rate", "funding", "rate", "value"]
                        if c in obj.columns), None)
            if col is None:
                num = obj.select_dtypes("number")
                if num.shape[1] == 0:
                    return None
                col = num.columns[-1]
            idxc = next((c for c in ["date", "fundingTime", "time", "timestamp", "datekey"]
                         if c in obj.columns), None)
            s = obj.set_index(idxc)[col] if idxc else obj[col].copy()
    else:
        return None
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        return None
    return pd.to_numeric(s, errors="coerce").dropna()


def _funding_panel(price_cols):
    """Per-coin Binance perp funding -> daily wide panel keyed by the yfinance ticker. Robust/guarded."""
    if _funding_rates is None:
        return pd.DataFrame()
    out = {}
    for t in price_cols:
        sym = t.replace("-USD", "USDT")
        ser = None
        for call in (lambda: _funding_rates(sym, START), lambda: _funding_rates(sym)):
            try:
                ser = _to_fund_series(call())
                if ser is not None and len(ser):
                    break
            except Exception:
                ser = None
        if ser is not None and len(ser):
            out[t] = ser.sort_index()
    if not out:
        return pd.DataFrame()
    F = pd.DataFrame(out).sort_index()
    F = F[~F.index.duplicated(keep="last")]
    return F.resample("1D").mean()


def load_data() -> pd.DataFrame:
    """Panel signal() consumes: spot-close columns + 'fund::'-prefixed funding columns."""
    P = yf_panel(MAJORS, START)
    if isinstance(P, pd.Series):
        P = P.to_frame()
    P = P.reindex(columns=[c for c in MAJORS if c in P.columns]).astype(float)
    P.index = pd.to_datetime(P.index)
    P = P.sort_index().dropna(how="all")
    F = _funding_panel(list(P.columns))
    if len(F):
        F = F.reindex(P.index).ffill(limit=3).reindex(columns=P.columns)
        F.columns = ["fund::" + c for c in F.columns]
        return pd.concat([P, F], axis=1)
    return P.copy()


def _split(panel):
    price_cols = [c for c in panel.columns if not str(c).startswith("fund::")]
    P = panel[price_cols].astype(float)
    fcols = [c for c in panel.columns if str(c).startswith("fund::")]
    if fcols:
        F = panel[fcols].astype(float)
        F.columns = [c[6:] for c in fcols]
        F = F.reindex(columns=price_cols)
    else:
        F = pd.DataFrame(index=P.index, columns=price_cols, dtype=float)
    return P, F


# ------------------------------------------ legs ----------------------------------------------
def _carry_weights(P, F):
    """Cross-sectional funding-dispersion carry: long most-negative funding, short most-positive,
    equal-weight dollar-neutral, gross scaled by trailing cross-sectional funding dispersion."""
    cols = P.columns
    if F.dropna(how="all").empty:
        return pd.DataFrame(0.0, index=P.index, columns=cols)
    mu = F.rolling(FUND_LB, min_periods=max(2, FUND_LB // 2)).mean()
    sig = -mu                                            # long where funding negative (paid to be long)
    ranks = sig.rank(axis=1, method="first")
    nv = sig.notna().sum(axis=1)
    k = (nv // 3).clip(lower=1)
    valid = (nv >= 4).astype(float)
    long_mask = ranks.gt(nv - k, axis=0)                 # highest carry sig = most-negative funding
    short_mask = ranks.le(k, axis=0)
    w = long_mask.mul(0.5 / k, axis=0).fillna(0.0) - short_mask.mul(0.5 / k, axis=0).fillna(0.0)
    w = w.mul(valid, axis=0)                             # dollar-neutral, gross 1 per active day
    # dispersion gate: shrink toward flat when the cross-coin funding spread collapses (fuel-less carry)
    disp = mu.std(axis=1)
    ref = disp.expanding(min_periods=VOL_LB).median()    # trailing-only reference, no lookahead
    scaler = (disp / (ref + EPS)).clip(lower=0.0, upper=1.0).fillna(0.0)
    return w.mul(scaler, axis=0).reindex(columns=cols).fillna(0.0)


def _trend_weights(P):
    """Canonical time-series trend: nan-aware sign-average over lookbacks, inverse-vol sized, gross 1."""
    R = P.pct_change()
    acc = cnt = None
    for L in TREND_LBS:
        s = np.sign(P / P.shift(L) - 1.0)
        acc = s if acc is None else acc.add(s, fill_value=0.0)
        c = s.notna().astype(float)
        cnt = c if cnt is None else cnt.add(c, fill_value=0.0)
    sig = acc / cnt.replace(0, np.nan)
    vol = R.rolling(VOL_LB, min_periods=20).std()
    raw = (sig / (vol + EPS)).replace([np.inf, -np.inf], np.nan)
    gross = raw.abs().sum(axis=1)
    return raw.div(gross + EPS, axis=0).fillna(0.0)


def _apply_band(W):
    """No-trade band: only rebalance a name when its target drifts >REL_BAND (relative) AND the
    absolute change clears the ~$50 ABS_FLOOR; otherwise carry the prior weight forward."""
    A = W.fillna(0.0).values
    held = np.zeros(A.shape[1])
    out = np.empty_like(A)
    for i in range(A.shape[0]):
        tgt = A[i]
        delta = tgt - held
        rel = np.abs(delta) / np.maximum(np.abs(held), ABS_FLOOR)
        do = (rel > REL_BAND) & (np.abs(delta) >= ABS_FLOOR)
        held = np.where(do, tgt, held)
        out[i] = held
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def _fund_pnl(W, Ff):
    """Perp funding cashflow on positions: a holder PAYS positive funding when long and RECEIVES it
    when short -> per-position cashflow = -w*f.  W is the (unlagged) target weight matrix; the
    position held during day t accrues day-t funding, so we use W.shift(1).  This is the actual
    carry premium being collected (long negative-funding/short positive-funding => systematically
    positive), NOT a price prediction."""
    return -(W.shift(1) * Ff).sum(axis=1)


# ----------------------------------------- signal ---------------------------------------------
def signal(panel, carry_off=False, trend_off=False, **params):
    P, _F = _split(panel)
    P = P.dropna(how="all")
    F = _F.reindex(index=P.index)
    Ff = F.fillna(0.0)                                   # funding cashflow inputs (0 where no data)
    R = P.pct_change().fillna(0.0)

    Wc = (pd.DataFrame(0.0, index=P.index, columns=P.columns)
          if carry_off else _carry_weights(P, F))
    Wt = (pd.DataFrame(0.0, index=P.index, columns=P.columns)
          if trend_off else _trend_weights(P))

    # leg risk equalization (carry 100% risk, trend TREND_RISK), trailing-vol scaled; weights lagged.
    # Leg returns INCLUDE the funding cashflow so vol is measured on the true (price+funding) P&L.
    rc = (Wc.shift(1) * R).sum(axis=1) + _fund_pnl(Wc, Ff)
    rt = (Wt.shift(1) * R).sum(axis=1) + _fund_pnl(Wt, Ff)
    sc = (1.0 / (rc.rolling(VOL_LB, min_periods=20).std() + EPS)).clip(0, 50).fillna(0.0)
    st = (TREND_RISK / (rt.rolling(VOL_LB, min_periods=20).std() + EPS)).clip(0, 50).fillna(0.0)
    Wcomb = Wc.mul(sc, axis=0) + Wt.mul(st, axis=0)      # zero-weight legs stay zero (0*finite=0)

    # whole-book vol target ~20% ann + hard gross cap 2x (trailing-only stats, funding-inclusive)
    rcomb = (Wcomb.shift(1) * R).sum(axis=1) + _fund_pnl(Wcomb, Ff)
    bv = rcomb.rolling(VOL_LB, min_periods=20).std()
    bscale = ((TARGET_VOL / np.sqrt(252.0)) / (bv + EPS)).clip(0, 3.0).fillna(0.0)
    Wcomb = Wcomb.mul(bscale, axis=0)
    gross = Wcomb.abs().sum(axis=1)
    Wcomb = Wcomb.mul((GROSS_CAP / (gross + EPS)).clip(upper=1.0), axis=0)

    # pre-registered no-trade band, then LAG 1 day (day-t weights earn day-(t+1) returns)
    held = _apply_band(Wcomb)
    Wexec = held.shift(1).fillna(0.0)

    name = "carryXtrend_crypto" + ("_trendonly" if carry_off else "") + ("_carryonly" if trend_off else "")
    # price return net of 20bps taker on the (already-lagged) executed book ...
    rets = net_of_cost(Wexec, R, cost_bps=COST_BPS, name=name)   # Wexec already lagged
    # ... PLUS the perp funding cashflow collected on the held (already-lagged) position. This is the
    # cross-sectional funding-dispersion carry premium actually being harvested.
    fund = -(Wexec * Ff).sum(axis=1)
    rets = rets.add(fund, fill_value=0.0)
    rets.name = name
    smap = {t: SECTOR_MAP.get(t, "crypto") for t in P.columns}
    trades = trades_from_weights(Wexec, R, smap)                 # entry_regime stamped by the kit
    return rets, trades


# --------------------------- soft expectations (machine-checkable claims) ----------------------
def _sharpe(r):
    r = pd.Series(r).dropna()
    sd = r.std()
    return float((r.mean() / sd) * np.sqrt(252.0)) if len(r) >= 20 and sd > 0 else 0.0


def _maxdd(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return 0.0
    eq = (1.0 + r).cumprod()
    return float(abs((eq / eq.cummax() - 1.0).min()))


def _chk_maxdd(ctx):
    g = ctx.get("grid", {})
    comb, tr = g.get("default"), g.get("trend_only")
    if comb is None or tr is None:
        return {"pass": False, "observed": "missing"}
    md_t = _maxdd(tr)
    if md_t <= 0:
        return {"pass": False, "observed": 0.0}
    ratio = _maxdd(comb) / md_t
    return {"pass": ratio <= 0.80, "observed": round(ratio, 3)}    # >=20% DD reduction vs trend-only


def _chk_sharpe(ctx):
    g = ctx.get("grid", {})
    comb, tr = g.get("default"), g.get("trend_only")
    if comb is None or tr is None:
        return {"pass": False, "observed": "missing"}
    sc, stt = _sharpe(comb), _sharpe(tr)
    if stt <= 0:
        return {"pass": sc >= 0, "observed": round(sc, 3)}
    deg = (stt - sc) / abs(stt)
    return {"pass": deg <= 0.10, "observed": round(deg, 3)}        # net-Sharpe degradation <=10%


def _chk_corr(ctx):
    g = ctx.get("grid", {})
    c, t = g.get("carry_only"), g.get("trend_only")
    if c is None or t is None:
        return {"pass": False, "observed": "missing"}
    j = pd.concat([pd.Series(c), pd.Series(t)], axis=1).dropna()
    if len(j) < 20:
        return {"pass": False, "observed": "insufficient"}
    rho = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    return {"pass": rho <= 0.10, "observed": round(rho, 3)}        # complementarity precondition


SPEC = StrategySpec(
    id="crypto_carry_x_trend_majors_v1",
    family="carry_trend_combo",
    title=("Carry x Trend two-premium book on liquid crypto-major perps "
           "(cross-sectional funding-DISPERSION carry hedged by canonical TS-trend; "
           "perp shorts deployable, no borrow wall; funding cashflow collected; "
           "net of conservative 20bps round-trip taker)"),
    markets=["crypto"],
    data_desc=("Daily yfinance spot closes for 10 major coins "
               "(BTC/ETH/SOL/BNB/XRP/ADA/DOGE/AVAX/LINK/LTC) + Binance perp funding via "
               "funding_rates() 2019+ (owned/free). Returns = spot price return + perp funding "
               "cashflow (-w*f) net of a fixed 20bps round-trip taker. funding_rates is guarded: "
               "if unavailable the carry leg is flat and the book is the deployable trend-only "
               "baseline."),
    pre_registration=(
        "FROZEN single spec. (1) CARRY (pro-cyclical): rank majors by trailing-7d mean funding; "
        "long most-negative-funding (paid to be long), short most-positive-funding (paid to be short), "
        "equal-weight dollar-neutral; gross scaled by trailing cross-sectional funding dispersion vs its "
        "expanding-median reference -> dispersion->0 shrinks the leg to flat (fuel-less carry honestly "
        "reduces the book to trend-only). Requires >=4 funded coins/day. The FUNDING CASHFLOW is collected "
        "in the P&L (-w*f per position) -- the premium is the funding, not a price prediction. This is "
        "RELATIVE-spread carry, NOT the dormant single-coin delta-neutral carry. (2) TREND (defensive "
        "crisis-alpha): canonical nan-aware sign-average over 21/63/126/252d, inverse-vol sized, evaluated "
        "daily. (3) Leg risk: carry 100% / trend 50%, each vol-equalized (price+funding P&L) on trailing-60d, "
        "summed; book vol-targeted to ~20% ann, gross hard-capped 2x. (4) Execution: pre-registered no-trade "
        "band -- rebalance a name only when its target drifts >20% relative AND clears a ~$50 (1% of $5k) "
        "absolute floor; large carry/trend moves blow through and execute fully, only small daily noise is "
        "suppressed. (5) Costs: conservative 20bps round-trip taker on turnover; ALL gates on NET returns "
        "(price + funding - cost). NO LOOK-AHEAD: every signal uses trailing windows only; the executed "
        "weight matrix is held.shift(1) (day-t weights earn day-(t+1) returns), and funding accrues on the "
        "held (already-lagged) position over the holding day (no further shift). PRE-REGISTERED SUCCESS "
        "(net, vs the trend_only deployable baseline over the identical window): combined MaxDD reduced "
        ">=20%; net-Sharpe degradation <=10%; carry-leg vs trend-leg net-return correlation <= +0.10; plus "
        "the full gate stack (MCPT absolute null, write-once holdout, DSR over the 3 declared variants). "
        "DIAGNOSTICS (reported, not gated): gross-vs-net Sharpe and turnover WITH vs WITHOUT the band; "
        "stress-window sign (2020 COVID / 2021 peak / 2022 LUNA-FTX / 2025-26 compressed funding); realized "
        "cross-sectional funding dispersion -- if dispersion is dead in 2025-26 the carry leg is fuel-less "
        "and the book honestly reduces to a fundable trend-only null."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={"default": {}, "trend_only": {"carry_off": True}, "carry_only": {"trend_off": True}},
    scope="local",                       # single deployable book; forward validation confirms
    generalization_universes=[],
    load_gen_data=None,
    holdout_start="2023-01-01",          # search <=2022 (COVID/2021 peak/LUNA-FTX); holdout 2023-2026 incl. compressed funding
    deploy_max_positions=10,
    expectations=[
        {"name": "maxdd_reduced_20pct",
         "claim": "combined search-window MaxDD <= 80% of trend-only MaxDD (>=20% reduction)",
         "check": _chk_maxdd},
        {"name": "sharpe_degradation_le_10pct",
         "claim": "combined net-Sharpe degradation vs trend-only <= 10%",
         "check": _chk_sharpe},
        {"name": "carry_trend_corr_le_0p1",
         "claim": "carry-only vs trend-only net-return correlation <= +0.10 (complementarity)",
         "check": _chk_corr},
    ],
)