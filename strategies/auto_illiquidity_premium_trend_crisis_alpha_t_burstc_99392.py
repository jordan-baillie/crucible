"""
Illiquidity x Trend — CRYPTO-PERP two-premium book (LOCAL scope).

Honest data caveat baked in: the sanctioned crypto adapter (yf_panel) returns
CLOSE ONLY — no volume — so the canonical Amihud illiquidity numerator
(|ret| / $-volume) is NOT buildable with free/owned tools. Per the proposal's
Gate-0 fallback we substitute Roll's (1984) serial-covariance effective-spread
estimator, a standard PRICE-ONLY relative-illiquidity proxy (less-liquid names
show stronger bid-ask bounce -> more negative return autocovariance), and lower
the prior accordingly. Everything else is the parent's frozen design ported to
perps: cross-sectional long-illiquid/short-liquid (pro-cyclical) + a 25%
long/short TSMOM crisis-alpha overlay on the liquid majors (perps short freely
at ~20bps -> removes the borrow wall that made the equity version PHYSICS-DEAD).
No external side effects.

FIX (vs prior draft): the frozen design specifies DAILY EOD rebalance, no
intraday. The prior draft did a weekly (Mon->Mon hold) rebalance — a cadence
deviation that changed turnover/cost. Reverted to daily EOD + 1-day lag.
"""
from sdk.harness import StrategySpec
from sdk.adapters import yf_panel
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

# ---- MECHANICAL, FROZEN universe (written down BEFORE any backtest) ----
# majors = trend (crisis-alpha) basket AND members of the illiquidity cross-section;
# alts    = the rest of the deep, individually-tradable perp cross-section.
MAJORS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
ALTS = ["ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD",
        "BCH-USD", "TRX-USD", "XLM-USD", "ATOM-USD", "ETC-USD", "XMR-USD",
        "ALGO-USD", "VET-USD", "FIL-USD", "EOS-USD", "AAVE-USD", "XTZ-USD",
        "NEO-USD", "DASH-USD", "ZEC-USD", "WAVES-USD", "QTUM-USD", "ICX-USD",
        "ONT-USD"]
ALL_CRYPTO = MAJORS + ALTS
START = "2017-01-01"

# pseudo-sector map (crypto category) so the trade-ledger spread gates have spread
SECTOR_MAP = {
    "BTC-USD": "store-of-value", "LTC-USD": "store-of-value", "BCH-USD": "store-of-value",
    "ETH-USD": "L1-platform", "SOL-USD": "L1-platform", "ADA-USD": "L1-platform", "AVAX-USD": "L1-platform",
    "TRX-USD": "L1-legacy", "EOS-USD": "L1-legacy", "NEO-USD": "L1-legacy", "QTUM-USD": "L1-legacy", "WAVES-USD": "L1-legacy",
    "BNB-USD": "exchange",
    "XRP-USD": "payments", "XLM-USD": "payments", "ALGO-USD": "payments",
    "DOT-USD": "interop", "ATOM-USD": "interop", "ICX-USD": "interop", "ONT-USD": "interop",
    "LINK-USD": "defi", "AAVE-USD": "defi",
    "XMR-USD": "privacy", "ZEC-USD": "privacy", "DASH-USD": "privacy",
    "VET-USD": "infra", "FIL-USD": "infra", "ETC-USD": "infra", "XTZ-USD": "infra",
    "DOGE-USD": "meme",
}


def load_data() -> pd.DataFrame:
    """CLOSE panel for the frozen crypto universe (yf_panel = CLOSE only; no volume)."""
    px = yf_panel(ALL_CRYPTO, start=START)
    cols = [t for t in ALL_CRYPTO if t in px.columns]
    px = px[cols].astype(float).dropna(how="all")
    return px


def load_gen_data(label: str) -> pd.DataFrame:
    """scope='local' -> the stage-2 generalisation battery is NOT run; defined for
    spec completeness only (generalisation is handled by the recent write-once holdout
    + the 3 in-sample crisis windows declared in pre_registration)."""
    return load_data()


# ----------------------------- signal helpers -----------------------------
def _roll_illiq(rets: pd.DataFrame, lb: int) -> pd.DataFrame:
    """Roll (1984) effective-spread illiquidity proxy on RETURNS (scale-invariant):
       2*sqrt(max(-cov(r_t, r_{t-1}), 0)) over trailing lb. Higher = less liquid."""
    r1 = rets.shift(1)
    cov = (rets * r1).rolling(lb).mean() - rets.rolling(lb).mean() * r1.rolling(lb).mean()
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def _leg_scale(baseW: pd.DataFrame, rets: pd.DataFrame, target: float, lb: int = 60) -> pd.Series:
    """Per-date multiplier that scales a unit-gross leg to `target` annual vol using
    TRAILING realised vol of the (already 1-day-lagged) leg return -> no look-ahead."""
    lr = (baseW.shift(1) * rets).sum(axis=1)
    rv = lr.rolling(lb).std() * np.sqrt(365.0)
    return (target / rv).replace([np.inf, -np.inf], np.nan).clip(upper=4.0)


def signal(panel, **params):
    iw = float(params.get("illiq_weight", 1.0))     # illiquidity book risk weight
    tw = float(params.get("trend_weight", 0.25))    # trend tail-overlay risk weight
    tlb = int(params.get("trend_lb", 90))           # canonical TSMOM lookback
    ilb = int(params.get("illiq_lb", 30))           # illiquidity lookback
    tert = float(params.get("tertile", 1.0 / 3.0))  # cross-sectional tertile cut
    tvol = float(params.get("target_vol", 0.10))    # equal per-leg target vol
    cbps = float(params.get("cost_bps", 20.0))      # crypto perp taker round-trip

    px = panel.astype(float)
    rets = px.pct_change()

    # ---- ILLIQUIDITY LEG: long less-liquid / short more-liquid, dollar-neutral, inv-vol ----
    roll = _roll_illiq(rets, ilb)
    z = xs_zscore(roll)                              # winsorised cross-sectional z (kit)
    rk = z.rank(axis=1, pct=True)                    # per-date percentile for tertile cut
    long_m = rk >= (1.0 - tert)                      # less liquid -> long
    short_m = rk <= tert                             # more liquid -> short
    vol = rets.rolling(60).std()
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    lw = iv.where(long_m, 0.0).fillna(0.0)
    sw = iv.where(short_m, 0.0).fillna(0.0)
    lw = lw.div(lw.sum(axis=1).replace(0.0, np.nan), axis=0)
    sw = sw.div(sw.sum(axis=1).replace(0.0, np.nan), axis=0)
    W_illiq_base = (0.5 * lw).subtract(0.5 * sw, fill_value=0.0).fillna(0.0)  # gross ~1, $-neutral

    # ---- TREND LEG: canonical long/short TSMOM on liquid majors (perps short freely) ----
    maj = [m for m in MAJORS if m in px.columns]
    mom = px[maj] / px[maj].shift(tlb) - 1.0
    sgn = np.sign(mom)
    volm = rets[maj].rolling(60).std()
    ivm = (1.0 / volm).replace([np.inf, -np.inf], np.nan)
    wt = sgn * ivm
    gt = wt.abs().sum(axis=1).replace(0.0, np.nan)
    W_trend_base = wt.div(gt, axis=0).fillna(0.0).reindex(columns=px.columns).fillna(0.0)  # gross ~1

    # ---- equal-vol the legs, then weight (illiquidity 100%, trend 25% tail overlay) ----
    W_il = W_illiq_base.mul(_leg_scale(W_illiq_base, rets, tvol), axis=0).fillna(0.0) * iw
    W_tr = W_trend_base.mul(_leg_scale(W_trend_base, rets, tvol), axis=0).fillna(0.0) * tw
    combined = W_il.add(W_tr, fill_value=0.0).fillna(0.0)

    # cap gross leverage <= 2x
    gross = combined.abs().sum(axis=1)
    cap = (2.0 / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    combined = combined.mul(cap, axis=0)

    # DAILY EOD rebalance (frozen design: 'daily EOD rebalance only, no intraday')
    # then LAG 1 day -> the decision-to-execution lag is OUR responsibility.
    W_lag = combined.shift(1).fillna(0.0)

    rcols = [c for c in W_lag.columns if c in rets.columns]
    W_lag, r = W_lag[rcols], rets[rcols]

    daily = net_of_cost(W_lag, r, cost_bps=cbps, name="illiq_x_trend_crypto")  # already lagged
    trades = trades_from_weights(W_lag, r, SECTOR_MAP)                          # kit stamps entry_regime
    return daily, trades


# ----------------------------- soft expectations -----------------------------
def _sharpe(s):
    s = pd.Series(s).dropna()
    sd = s.std()
    if len(s) < 30 or sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(s.mean() / sd * np.sqrt(365.0))


def _maxdd(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return 0.0
    c = (1.0 + s).cumprod()
    return float((c / c.cummax() - 1.0).min())


def _chk_corr(ctx):
    g = ctx.get("grid", {})
    a, b = g.get("illiq_only"), g.get("trend_only")
    if a is None or b is None:
        return {"pass": False, "observed": "missing_variant"}
    j = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(j) < 30:
        return {"pass": False, "observed": "insufficient_overlap"}
    c = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    return {"pass": bool(c <= 0.10), "observed": round(c, 3)}


def _chk_dd(ctx):
    comb, il = ctx.get("search"), ctx.get("grid", {}).get("illiq_only")
    if comb is None or il is None:
        return {"pass": False, "observed": "missing"}
    dc, di = _maxdd(comb), _maxdd(il)
    if di == 0:
        return {"pass": False, "observed": "no_illiq_dd"}
    ratio = abs(dc) / abs(di)
    return {"pass": bool(ratio <= 0.80), "observed": round(float(ratio), 3)}


def _chk_sharpe(ctx):
    comb, il = ctx.get("search"), ctx.get("grid", {}).get("illiq_only")
    if comb is None or il is None:
        return {"pass": False, "observed": "missing"}
    sc, si = _sharpe(comb), _sharpe(il)
    if si <= 0:
        return {"pass": False, "observed": round(si, 3)}
    ratio = sc / si
    return {"pass": bool(ratio >= 0.90), "observed": round(float(ratio), 3)}


_PRE_REG = (
    "FROZEN, single design, ported asset-class (US-equity Amihud x trend -> crypto perps). "
    "MECHANISM: a NON-carry two-premium complementary-tails book. (A) pro-cyclical cross-sectional "
    "ILLIQUIDITY premium — long the relatively-LESS-liquid / short the relatively-MORE-liquid perps "
    "WITHIN a deep, individually $5K-tradable universe (relative, not micro-cap illiquidity): paid to "
    "provide liquidity in a young/less-arbitraged corner; earns in calm, crashes when crypto liquidity "
    "evaporates. (B) defensive long/short TSMOM crisis-alpha on the liquid majors — perps let it go SHORT "
    "freely (~20bps, no borrow), so it EARNS in sustained crashes (2018 bear, May-2021, 2022 LUNA/FTX) "
    "exactly where (A) bleeds, and chops otherwise. The combination — not either standalone — is the claim "
    "(naive crypto momentum has failed; funding carry is DEAD and explicitly excluded/monitored as a null). "
    "DATA HONESTY: the sanctioned yf_panel adapter returns CLOSE only (no volume), so true Amihud (|ret|/$vol) "
    "is NOT buildable free; we substitute Roll's serial-covariance effective-spread estimator — a standard "
    "PRICE-ONLY illiquidity proxy distinct from raw volatility — and LOWER the prior to medium accordingly. "
    "RULES: universe = 5 majors + 25 alts fixed before any backtest (no return selection). illiquidity lookback 30d, "
    "tertile long/short, dollar-neutral, inverse-vol; trend = 90d TSMOM sign, inverse-vol across majors; each leg "
    "equal-vol-scaled to 10% then weighted illiquidity 100% / trend 25% (tail overlay); <=2x gross; DAILY EOD "
    "rebalance (no intraday); signals lagged 1 day; 20bps round-trip taker on turnover. SUCCESS (machine-checked vs standalone "
    "crypto-illiquidity over the identical window): leg correlation <=+0.10, combined MaxDD <=80% of standalone "
    "(>=20% reduction), Sharpe degradation <=10% — plus the harness gate stack (market-neutral MCPT null on the "
    "illiquidity panel, write-once holdout on the most-recent regime from 2024-01-01, DSR over the declared grid). "
    "DIAGNOSTICS (reported, NOT gated, prose-only because not cheaply checkable in ctx): (a) funding/basis readout "
    "confirming funding ~0/negative so we are not secretly harvesting carry — needs funding_rates(), unavailable in "
    "ctx; (b) stress-window sign decomposition over 2018/May-2021/2022 (these crashes are IN-SAMPLE by construction, "
    "so the harness regime gates cover them); (c) breadth-ablation (majors-only vs full perp set) measuring marginal "
    "breadth value in a BTC-beta-dominated tape. FALSIFICATION: if the legs co-crash (crypto is single-factor BTC-beta "
    "at this scale) the correlation/DD checks fail and it is a clean, honest negative. scope=LOCAL: tested book = paper "
    "book = deployable book are the SAME perps, so forward $0-paper validation carries full evidential weight; operator "
    "personally gates any deployment regardless of outcome."
)

SPEC = StrategySpec(
    id="illiq_x_trend_crypto_perp",
    family="crypto_two_premium",
    title="Illiquidity x Trend two-premium book — crypto-perp port (cross-sectional Roll-illiquidity long/short alts, hedged by long/short TSMOM crisis-alpha on liquid majors)",
    markets=["crypto_perp"],
    data_desc="yfinance daily CLOSE for 30 crypto majors+alts (BTC/ETH/SOL/BNB/XRP + 25 alts), 2017+. yf_panel returns CLOSE ONLY (no volume) -> Amihud not free-buildable; illiquidity uses Roll's price-only effective-spread proxy (prior lowered). $0 incremental data.",
    pre_registration=_PRE_REG,
    load_data=load_data,
    signal=signal,
    default_params={"illiq_weight": 1.0, "trend_weight": 0.25, "trend_lb": 90,
                    "illiq_lb": 30, "tertile": 1.0 / 3.0, "target_vol": 0.10, "cost_bps": 20.0},
    grid={
        "default": {},
        "illiq_only": {"trend_weight": 0.0},   # standalone illiquidity baseline (used by checks)
        "trend_only": {"illiq_weight": 0.0},   # standalone trend leg (used by corr check)
        "trend_lb_120": {"trend_lb": 120},     # honest robustness variant
        "tertile_25": {"tertile": 0.25},       # honest robustness variant
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2024-01-01",
    deploy_max_positions=24,
    expectations=[
        {"name": "legs_uncorrelated",
         "claim": "illiquidity-leg vs trend-leg net-return correlation <= +0.10 over the search window",
         "check": _chk_corr},
        {"name": "drawdown_reduced_20pct",
         "claim": "combined-book MaxDD <= 80% of standalone-illiquidity MaxDD (>=20% reduction)",
         "check": _chk_dd},
        {"name": "sharpe_degradation_le_10pct",
         "claim": "combined Sharpe >= 90% of standalone-illiquidity Sharpe (<=10% dilution)",
         "check": _chk_sharpe},
    ],
)