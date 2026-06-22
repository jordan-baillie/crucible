"""
Stablecoin macro-liquidity regime gate on a crypto-majors trend book.

Premium (LOCAL, crypto-native): canonical long-or-flat TS-momentum on ~8 liquid
crypto majors, DE-GROSSED by an aggregate stablecoin (USDT+USDC) supply-flow gate.
Stablecoins are crypto's endogenous money supply; aggregate supply expansion =
capital flowing IN (risk-on), contraction = a liquidity DRAIN that trend is slow
to register. The gate cuts whole-book exposure during drains (whipsaw/exit
protection) while keeping the trend tail in inflow regimes.

FROZEN design (single pre-registered gate thresholds, no threshold grid tuning):
  - trend: each coin long when 100d trend > 0 else flat, per-leg vol-target 12% ann.
  - gate : (USDT+USDC) SplyCur 30d log-growth, z-scored on an EXPANDING (PIT) window;
           z>+0.5 -> 1.0x, |z|<=0.5 -> 0.5x, z<-0.5 -> 0.0x; +/-0.5 hysteresis band
           + 5-day min-hold to throttle turnover.
  - <=1x gross, daily rebalance, net of ~20bps round-trip taker (=10bps per side).
  - LAG: weights are built same-day then SHIFT(1) for execution -> no look-ahead.
         Gate z uses only shift(30) growth + expanding(past-only) stats -> PIT-safe.

Falsifiable claim (vs the ungated trend book, run as a free grid benchmark): the
flow gate must CUT drawdown and realized vol without gutting Sharpe.

Note: binance_klines / coinmetrics_metrics are the tested OWNED crypto adapters
documented in research-wiki/DATA_CATALOG.md (same loaders used by stableflow_liqbeta
and boreas-tsmom); we do not download raw or reinvent any data path here.
"""

from sdk.harness import StrategySpec
from sdk.adapters import binance_klines, coinmetrics_metrics
from sdk.signal_kit import net_of_cost, trades_from_weights
import numpy as np
import pandas as pd

# ---- frozen universe -------------------------------------------------------
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
          "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT"]

# crypto has no GICS sectors; map to coarse functional buckets so the ledger's
# sector-spread / single-name gates have real structure to evaluate.
SECTORS = {
    "BTCUSDT": "StoreOfValue", "ETHUSDT": "SmartContract", "BNBUSDT": "Exchange",
    "SOLUSDT": "SmartContract", "XRPUSDT": "Payments", "ADAUSDT": "SmartContract",
    "DOGEUSDT": "Meme", "LTCUSDT": "Payments",
}

DEFAULTS = dict(
    trend_lb=100,      # long-or-flat momentum lookback (days)
    vol_target=0.12,   # per-leg annualized vol target
    vol_lb=30,         # realized-vol estimation window
    max_leg=0.50,      # per-leg weight cap (before <=1x gross renorm)
    growth_lb=30,      # stablecoin supply log-growth horizon
    z_min=90,          # min obs before the expanding z is trusted
    z_hi=0.5,          # gate upper threshold (inflow -> 1.0x)
    z_lo=-0.5,         # gate lower threshold (drain  -> 0.0x)
    min_hold=5,        # gate min-hold (days) to throttle turnover
    cost_bps=10.0,     # per-side turnover cost (~20bps round-trip taker)
    gate_off=False,    # benchmark switch: ungated trend book
)


# ---- data ------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Daily close panel for the majors + aggregate stablecoin supply column."""
    px = binance_klines(MAJORS, market="perp")          # close panel, cols=tickers
    px = px.reindex(columns=[c for c in MAJORS if c in px.columns])

    sply = coinmetrics_metrics(("usdt", "usdc"), ("SplyCur",))  # daily supply
    agg = sply.apply(pd.to_numeric, errors="coerce").sum(axis=1)  # USDT+USDC total
    agg.name = "STABLE_SPLY"

    panel = px.astype(float).copy()
    panel["STABLE_SPLY"] = agg.reindex(panel.index).ffill()
    return panel.dropna(how="all")


def load_gen_data(label: str) -> pd.DataFrame:
    # scope='local': the stablecoin money-supply mechanism is crypto-endogenous and
    # has no clean cross-market analogue, so there is no generalization battery.
    raise NotImplementedError("scope='local'; validated by forward-paper, not stage-2")


# ---- gate helper -----------------------------------------------------------
def _gate_multiplier(z: pd.Series, z_hi: float, z_lo: float, min_hold: int) -> pd.Series:
    """3-level regime multiplier with hysteresis + min-hold (state-machine, past-only)."""
    vals = z.values
    out = np.empty(len(vals))
    cur = 0.5                 # neutral until the gate has a trusted reading
    last_change = -10 ** 9
    for i, zi in enumerate(vals):
        if np.isnan(zi):
            out[i] = cur
            continue
        if zi > z_hi:
            target = 1.0
        elif zi < z_lo:
            target = 0.0
        else:
            target = 0.5
        if target != cur and (i - last_change) >= min_hold:
            cur = target
            last_change = i
        out[i] = cur
    return pd.Series(out, index=z.index)


# ---- signal ----------------------------------------------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    coins = [c for c in MAJORS if c in panel.columns]
    px = panel[coins].astype(float)
    rets = px.pct_change()

    # (1) base engine: long-or-flat 100d trend, per-leg vol-target, <=1x gross
    long_flag = (px > px.shift(p["trend_lb"])).astype(float)
    ann_vol = rets.rolling(p["vol_lb"]).std() * np.sqrt(365.0)
    inv = (p["vol_target"] / ann_vol).clip(upper=p["max_leg"])
    raw = long_flag * inv / len(coins)
    gross = raw.sum(axis=1)
    scale = (1.0 / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    W = raw.mul(scale, axis=0)

    # (2) macro-liquidity gate: aggregate stablecoin supply 30d log-growth z-score
    if not p["gate_off"]:
        sply = panel["STABLE_SPLY"].astype(float).reindex(px.index).ffill()
        g = np.log(sply) - np.log(sply.shift(p["growth_lb"]))
        mu = g.expanding(min_periods=p["z_min"]).mean()       # past-only -> PIT-safe
        sd = g.expanding(min_periods=p["z_min"]).std()
        z = (g - mu) / sd.replace(0.0, np.nan)
        mult = _gate_multiplier(z, p["z_hi"], p["z_lo"], p["min_hold"])
        W = W.mul(mult, axis=0)

    W = W.fillna(0.0)

    # (3) execution lag is OURS: shift weights 1 day before P&L / costs / ledger
    Wl = W.shift(1)
    daily = net_of_cost(Wl, rets, cost_bps=p["cost_bps"], name="stableflow_gated_trend")
    trades = trades_from_weights(Wl, rets, SECTORS)
    return daily, trades


# ---- soft-expectation checks (free: use ctx['grid'] variants) --------------
def _sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(365.0))


def _maxdd(r):
    eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _chk_dd(ctx):
    g = ctx["grid"]
    d, u = _maxdd(g["default"]), _maxdd(g["ungated"])
    return {"pass": d >= u - 1e-9, "observed": round(d - u, 4)}  # gated DD shallower


def _chk_vol(ctx):
    g = ctx["grid"]
    vd = float(pd.Series(g["default"]).dropna().std())
    vu = float(pd.Series(g["ungated"]).dropna().std())
    return {"pass": vd <= vu + 1e-12, "observed": round(vd - vu, 6)}


def _chk_sharpe(ctx):
    g = ctx["grid"]
    sd, su = _sharpe(g["default"]), _sharpe(g["ungated"])
    return {"pass": sd >= 0.8 * su, "observed": round(sd - su, 3)}


SPEC = StrategySpec(
    id="stableflow_gated_trend",
    family="crypto_macro_liquidity",
    title="Stablecoin supply-flow regime gate on a crypto-majors trend book",
    markets=["crypto"],
    data_desc="binance_klines(majors, perp) close panel + coinmetrics SplyCur(usdt,usdc) aggregate supply",
    pre_registration=(
        "FROZEN. Base = equal-weight long-or-flat 100d TS-momentum on 8 crypto majors, "
        "per-leg vol-target 12% ann, <=1x gross. Gate = aggregate (USDT+USDC) SplyCur 30d "
        "log-growth z-scored on an EXPANDING window; z>+0.5->1.0x, |z|<=0.5->0.5x, z<-0.5->0.0x, "
        "with +/-0.5 hysteresis + 5d min-hold. Net of ~20bps round-trip taker (10bps/side), "
        "daily rebalance, weights SHIFT(1)-lagged (gate uses past-only growth+stats -> no look-ahead). "
        "Single pre-registered gate thresholds, NO threshold grid tuning. CLAIM (falsifiable, checked vs "
        "the ungated trend book as a benchmark grid variant): the flow gate CUTS drawdown and realized vol "
        "without gutting Sharpe, and must clear the beta-confound (timing selection-alpha) + benchmark-adjusted "
        "MCPT gates. LOCAL scope: the stablecoin money-supply mechanism is crypto-endogenous (no cross-market "
        "analogue) -> validated by forward-paper, not a stage-2 generalization battery."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},                 # primary: gated book
        "ungated": {"gate_off": True}, # benchmark: ungated trend book (drives expectations)
        "trend90": {"trend_lb": 90},   # honest search burden
        "trend120": {"trend_lb": 120},
        "growth45": {"growth_lb": 45},
    },
    scope="local",
    generalization_universes=[],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=8,
    expectations=[
        {"name": "gate_cuts_drawdown",
         "claim": "gated max-drawdown is shallower (>=) than the ungated trend book (search window)",
         "check": _chk_dd},
        {"name": "gate_cuts_vol",
         "claim": "gated daily-return vol <= ungated trend book vol (de-gross reduces risk)",
         "check": _chk_vol},
        {"name": "gate_preserves_sharpe",
         "claim": "gated Sharpe >= 0.8x ungated Sharpe (drawdown cut does not gut the edge)",
         "check": _chk_sharpe},
    ],
)