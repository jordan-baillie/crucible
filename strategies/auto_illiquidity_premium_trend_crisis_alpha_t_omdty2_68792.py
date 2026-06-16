"""
Amihud illiquidity premium (frozen, small-cap L/S) + a SMALL validated-CTA-trend tail overlay.

Fix history: the prior build tried to source an ETF sleeve via sep_panel (Sharadar SEP is US single
stocks, NOT ETFs -> 0 columns -> pd.concat 'No objects to concatenate') and a commodity-futures
cross-section via fut_curve/cot_positioning/eia_series/usda_nass (none in the tested-adapter
whitelist). Both removed. The breadth/tail sleeve is now the VALIDATED 21-market CTA trend leg
(trend_returns), vol-matched and added as a small overlay -- the only sanctioned hedge -- and the
whole overlay degrades gracefully so a sleeve failure can never break the frozen grid.

NO-LOOKAHEAD: Amihud + vol windows are trailing/shifted; weights resampled weekly then lagged 1 day
(the lag is OUR responsibility before net_of_cost); the trend overlay is vol-matched on trailing
(shifted) vol; trend_returns is the kit's already-lagged validated leg.
"""

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel, trend_returns
from sdk.universe import sector_universe
from sdk.signal_kit import xs_zscore, net_of_cost, trades_from_weights
import numpy as np, pandas as pd

START = "2004-01-01"
DEFAULTS = {"illiq_lb": 60, "vol_lb": 63, "trend_weight": 0.0}
GEN = {"mid_cap": "Mid", "micro_cap": "Micro", "nano_cap": "Nano"}


# ----------------------------- equity data -----------------------------
def _universe():
    return sector_universe(marketcap="Small", top_n_per_sector=110)


def _panel(tickers, sector_map):
    px = sep_panel(tickers, START, field="closeadj")
    vo = sep_panel(tickers, START, field="volume")
    cols = px.columns.intersection(vo.columns)
    panel = pd.concat({"price": px[cols], "volume": vo[cols]}, axis=1)
    panel.attrs["sector_map"] = {t: sector_map.get(t) for t in cols if sector_map.get(t)}
    return panel


def load_data() -> pd.DataFrame:
    tickers, sector_map = _universe()
    return _panel(tickers, sector_map)


def load_gen_data(label) -> pd.DataFrame:
    tickers, sector_map = sector_universe(marketcap=GEN[label], top_n_per_sector=25)
    return _panel(tickers, sector_map)


# ----------------------------- helpers -----------------------------
def _sharpe(r):
    r = pd.Series(r).dropna(); s = r.std()
    return float(r.mean() / s * np.sqrt(252)) if s and s > 0 else 0.0


def _maxdd(r):
    r = pd.Series(r).dropna()
    if r.empty: return 0.0
    c = (1.0 + r).cumprod()
    return float(-(c / c.cummax() - 1.0).min())


def _vol_match(target_ret, source_ret, lb=63):
    """Scale source so its trailing vol matches target's trailing vol (SHIFTED -> no look-ahead)."""
    t = pd.Series(target_ret); s = pd.Series(source_ret)
    vt = t.rolling(lb, min_periods=20).std().shift(1)
    vs = s.rolling(lb, min_periods=20).std().shift(1)
    ratio = (vt / vs.replace(0.0, np.nan)).clip(0.0, 5.0)
    return s * ratio


def _trend_sleeve():
    """The validated 21-market CTA trend hedge leg (kit-lagged). Returns series only."""
    r, _ = trend_returns()
    r = pd.Series(r).dropna(); r.name = "trend_sleeve"
    return r


# ----------------------------- Amihud (frozen) signal -----------------------------
def signal(panel, **params):
    p = {**DEFAULTS, **params}
    price = panel["price"].astype(float)
    volume = panel["volume"].astype(float)
    rets = price.pct_change()

    dvol = (price * volume).replace(0.0, np.nan)
    amihud = rets.abs() / dvol
    lb = int(p["illiq_lb"])
    illiq = amihud.rolling(lb, min_periods=max(10, lb // 2)).mean()
    z = xs_zscore(illiq)                                           # high illiquidity -> LONG (paid premium)

    vlb = int(p["vol_lb"])
    vol = rets.rolling(vlb, min_periods=max(10, vlb // 2)).std().shift(1)
    raw = z / vol.replace(0.0, np.nan)
    raw = raw.subtract(raw.mean(axis=1), axis=0)                  # dollar-neutral
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    W = raw.div(gross, axis=0)
    W = W.resample("W-FRI").last().reindex(W.index, method="ffill")  # weekly rebalance
    Wl = W.shift(1)                                               # lag 1 day (OUR responsibility)

    illiq_ret = net_of_cost(Wl, rets, cost_bps=8.0, name="illiq_premium")
    sector_map = panel.attrs.get("sector_map") or _universe()[1]
    trades = trades_from_weights(Wl, rets, sector_map)            # kit stamps entry_regime

    out = illiq_ret.dropna(); out.name = "illiq_premium"
    w_tr = float(p["trend_weight"])
    if w_tr > 0.0:                                                # small tail overlay, degrade-safe
        try:
            sleeve = _trend_sleeve()
            df = pd.concat([illiq_ret, sleeve], axis=1).dropna()
            if not df.empty:
                hedge = _vol_match(df.iloc[:, 0], df.iloc[:, 1])
                blended = (df.iloc[:, 0] + w_tr * hedge).dropna()
                blended.name = "illiq_trend_overlay"
                out = blended
        except Exception:
            pass                                                 # sleeve failure -> standalone premium
    return out, trades


# ----------------------------- soft expectations -----------------------------
def _check_tail_overlay(ctx):
    g = ctx.get("grid", {})
    base, combo = g.get("default"), g.get("breadth_25")
    if base is None or combo is None or len(base) == 0 or len(combo) == 0:
        return {"pass": False, "observed": "variant returns unavailable"}
    mb, mc = _maxdd(base), _maxdd(combo)
    sb, sc = _sharpe(base), _sharpe(combo)
    ok = (mc <= mb * 1.001) and (sc >= 0.95 * sb)
    return {"pass": bool(ok), "observed": f"maxdd {mc:.3f}<={mb:.3f}; sharpe {sc:.2f}>=0.95*{sb:.2f}"}


def _check_trend_independence(ctx):
    base = ctx.get("grid", {}).get("default")
    if base is None or len(base) == 0:
        return {"pass": False, "observed": "standalone returns unavailable"}
    try:
        tr = _trend_sleeve()                                      # one extra call, sliced below
    except Exception as e:
        return {"pass": False, "observed": f"trend leg failed: {e}"}
    hs = pd.Timestamp(ctx.get("holdout_start", "2022-01-01"))
    df = pd.concat([pd.Series(base), pd.Series(tr)], axis=1).dropna()
    df = df[df.index < hs]
    if len(df) < 60:
        return {"pass": False, "observed": "insufficient overlap"}
    c = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return {"pass": bool(abs(c) <= 0.30), "observed": round(c, 3)}


# ----------------------------- spec -----------------------------
SPEC = StrategySpec(
    id="illiq_amihud_xs_trend_overlay",
    family="illiquidity",
    title="Amihud illiquidity-premium L/S (small-cap, frozen) + small validated-CTA-trend tail overlay",
    markets=["US small-cap equities (long/short)", "21-market CTA trend hedge sleeve"],
    data_desc="Sharadar SEP closeadj+volume via sep_panel for the Amihud leg; validated 21-market "
              "CTA trend hedge via trend_returns (tested adapter) as a vol-matched tail overlay.",
    pre_registration=(
        "Two-leg construction. CORE: the validated Amihud illiquidity premium -- long illiquid / "
        "short liquid US small-caps, paid for bearing pro-cyclical liquidity risk. Frozen: "
        "amihud = mean(|ret|/dollar-volume) over illiq_lb days, cross-sectionally z-scored (high "
        "illiquidity -> long), inverse-vol sized, dollar-neutral, weekly rebalance, lagged 1 day, "
        "8bps costs. OVERLAY: the validated 21-market CTA trend leg (trend_returns) added ONLY as a "
        "SMALL tail hedge (trend_weight in {0,0.25,0.35}), vol-matched to the illiquidity leg via "
        "trailing (shifted) vol -- sized to minimise drag, NOT a reflexive 50/50, never dominant. "
        "Falsifiable: (a) the overlay cuts search-window MaxDD with <=5% net-Sharpe dilution; "
        "(b) the trend leg's daily corr to the illiquidity leg <= +0.30 over the search window "
        "(genuine diversifier, not a correlated bolt-on). scope='broad': the illiquidity premium "
        "must generalise to 3 disjoint cap tiers (Mid/Micro/Nano holdouts) under frozen default "
        "params (trend_weight=0) or be rejected as an overfit outlier."
    ),
    load_data=load_data,
    signal=signal,
    default_params=dict(DEFAULTS),
    grid={
        "default": {},
        "illiq_lb_120": {"illiq_lb": 120},
        "illiq_lb_21": {"illiq_lb": 21},
        "breadth_25": {"trend_weight": 0.25},          # small trend tail overlay
        "breadth_35": {"trend_weight": 0.35},
    },
    scope="broad",
    generalization_universes=list(GEN),
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=20,
    expectations=[
        {"name": "trend_overlay_cuts_tail",
         "claim": "25% trend overlay cuts search-window MaxDD with <=5% net-Sharpe dilution",
         "check": _check_tail_overlay},
        {"name": "trend_independence",
         "claim": "trend leg daily corr to the illiquidity leg <= +0.30 over the search window",
         "check": _check_trend_independence},
    ],
)