"""
Dividend-Month Premium (Hartzmark & Solomon, JFE 2013) — mid-cap, long-only, monthly.

Mechanism: liquidity-provision against PREDICTABLE dividend-seeking flow. A stock is a
"predicted payer" for month M iff it had an ex-dividend event in calendar month M-12 —
strictly point-in-time (only events >= 11 months old are ever consulted).

Ex-div inference (the 2026-06-10 runtime_error fix): SEP cache v2 carries the
div-UNADJUSTED 'close' alongside 'closeadj'. The dividend adjustment factor
(closeadj/close) steps UP on each ex-div day; those steps are historical facts
independent of snapshot date, so detection is lookahead-safe. Fallback to 'closeunadj'
with a split guard (events capped at 25% to exclude split artifacts).

Lag discipline: weights are formed at month-end from trailing data only and passed to
net_of_cost / trades_from_weights as W.shift(1) — returns accrue from the NEXT day.
"""

import numpy as np
import pandas as pd

from sdk.harness import StrategySpec
from sdk.adapters import sep_panel
from sdk.universe import sector_universe
from sdk.signal_kit import net_of_cost, trades_from_weights

START = "2008-01-01"


def _resample_me(obj):
    """Month-end resample, robust across pandas versions ('ME' vs 'M')."""
    try:
        return obj.resample("ME")
    except ValueError:
        return obj.resample("M")


def _mid_search_universe():
    # Deterministic — also used by load_gen_data('mid_rest') to guarantee disjointness.
    return sector_universe(marketcap="Mid", top_n_per_sector=30)


def _panel(tickers, sector_map):
    """MultiIndex-column panel: ('closeadj', t) and ('raw', t) = div-unadjusted price."""
    adj = sep_panel(tickers, START, field="closeadj")
    try:
        raw = sep_panel(tickers, START, field="close")       # split-adj, div-UNadjusted
    except Exception:
        raw = sep_panel(tickers, START, field="closeunadj")  # fallback (split guard below)
    cols = adj.columns.intersection(raw.columns)
    panel = pd.concat({"closeadj": adj[cols], "raw": raw[cols]}, axis=1)
    panel.attrs["sector_map"] = {t: sector_map.get(t, "Unknown") for t in cols}
    return panel


def load_data():
    tickers, smap = _mid_search_universe()          # ~330 mid-caps, sector-spread
    return _panel(tickers, smap)


def load_gen_data(label):
    """Disjoint confirmation universes (different cap tiers / untouched mid names)."""
    if label == "large":
        tickers, smap = sector_universe(marketcap="Large", top_n_per_sector=30)
    elif label == "small":
        tickers, smap = sector_universe(marketcap="Small", top_n_per_sector=35)
    elif label == "mid_rest":
        wide, smap = sector_universe(marketcap="Mid", top_n_per_sector=62)
        search, _ = _mid_search_universe()
        searched = set(search)
        tickers = [t for t in wide if t not in searched][:350]   # shares NO tickers w/ search
    else:
        raise ValueError(f"unknown gen universe: {label}")
    return _panel(tickers, smap)


def signal(panel, **params):
    p = dict(
        div_eps=0.001,    # min per-event implied yield (filters float noise)
        div_cap=0.25,     # max per-event yield (excludes split artifacts on fallback path)
        vol_lb=63,        # trailing vol lookback for inverse-vol sizing
        min_names=10,     # breadth floor — below this, stay in cash
        max_w=0.05,       # per-name cap
        inv_vol=True,
        hedge_beta=False, # grid variant: overlay short of the universe EW book (rolling beta)
        cost_bps=15.0,    # mid-cap cost assumption (registered; stricter than the 8bp default)
    )
    p.update(params)

    adj = panel["closeadj"]
    raw = panel["raw"]
    rets = adj.pct_change()

    # ---- ex-dividend day inference: dividend adjustment factor steps UP on ex-date ----
    ratio = adj / raw
    chg = ratio.pct_change()
    exdiv = (chg > p["div_eps"]) & (chg < p["div_cap"])

    # ---- monthly aggregation (rows stamped at calendar month-end) ----
    ex_m = _resample_me(exdiv.astype(float)).max()
    vol_m = _resample_me(rets.rolling(p["vol_lb"]).std()).last()
    px_m = _resample_me(raw).last()

    # Row at month-end of month X is HELD during month X+1.
    # Predicted payer in X+1  <=>  ex-div in month (X+1)-12 = X-11  <=>  ex_m.shift(11) at row X.
    # => only events >= ~11 months old are used: point-in-time safe by construction.
    pred = (ex_m.shift(11) > 0) & vol_m.notna() & (vol_m > 0) & px_m.notna()

    # ---- inverse-vol long book, capped, breadth-gated ----
    if p["inv_vol"]:
        w = (1.0 / vol_m).where(pred, 0.0)
    else:
        w = pred.astype(float)
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w.loc[(w > 0).sum(axis=1) < p["min_names"]] = 0.0
    gross = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0).clip(upper=p["max_w"])
    gross = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)

    # ---- monthly weights -> daily, then LAG (the shift(1) is the lookahead guard) ----
    W = w.reindex(rets.index.union(w.index)).ffill().reindex(rets.index).fillna(0.0)
    W_lag = W.shift(1).fillna(0.0)

    daily = net_of_cost(W_lag, rets, cost_bps=p["cost_bps"], name="div_month_premium_mid")

    if p["hedge_beta"]:
        # Return overlay: short the universe EW book at trailing beta (beta lagged 1d).
        ew = rets.mean(axis=1)
        book = (W_lag * rets).sum(axis=1)
        beta = (book.rolling(252).cov(ew) / ew.rolling(252).var()).shift(1)
        beta = beta.clip(0.0, 2.0).fillna(0.0)
        hedge_cost = beta.diff().abs().fillna(0.0) * (p["cost_bps"] / 1e4)
        daily = (daily - beta * ew - hedge_cost).rename("div_month_premium_mid_hedged")

    trades = trades_from_weights(W_lag, rets, panel.attrs.get("sector_map", {}))
    return daily, trades


SPEC = StrategySpec(
    id="dividend_month_premium_mid_v1",
    family="event_flow_div_month",
    title="Dividend-Month Premium: predicted ex-div mid-caps, long-only monthly book "
          "(Hartzmark-Solomon 2013; price-pressure / flow premium)",
    markets=["US_equities_mid"],
    data_desc="Sharadar SEP cache v2 (closeadj + div-unadjusted close), survivorship-clean "
              "mid-cap sector-spread universe (~330 names). Ex-div events inferred from steps "
              "in the dividend adjustment factor; predictor consults only events >=11 months old.",
    pre_registration=(
        "FROZEN CONSTRUCTION: at each month-end, LONG the inverse-vol-weighted (5% cap, breadth "
        "floor 10) book of mid-caps with an ex-dividend event in the same calendar month 12 months "
        "ago. Monthly rebalance, 15bps costs on turnover, weights shift(1)-lagged. PREDICTION: "
        "positive BETA-ADJUSTED SELECTION ALPHA vs the equal-weight mid universe — the primary "
        "registered statistic (long-only book, beta confound MUST be isolated first per the "
        "2026-06-10 MCPT law; raw Sharpe is NOT the claim). Mechanism is liquidity provision "
        "against predictable dividend-seeking flow around a NON-information event — distinct from "
        "PEAD/earnings-announcement (information events) and all closed factor families. "
        "GENERALIZATION (broad): same-SIGNED selection alpha on disjoint large-cap, small-cap, and "
        "untouched-mid universes; large-cap expected weaker (2026-06-09 liquidity lesson — bar is "
        "same-sign, not significance)."
    ),
    load_data=load_data,
    signal=signal,
    default_params={},
    grid={
        "default": {},
        "beta_hedged": {"hedge_beta": True},
        "equal_weight": {"inv_vol": False},
        "tight_cap": {"max_w": 0.03},
    },
    scope="broad",
    generalization_universes=["large", "small", "mid_rest"],
    load_gen_data=load_gen_data,
    holdout_start="2022-01-01",
    deploy_max_positions=60,
)