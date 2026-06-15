"""sdk/cost_model.py — liquidity-and-borrow-aware deployability cost model.

Single source of truth for the FROZEN dollar-volume cost ladder, borrow-feasibility
(Alpaca shortable set), and a drop-in `net_of_cost` replacement that re-prices a weight
matrix under realistic per-name cost AND zeroes borrow-infeasible shorts.

Used by BOTH the live deployability gate (research_integrity) and the 96-corpus re-score
(forward/cost_rescore.py). Spec: research-wiki/methodology/prereg-cost-aware-deployability-gate.md
(FROZEN 2026-06-15). DO NOT tune ladder levels to make any strategy survive (First Principle).
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd

import crucible_paths as P

# --------------------------------------------------------------------------- FROZEN ladder
# Round-trip cost in bps by dollar-volume DECILE *within a strategy's traded universe*
# (decile 10 = most liquid / cheapest, decile 1 = most illiquid / dearest). Set from
# microstructure priors, NOT fit to our (contaminated) live fills. See pre-reg §2.
LADDER_CENTRAL = {10: 5, 9: 7, 8: 10, 7: 14, 6: 20, 5: 28, 4: 40, 3: 55, 2: 75, 1: 100}
LADDER_CONSERVATIVE = {10: 8, 9: 11, 8: 16, 7: 22, 6: 32, 5: 45, 4: 64, 3: 88, 2: 120, 1: 160}

# pre-reg §1: a short book with > this share of position-days in non-shortable names FAILS deployability.
BORROW_INFEASIBLE_CAP = 0.20

# CRYPTO deployability (2026-06-15, operator re-point to crypto): perps short FREELY (no stock
# borrow), so the equity borrow filter + DV ladder do NOT apply. Realistic round-trip taker cost on
# the liquid majors is ~18-20bps (Binance 4.5bps/leg x spot+perp, + small slippage on deep books).
# net_of_cost charges per turnover-unit (one-way per trade), so the per-trade figure is ~10bps.
# Frozen conservative default; alt-coin illiquidity is handled by biasing the forge to liquid majors.
CRYPTO_COST_BPS = 10.0  # one-way per-trade (~20bps round-trip); majors


def is_crypto(markets) -> bool:
    """True if a StrategySpec.markets list designates a crypto book (perps short freely; the equity
    borrow + DV-ladder are the wrong model). Matches 'crypto' case-insensitively in any market tag."""
    try:
        return any("crypto" in str(m).lower() for m in (markets or []))
    except Exception:
        return False


def make_crypto_net_of_cost(cost_bps: float = CRYPTO_COST_BPS, record: dict | None = None):
    """DROP-IN net_of_cost for crypto: flat per-trade taker cost on turnover, NO borrow zeroing
    (perps short freely). Ignores the passed cost_bps so a too-cheap strategy assumption is replaced
    by the realistic taker cost."""
    def _net(W: pd.DataFrame, rets: pd.DataFrame, cost_bps_ignored: float = 8.0, name: str = "strategy") -> pd.Series:
        gross = (W * rets).sum(axis=1)
        cost_day = (W - W.shift(1)).abs().sum(axis=1) * cost_bps * 1e-4
        net = (gross - cost_day).fillna(0.0)
        net.name = name
        if record is not None:
            record["crypto_cost_bps_per_trade"] = cost_bps
            record["ann_cost_drag"] = round(float(cost_day.mean() * 365), 4)
        return net
    return _net


# --------------------------------------------------------------------------- borrow feasibility
@lru_cache(maxsize=1)
def shortable_set(path: str | None = None) -> frozenset:
    """Alpaca shortable tickers (present-day snapshot; pre-reg §1 declares the PIT caveat —
    present-day shortability is the conservative/optimistic-for-history direction)."""
    p = path or os.path.join(str(P.DATA), "cache", "alpaca_tradable_assets.json")
    try:
        d = json.load(open(p))
        return frozenset(str(t).upper() for t in d.get("shortable", []))
    except Exception:
        return frozenset()


def short_infeasible_share_from_weights(W: pd.DataFrame, shortable: frozenset) -> float:
    """Fraction of SHORT position-days held in non-shortable names (0 if no shorts)."""
    short = W < 0
    total = float(short.values.sum())
    if total <= 0:
        return 0.0
    feasible_cols = [c for c in W.columns if str(c).upper() in shortable]
    feas = short[feasible_cols].values.sum() if feasible_cols else 0.0
    return float((total - feas) / total)


def short_infeasible_share_from_trades(trades: list, shortable: frozenset) -> float:
    """Same metric from a CONTRACT trade ledger (position_value<0 == short); weights by hold_days."""
    tot = inf = 0.0
    for t in trades or []:
        if float(t.get("position_value", 0.0)) < 0:
            hd = float(t.get("hold_days", 1) or 1)
            tot += hd
            if str(t.get("ticker", "")).upper() not in shortable:
                inf += hd
    return float(inf / tot) if tot > 0 else 0.0


# --------------------------------------------------------------------------- dollar-volume map
_DV_CACHE = os.path.join(str(P.DATA), "cache", "dollar_volume_map.json")


def build_dollar_volume_map(lookback: int = 63, persist: bool = True) -> dict:
    """Build the global ticker -> trailing-median dollar volume ($) map from the SEP cache
    (closeadj*volume) and persist it. EXPENSIVE (full-universe read, multi-GB) -> call ONCE in
    the parent, never per worker (silent empty maps under a memory cap toothless the ladder)."""
    from sdk.adapters import sep_panel
    close = sep_panel(field="closeadj")
    vol = sep_panel(field="volume")
    dv = (close * vol).tail(max(lookback, 21)).median().dropna()
    m = {str(k).upper(): float(v) for k, v in dv.items() if v > 0}
    if persist and m:
        os.makedirs(os.path.dirname(_DV_CACHE), exist_ok=True)
        tmp = _DV_CACHE + ".tmp"
        json.dump(m, open(tmp, "w"))
        os.replace(tmp, _DV_CACHE)
    return m


@lru_cache(maxsize=1)
def dollar_volume_map(lookback: int = 63) -> dict:
    """Global ticker -> trailing-median dollar volume ($). Loads the persisted cache (cheap,
    worker-safe); builds + persists it only if absent. Tickers absent here (ETFs/crypto/futures)
    are treated as most-liquid at re-price time (pre-reg §2: equity micro-caps are the concern)."""
    try:
        if os.path.exists(_DV_CACHE):
            m = json.load(open(_DV_CACHE))
            if m:
                return {str(k).upper(): float(v) for k, v in m.items()}
    except Exception:
        pass
    try:
        return build_dollar_volume_map(lookback=lookback)
    except Exception:
        return {}


def ladder_cost_bps(tickers, ladder: dict, dv_map: dict | None = None) -> dict:
    """Assign each ticker a round-trip cost (bps) by its DV DECILE *within this ticker set*.
    Names with no DV (non-equity) -> most-liquid decile (flagged by caller via n_unmapped)."""
    dv_map = dv_map if dv_map is not None else dollar_volume_map()
    tickers = [str(t).upper() for t in tickers]
    dv = pd.Series({t: dv_map.get(t, np.nan) for t in tickers}, dtype=float)
    have = dv.dropna()
    out = {}
    if len(have) >= 2:
        q = have.rank(pct=True)  # lowest DV -> ~0 -> decile 1 (dearest)
        for t in have.index:
            dec = int(min(10, max(1, math.ceil(q[t] * 10))))
            out[t] = float(ladder[dec])
    elif len(have) == 1:
        out[have.index[0]] = float(ladder[5])  # single name -> mid ladder
    # unmapped (non-equity / missing) -> most liquid bucket
    for t in tickers:
        out.setdefault(t, float(ladder[10]))
    return out


# --------------------------------------------------------------------------- re-priced net_of_cost
def make_net_of_cost(ladder: dict, dv_map: dict | None = None,
                     shortable: frozenset | None = None, record: dict | None = None):
    """Return a DROP-IN replacement for sdk.signal_kit.net_of_cost(W, rets, cost_bps, name)
    that IGNORES the flat cost_bps and instead charges per-name ladder cost on turnover and
    ZEROES borrow-infeasible short weights. If `record` is given, writes diagnostics into it.
    Used by the re-score via monkeypatch; the deployability gate uses the helpers above."""
    dv_map = dv_map if dv_map is not None else dollar_volume_map()
    shortable = shortable if shortable is not None else shortable_set()

    def _net(W: pd.DataFrame, rets: pd.DataFrame, cost_bps: float = 8.0, name: str = "strategy") -> pd.Series:
        W = W.copy()
        # borrow: a short in a non-shortable name cannot be held -> zero it (remove, don't assume)
        infeasible_share = short_infeasible_share_from_weights(W, shortable)
        if (W < 0).values.any():
            non_short = [c for c in W.columns if str(c).upper() not in shortable]
            if non_short:
                block = W[non_short]
                W[non_short] = block.where(block >= 0, 0.0)
        cost_vec = pd.Series(ladder_cost_bps(list(W.columns), ladder, dv_map), dtype=float).reindex(W.columns)
        gross = (W * rets).sum(axis=1)
        dW = (W - W.shift(1)).abs()
        cost_day = (dW.mul(cost_vec, axis=1) * 1e-4).sum(axis=1)
        net = (gross - cost_day).fillna(0.0)
        net.name = name
        if record is not None:
            record["short_infeasible_share"] = round(infeasible_share, 4)
            record["mean_name_cost_bps"] = round(float(cost_vec.mean()), 2)
            record["n_names"] = int(len(cost_vec))
            record["n_unmapped"] = int(sum(1 for t in W.columns if str(t).upper() not in dv_map))
            record["ann_cost_drag"] = round(float(cost_day.mean() * 252), 4)
        return net

    return _net


def borrow_verdict(trades: list, shortable: frozenset | None = None) -> dict:
    """Deployability borrow check for the live gate (operates on a trade ledger).
    FAIL when > BORROW_INFEASIBLE_CAP of short position-days are in non-shortable names."""
    shortable = shortable if shortable is not None else shortable_set()
    share = short_infeasible_share_from_trades(trades, shortable)
    ok = share <= BORROW_INFEASIBLE_CAP
    return {
        "borrow_feasible": bool(ok),
        "short_infeasible_share": round(share, 4),
        "cap": BORROW_INFEASIBLE_CAP,
        "reason": None if ok else (
            f"short_infeasible_share {share:.2f} > {BORROW_INFEASIBLE_CAP} — "
            f"edge depends on un-borrowable shorts (un-deployable)"),
    }
