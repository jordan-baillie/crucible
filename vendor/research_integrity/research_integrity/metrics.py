"""Shared performance metrics for the Cross-OOS validation harness (Midas #102).

Strategy-agnostic, pure functions over a per-period return/PnL Series or ndarray.
Convention matches research/perp_validation/funding_spot_carry/metrics.py:
additive equity on a 1.0 base (no compounding), Sharpe = mean/std(ddof=1)*sqrt(periods).

No I/O, no network, no randomness.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


def _arr(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        return returns.to_numpy(dtype=float)
    return np.asarray(returns, dtype=float)


def sharpe(returns, periods: int = 1) -> float:
    """Sharpe ratio. periods=1 → per-period (raw); periods=N → annualized by sqrt(N)."""
    r = _arr(returns)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods))


def annualized_sharpe(returns, periods_per_year: int = DAYS_PER_YEAR) -> float:
    return sharpe(returns, periods=periods_per_year)


def profit_factor(returns) -> float:
    """Sum of positive returns / abs(sum of negative returns)."""
    r = _arr(returns)
    if r.size == 0:
        return float("nan")
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def equity_curve(returns) -> np.ndarray:
    """Additive equity on a 1.0 base (no compounding)."""
    r = _arr(returns)
    if r.size == 0:
        return np.asarray([], dtype=float)
    return 1.0 + np.cumsum(r)


def max_drawdown(equity) -> float:
    """Max fractional drawdown of an equity curve (peak-to-trough, as a positive fraction)."""
    e = _arr(equity)
    if e.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    # Guard against non-positive peaks in additive curves
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - e) / peak, 0.0)
    return float(np.nanmax(dd)) if dd.size else float("nan")


def max_drawdown_from_returns(returns) -> float:
    return max_drawdown(equity_curve(returns))


def skewness(returns) -> float:
    """Sample skewness (Fisher, bias-corrected via population moment; normal → 0)."""
    r = _arr(returns)
    n = r.size
    if n < 3:
        return float("nan")
    m = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return float("nan")
    return float(np.mean(((r - m) / sd) ** 3))


def kurtosis(returns, excess: bool = False) -> float:
    """Sample kurtosis. excess=False → normal ≈ 3.0; excess=True → normal ≈ 0.0."""
    r = _arr(returns)
    n = r.size
    if n < 4:
        return float("nan")
    m = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return float("nan")
    k = float(np.mean(((r - m) / sd) ** 4))
    return k - 3.0 if excess else k


def summary(returns, periods_per_year: int = DAYS_PER_YEAR) -> dict:
    """Full metric bundle for a per-period return series."""
    r = _arr(returns)
    if r.size == 0:
        return {
            "n": 0, "net": 0.0, "mean": float("nan"), "sharpe_ann": float("nan"),
            "sharpe_raw": float("nan"), "profit_factor": float("nan"),
            "max_drawdown": float("nan"), "skew": float("nan"), "kurtosis": float("nan"),
        }
    return {
        "n": int(r.size),
        "net": float(r.sum()),
        "mean": float(r.mean()),
        "sharpe_ann": annualized_sharpe(r, periods_per_year),
        "sharpe_raw": sharpe(r, periods=1),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown_from_returns(r),
        "skew": skewness(r),
        "kurtosis": kurtosis(r, excess=False),
    }


__all__ = [
    "DAYS_PER_YEAR", "sharpe", "annualized_sharpe", "profit_factor", "equity_curve",
    "max_drawdown", "max_drawdown_from_returns", "skewness", "kurtosis", "summary",
]
