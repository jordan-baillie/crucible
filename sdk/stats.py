"""sdk/stats.py — THE canonical stats helpers. One definition, everywhere.

History: `sharpe` was independently re-defined in EIGHT files with subtly different
rounding (none/2dp/3dp) and min-length rules (none/len>20) — divergent stats helpers
in a statistics shop is how inconsistent verdicts happen. The harness's definition is
canonical (no min-length, no rounding — presentation belongs at the call site).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(r, ann: int = 252) -> float:
    """Annualized Sharpe. CANONICAL (harness definition): no min-length gate, no rounding.
    Insufficient/degenerate data -> 0.0."""
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def sharpe_or_none(r, ann: int = 252, min_obs: int = 20, ndigits: int | None = 2):
    """Battery/report variant: None when there's too little data to mean anything
    (so tables show a hole, not a fake 0), optional rounding for display."""
    r = pd.Series(r).dropna()
    if len(r) <= min_obs or r.std() == 0:
        return None
    s = float(r.mean() / r.std() * np.sqrt(ann))
    return round(s, ndigits) if ndigits is not None else s


# --- Sharpe-ratio inference (prereg-sharpe-inference-gate.md). Verified against worked examples:
#     Lo (2002) eta; Bailey & Lopez de Prado (2012/14) PSR + MinTRL. See tests/test_sharpe_inference.py.
import math as _math


def _autocorr(r, k: int) -> float:
    """Sample autocorrelation at lag k: sum (x_t-xbar)(x_{t-k}-xbar) / sum (x_t-xbar)^2."""
    x = pd.Series(r).dropna().values
    n = len(x)
    if n <= k + 1:
        return 0.0
    xm = x.mean()
    d = float(((x - xm) ** 2).sum())
    return float(((x[k:] - xm) * (x[:-k] - xm)).sum() / d) if d > 0 else 0.0


def lo_eta(rhos, q: int) -> float:
    """Lo (2002) annualization factor: eta(q) = q / sqrt(q + 2*sum_{k=1}^{q-1}(q-k)*rho_k).
    `rhos` is rho_1.. (lags beyond its length treated as 0). IID (all rho=0) -> eta = sqrt(q)."""
    s = sum((q - k) * rhos[k - 1] for k in range(1, q) if k - 1 < len(rhos))
    denom = max(q + 2.0 * s, q * 0.01)   # frozen clamp: guard sqrt under strong negative autocorrelation
    return q / _math.sqrt(denom)


def lo_deflation_factor(r, q: int = 252, max_lag: int = 5) -> float:
    """Lo-adjusted / naive annualized-Sharpe ratio = eta(q)/sqrt(q). <1 => positive serial correlation
    inflated the naive sqrt(q) annualization (stale/smoothed marks). Autocorr truncated at max_lag."""
    rhos = [_autocorr(r, k) for k in range(1, max_lag + 1)]
    return lo_eta(rhos, q) / _math.sqrt(q)


def lo_adjusted_sharpe(r, ann: int = 252, max_lag: int = 5) -> float:
    """Annualized Sharpe with Lo's serial-correlation correction."""
    return sharpe(r, ann) * lo_deflation_factor(r, q=ann, max_lag=max_lag)


def psr_from_stats(sr_hat: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio from summary stats. sr_hat: PER-PERIOD (non-annualized) Sharpe;
    kurt: RAW (non-excess; Gaussian=3); uses n-1. PSR = P(true SR > sr_star)."""
    if n < 2:
        return float("nan")
    from scipy import stats as _ss
    denom = _math.sqrt(max(1e-12, 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr_star) * _math.sqrt(n - 1) / denom
    return float(_ss.norm.cdf(z))


def probabilistic_sharpe_ratio(r, sr_star: float = 0.0) -> float:
    x = pd.Series(r).dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    from scipy import stats as _ss
    sr = float(x.mean() / x.std())                          # per-period (non-annualized)
    return psr_from_stats(sr, len(x), float(_ss.skew(x)), float(_ss.kurtosis(x, fisher=False)), sr_star)


def mintrl_from_stats(sr_hat: float, skew: float, kurt: float,
                      sr_star: float = 0.0, confidence: float = 0.95) -> float:
    """Minimum Track Record Length: obs needed for the Sharpe to be significant at `confidence`.
    inf when sr_hat <= sr_star (target unreachable). kurt RAW (Gaussian=3)."""
    if sr_hat <= sr_star:
        return float("inf")
    from scipy import stats as _ss
    z = float(_ss.norm.ppf(confidence))
    factor = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    return 1.0 + factor * (z / (sr_hat - sr_star)) ** 2


def min_track_record_length(r, sr_star: float = 0.0, confidence: float = 0.95) -> float:
    x = pd.Series(r).dropna()
    if len(x) < 2 or x.std() == 0:
        return float("inf")
    from scipy import stats as _ss
    sr = float(x.mean() / x.std())
    return mintrl_from_stats(sr, float(_ss.skew(x)), float(_ss.kurtosis(x, fisher=False)), sr_star, confidence)


def maxdd(r) -> float:
    """Max drawdown of a daily-returns series (negative number, e.g. -0.23)."""
    eq = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def split_holdout(r, holdout_start: str):
    """(search, holdout) split of a DatetimeIndex returns series at the quarantine date."""
    r = pd.Series(r).dropna()
    return r[r.index < holdout_start], r[r.index >= holdout_start]
