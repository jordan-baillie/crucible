"""Multiple-testing / overfitting controls for the Cross-OOS harness (Midas #102).

- Probabilistic Sharpe Ratio (PSR)         — Bailey & López de Prado (2012)
- Expected maximum Sharpe under N trials    — used to deflate
- Deflated Sharpe Ratio (DSR)               — Bailey & López de Prado (2014)
- Probability of Backtest Overfitting (PBO) — CSCV, Bailey et al. (2015)

All Sharpe inputs to PSR/DSR are PER-PERIOD (raw, not annualized). Pure functions; the only
randomness is none — CSCV is exhaustive over combinations. No I/O, no network.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR(sr*) = P(true SR > sr_benchmark). sr and sr_benchmark are PER-PERIOD Sharpes.

    kurtosis is non-excess (normal = 3.0).
    """
    if n_obs < 2 or not math.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of N i.i.d. ~N(0, sr_variance) Sharpe estimates (the deflation bar).

    SR0 = sqrt(V) * [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]
    """
    if n_trials < 1 or sr_variance < 0:
        return float("nan")
    if n_trials == 1:
        return 0.0
    if sr_variance == 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR(SR0) where SR0 is the expected-max Sharpe across n_trials configurations.

    sr, the implied benchmark SR0, and sr_variance must all be in PER-PERIOD units.
    Returns P(true SR > expected best-of-N noise SR) ∈ [0, 1]; > 0.95 is the usual bar.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, sr_benchmark=sr0, skew=skew, kurtosis=kurtosis)


def _col_sharpe(mat: np.ndarray) -> np.ndarray:
    """Per-column per-period Sharpe of an (obs × config) matrix (ddof=1)."""
    mean = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, np.nan)
    return sr


def effective_num_trials(perf_matrix: np.ndarray) -> float:
    """Effective number of INDEPENDENT trials via the eigenvalue participation ratio.

    Given a (T observations x N trials) performance matrix (e.g. per-config return series),
    compute N_eff = (sum lambda_i)^2 / sum(lambda_i^2) over the eigenvalues of the N x N
    correlation matrix of the trials. N_eff -> N when trials are mutually orthogonal and -> 1
    when they are perfectly correlated. This corrects the multiple-testing count used by the
    Deflated Sharpe Ratio when the trials are correlated (e.g. coordinate-descent perturbations
    of one strategy), which a raw config count badly over-states.

    Pure; no I/O. Returns NaN if fewer than 1 usable (non-degenerate) trial column.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 1:
        return float("nan")
    if M.shape[1] == 1:
        return 1.0
    sd = M.std(axis=0, ddof=1)
    M = M[:, sd > 0]                     # drop zero-variance trials
    if M.shape[1] < 2:
        return float(M.shape[1])
    C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-12]
    if w.size == 0:
        return float("nan")
    pr = float((w.sum() ** 2) / np.square(w).sum())
    return float(min(max(pr, 1.0), M.shape[1]))


def pbo_cscv(perf_matrix: np.ndarray, n_splits: int = 16, metric: str = "sharpe") -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    Parameters
    ----------
    perf_matrix : (T observations × N configs) per-observation performance (e.g. returns).
    n_splits : S, the number of disjoint row sub-matrices (must be even). Combinations
        choose S/2 for training; defaults to 16 (C(16,8)=12870 evaluations).
    metric : 'sharpe' (default) or 'mean' — the in/out-of-sample ranking statistic.

    Returns dict with: pbo, logits (list), n_combos, n_configs.
    PBO is the fraction of train/test partitions where the IS-best config ranks below the
    OOS median (logit <= 0). PBO≈0.5 ⇒ selection is noise; low PBO ⇒ genuine, robust edge.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("perf_matrix must be (T x N) with N >= 2 configs")
    if n_splits % 2 != 0 or n_splits < 2:
        raise ValueError("n_splits must be a positive even integer")
    T, N = M.shape
    if n_splits > T:
        raise ValueError("n_splits cannot exceed number of observations T")

    edges = [int(round(i * T / n_splits)) for i in range(n_splits + 1)]
    blocks = [M[edges[i]:edges[i + 1]] for i in range(n_splits)]

    def _stat(block_stack: np.ndarray) -> np.ndarray:
        if metric == "mean":
            return block_stack.mean(axis=0)
        return _col_sharpe(block_stack)

    half = n_splits // 2
    logits: list[float] = []
    for train_groups in combinations(range(n_splits), half):
        train_set = set(train_groups)
        is_stack = np.vstack([blocks[g] for g in train_groups])
        oos_stack = np.vstack([blocks[g] for g in range(n_splits) if g not in train_set])

        is_perf = _stat(is_stack)
        oos_perf = _stat(oos_stack)
        if not np.any(np.isfinite(is_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))
        # OOS relative rank of the IS-best config (1=worst .. N=best), then to (0,1)
        finite = np.isfinite(oos_perf)
        # rank among finite configs
        order = np.argsort(np.where(finite, oos_perf, -np.inf))
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        omega = ranks[n_star] / (N + 1.0)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(float(math.log(omega / (1.0 - omega))))

    if not logits:
        return {"pbo": float("nan"), "logits": [], "n_combos": 0, "n_configs": N}
    arr = np.asarray(logits)
    pbo = float(np.mean(arr <= 0.0))
    return {"pbo": pbo, "logits": logits, "n_combos": len(logits), "n_configs": N}


__all__ = [
    "EULER_MASCHERONI", "probabilistic_sharpe_ratio", "expected_max_sharpe",
    "deflated_sharpe_ratio", "effective_num_trials", "pbo_cscv",
]
