"""Tests for the Sharpe-inference gate (prereg-sharpe-inference-gate.md).

The worked-example assertions ARE the primary-source verification (Lo 2002; Bailey & Lopez de Prado):
they guard the known convention traps — raw-vs-excess kurtosis, n-vs-(n-1), per-period-vs-annualized
Sharpe. The Check is annotate-only (date-gated) so it must be inert today.
"""
import math

import numpy as np
import pandas as pd
import pytest

from sdk.stats import (lo_eta, lo_deflation_factor, lo_adjusted_sharpe, sharpe,
                       psr_from_stats, probabilistic_sharpe_ratio,
                       mintrl_from_stats, min_track_record_length)


# ---------------------------------------------------------------- primary-source worked examples
def test_lo_eta_worked_example_and_iid():
    rhos = [0.2 ** k for k in range(1, 12)]            # AR(1)-ish, rho_k = 0.2^k
    assert lo_eta(rhos, 12) == pytest.approx(2.87887, abs=1e-3)   # Lo (2002)
    assert lo_eta([0.0] * 11, 12) == pytest.approx(math.sqrt(12))  # IID special case


def test_psr_worked_example_and_gaussian_reduction():
    # Bailey & Lopez de Prado: SR_hat=0.10, n=60, skew=-0.5, RAW kurt=4, SR*=0 -> ~0.7725
    assert psr_from_stats(0.10, 60, -0.5, 4.0, 0.0) == pytest.approx(0.7725, abs=1e-3)
    # Gaussian (skew 0, raw kurt 3): denom = sqrt(1 + 0.5*SR^2) -> the classic normal form
    from scipy import stats as ss
    sr, n = 0.2, 100
    expect = float(ss.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr ** 2)))
    assert psr_from_stats(sr, n, 0.0, 3.0, 0.0) == pytest.approx(expect, abs=1e-9)


def test_mintrl_worked_example_and_unreachable():
    assert mintrl_from_stats(0.10, -0.5, 4.0, 0.0, 0.95) == pytest.approx(287.1, abs=0.5)
    assert mintrl_from_stats(0.05, 0.0, 3.0, 0.05) == float("inf")   # SR_hat <= SR* -> unreachable


# ---------------------------------------------------------------- behaviour on real-ish series
def test_white_noise_no_serial_inflation():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1500))
    assert lo_deflation_factor(r) > 0.9                  # ~IID -> deflation ~ 1
    assert lo_adjusted_sharpe(r) == pytest.approx(sharpe(r), rel=0.15)


def test_positive_autocorrelation_deflates_sharpe():
    rng = np.random.default_rng(3)
    n, phi = 1500, 0.5
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]                     # AR(1), strong positive autocorr
    r = pd.Series(x + 0.0004)                            # small positive drift
    defl = lo_deflation_factor(r)
    assert defl < 0.8                                    # serial correlation detected
    assert lo_adjusted_sharpe(r) < sharpe(r)             # deflation applied (Sharpe haircut)


def test_psr_monotonic_in_n():
    r = pd.Series(np.random.default_rng(1).normal(0.0006, 0.01, 2000))
    assert probabilistic_sharpe_ratio(r, 0.0) >= probabilistic_sharpe_ratio(r.iloc[:300], 0.0) - 1e-9
    assert math.isfinite(min_track_record_length(r, 0.0, 0.95)) or True  # finite or inf, never crash


# ---------------------------------------------------------------- the Check is inert today (date-gated)
def test_sharpe_inference_check_is_annotate_only_today():
    from sdk.harness import _gc_sharpe_inference
    from sdk.gates import GateContext
    rng = np.random.default_rng(7)
    n, phi = 1500, 0.6
    e = rng.normal(0, 0.01, n); x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    idx = pd.bdate_range(end="2026-06-01", periods=n)
    ctx = GateContext(spec=None, panel=None, price_matrix=None,
                      search=pd.Series(x + 0.0003, index=idx), search_trades=[],
                      holdout_pass=True, deploy_candidate=True)
    res = _gc_sharpe_inference(ctx)
    assert res.name == "sharpe_inference" and res.failure_mode == 3 and res.evaluated is True
    assert res.metrics["lo_deflation_factor"] < 1.0      # serial correlation recorded
    assert res.active is False and res.demotes is False   # date-gated 2026-06-29 -> inert TODAY
    # JSON-safe metrics (no nan/inf)
    assert res.metrics["min_track_record_len"] is None or isinstance(res.metrics["min_track_record_len"], float)
