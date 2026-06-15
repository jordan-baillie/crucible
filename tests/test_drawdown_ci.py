"""Tests for the stationary-bootstrap drawdown/Calmar CI diagnostic (finding #50).

The block-length worked example IS the primary-source verification (Politis-White 2004). The CI is a
DIAGNOSTIC (never gates) — Sharpe-CI (redundant w/ Lo+DSR+CPCV) and Bai-Perron (underpowered on daily
returns) are deliberately NOT built.
"""
import numpy as np
import pandas as pd
import pytest

from sdk.stats import politis_white_block_length, bootstrap_drawdown_ci


def _ar1(n, phi, seed, drift=0.0004):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 0.01)
    return pd.Series(x + drift, index=pd.bdate_range(end="2026-06-01", periods=n))


def test_politis_white_block_length_worked_examples():
    rng = np.random.default_rng(0); n = 2520
    e = rng.normal(0, 1, n)
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.1 * ar1[t - 1] + e[t]
    assert politis_white_block_length(ar1) == pytest.approx(4.7, abs=1.5)   # worked example ~4.7
    assert politis_white_block_length(rng.normal(0, 1, n)) < 3.0            # white noise -> ~1
    ar5 = np.zeros(n)
    for t in range(1, n):
        ar5[t] = 0.5 * ar5[t - 1] + e[t]
    assert politis_white_block_length(ar5) > 8.0                            # strong autocorr -> longer


def test_drawdown_ci_shape_and_worst_case():
    r = _ar1(1500, phi=0.3, seed=1)
    ci = bootstrap_drawdown_ci(r, B=200)
    assert set(ci) >= {"maxdd_point", "maxdd_worst_p05", "maxdd_median", "block_length", "n_boot"}
    # worst-case (5th-pctile) drawdown is at least as severe as the in-sample point
    assert ci["maxdd_worst_p05"] <= ci["maxdd_point"] + 1e-9
    assert ci["block_length"] >= 1.0


def test_drawdown_ci_none_on_thin_data():
    assert bootstrap_drawdown_ci(pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))) is None


def test_serial_dependence_widens_drawdown_tail():
    """The point of the block bootstrap: serially-dependent returns have a WORSE bootstrap worst-case
    drawdown than an IID series of the same vol (an IID bootstrap would understate it)."""
    iid = bootstrap_drawdown_ci(_ar1(1500, phi=0.0, seed=4), B=200)
    dep = bootstrap_drawdown_ci(_ar1(1500, phi=0.5, seed=4), B=200)
    assert dep["block_length"] > iid["block_length"]      # detects the dependence
