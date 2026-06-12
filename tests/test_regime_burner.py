"""Stage 2a regime burner (pre-reg 2026-06-12: research-wiki/methodology/prereg-regime-burner.md).
Tests Part A (calm/turbulent split semantics) and Part B (coverage floor). The rule is FROZEN —
these tests pin the implementation to the pre-registration text."""
import numpy as np
import pandas as pd
import pytest

from sdk.harness import (_regime_split, _regime_coverage, REGIME_MIN_OBS,
                         REGIME_VOL_LB, REGIME_COVERAGE_FLOOR)


def _panel(n_days=1600, n_assets=20, vol_regimes=True, seed=0):
    """Price panel with a calm first half and turbulent second half (vol_regimes=True)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    sig = np.where(np.arange(n_days) < n_days // 2, 0.005, 0.025) if vol_regimes else 0.01
    rets = rng.normal(0.0003, np.repeat(np.asarray(sig).reshape(-1, 1), n_assets, axis=1))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates,
                      columns=[f"A{i}" for i in range(n_assets)])
    return px


def test_pass_when_edge_in_both_halves():
    px = _panel()
    rng = np.random.default_rng(1)
    # strategy earns a steady positive edge in BOTH halves
    r = pd.Series(rng.normal(0.0008, 0.004, len(px)), index=px.index)
    out = _regime_split(r, px)
    assert out["evaluated"] is True
    assert out["pass"] is True
    assert out["n_calm"] >= REGIME_MIN_OBS and out["n_turbulent"] >= REGIME_MIN_OBS


def test_fail_when_edge_only_in_calm():
    px = _panel()
    # strong edge in the calm half, clear LOSSES in the turbulent half
    half = len(px) // 2
    vals = np.r_[np.full(half, 0.002), np.full(len(px) - half, -0.002)]
    r = pd.Series(vals, index=px.index) + pd.Series(
        np.random.default_rng(2).normal(0, 0.001, len(px)), index=px.index)
    out = _regime_split(r, px)
    assert out["evaluated"] is True
    assert out["pass"] is False
    assert out["sharpe_turbulent"] < 0 < out["sharpe_calm"]


def test_not_evaluated_short_history_does_not_block():
    px = _panel(n_days=REGIME_VOL_LB + 50)  # too short to label >=120 obs per half
    r = pd.Series(0.001, index=px.index)
    out = _regime_split(r, px)
    assert out["evaluated"] is False and out["pass"] is None
    assert "not_evaluated" in out["reason"]


def test_not_evaluated_without_price_panel():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=500))
    out = _regime_split(r, None)
    assert out["evaluated"] is False and "no price panel" in out["reason"]


def test_no_lookahead_labels_are_lagged():
    """Day t's label must use vol through t-1: perturbing day t's return must not change day t's label
    assignment (only later days')."""
    px = _panel()
    r = pd.Series(0.001, index=px.index)
    base = _regime_split(r, px)
    px2 = px.copy()
    px2.iloc[-1] *= 1.5  # violent move on the LAST day
    out2 = _regime_split(r, px2)
    # last-day shock cannot relabel the last day itself (shift(1)) -> counts differ by at most 0 here;
    # the split result must be identical because only day t+1 (nonexistent) would see it
    assert (base["n_calm"], base["n_turbulent"]) == (out2["n_calm"], out2["n_turbulent"])


def test_coverage_floor():
    ok = [{"entry_regime": "bull_calm"}] * 9 + [{"entry_regime": "?"}]
    bad = [{"entry_regime": "?"}] * 7 + [{"entry_regime": "bull_vol"}] * 3
    assert _regime_coverage(ok)["ok"] is True
    c = _regime_coverage(bad)
    assert c["ok"] is False and c["coverage"] == 0.3 and "NOT EVALUATED" in c["note"]
    assert _regime_coverage([])["ok"] is False
