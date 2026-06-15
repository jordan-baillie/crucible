"""Tests for the breadth / Fundamental-Law overfit gate (prereg-breadth-overfit-gate.md).

Worked-example assertions = primary-source verification (Grinold; Clarke-de Silva-Thorley; Buckle).
The Check is annotate-only (date-gated 2026-06-29) so it must be inert today.
"""
import numpy as np
import pandas as pd
import pytest

from sdk.stats import effective_breadth, implied_ic, break_even_cost_bps


# ---------------------------------------------------------------- primary-source worked examples
def test_effective_breadth_worked_examples():
    assert effective_breadth(100, 0.2) == pytest.approx(4.808, abs=1e-2)   # saturates toward 1/rho=5
    assert effective_breadth(100, 0.0) == 100.0                            # rho=0 -> nominal
    assert effective_breadth(100, 1.0) == pytest.approx(1.0, abs=1e-2)     # rho clamped to 0.999 -> ~1 bet
    assert effective_breadth(100, -0.5) == 100.0                           # negative rho clamped to 0 (conservative)


def test_implied_ic_and_break_even_worked_examples():
    assert implied_ic(1.0, 100) == pytest.approx(0.10, abs=1e-9)           # IR/sqrt(BR)
    assert break_even_cost_bps(0.12, 6) == pytest.approx(200.0, abs=1e-6)  # 12% / 6x -> 200 bps
    assert break_even_cost_bps(0.10, 0) == float("inf")                    # no turnover -> infinite headroom


def test_implied_ic_flags_narrow_high_ir_book():
    """Discrimination: a broad daily book is plausible; a narrow correlated monthly book is not."""
    broad = implied_ic(1.5, effective_breadth(100, 0.1) * 252)
    narrow = implied_ic(1.5, effective_breadth(5, 0.3) * 12)
    assert broad < 0.10 and narrow > 0.20


# ---------------------------------------------------------------- the Check (annotate-only today)
def _ctx(n_names, rho, n_days=1200, rebal="ME", drift=0.0006, seed=0):
    from sdk.gates import GateContext
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-01", periods=n_days)
    common = rng.normal(0, 0.01, n_days)
    cols = {}
    for i in range(n_names):
        idio = rng.normal(0, 0.01, n_days)
        x = np.sqrt(rho) * common + np.sqrt(max(1e-9, 1 - rho)) * idio    # target pairwise corr ~rho
        cols[f"N{i}"] = 100 * np.exp(np.cumsum(x))
    px = pd.DataFrame(cols, index=idx)
    rib = px.pct_change().mean(axis=1) + drift                            # equal-weight book + drift
    search = rib[rib.index < "2024-01-01"]
    rebal_dates = px.resample(rebal).last().index
    trades = [{"ticker": c, "entry_date": str(d.date())}
              for d in rebal_dates if d < pd.Timestamp("2024-01-01") for c in cols]
    return GateContext(spec=None, panel=None, price_matrix=px, search=search,
                       search_trades=trades, holdout_pass=True, deploy_candidate=True)


def test_breadth_check_computes_and_is_inert_today():
    from sdk.harness import _gc_breadth_overfit
    res = _gc_breadth_overfit(_ctx(n_names=40, rho=0.15, rebal="ME", seed=1))
    assert res.name == "breadth_overfit" and res.failure_mode == 1
    if res.evaluated:
        assert "implied_ic" in res.metrics and res.metrics["n_names"] >= 2
    assert res.active is False and res.demotes is False    # date-gated 2026-06-29 -> inert TODAY


def test_breadth_check_not_evaluated_on_losing_or_thin():
    from sdk.harness import _gc_breadth_overfit
    res = _gc_breadth_overfit(_ctx(n_names=40, rho=0.15, drift=-0.001, seed=2))  # negative drift -> IR<=0
    assert res.evaluated is False and res.passed is None    # no positive edge -> nothing to flag
