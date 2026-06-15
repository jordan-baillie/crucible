"""Tests for the uniform gate-check contract (sdk/gates.py; design-gate-system-unification.md).

Focus: the SINGLE-SOURCED demotion logic + phase-in date-gate — the part that replaced five copies
of the inline if/append idiom (where the (p or 1.0) footgun was born). Per-check behaviour is guarded
end-to-end by tests/test_harness_integration.py + the gate-diff harness.
"""
import pandas as pd
import pytest

from sdk.gates import CheckResult, apply_demotions, gates_report, run_checks, GateContext

FUTURE = "2099-01-01"
PAST = "2000-01-01"


def _r(name, passed, reason=None, active_from=None, fm=5, evaluated=True):
    return CheckResult(name=name, failure_mode=fm, evaluated=evaluated, passed=passed,
                       reason=reason, active_from=active_from)


# ---------------------------------------------------------------- active / demotes semantics
def test_active_and_demotes_semantics():
    assert _r("a", False).demotes is True                      # always-active hard fail demotes
    assert _r("a", True).demotes is False                      # pass never demotes
    assert _r("a", None).demotes is False                      # not-evaluated never demotes
    assert _r("a", False, active_from=FUTURE).active is False  # future date-gate -> inactive
    assert _r("a", False, active_from=FUTURE).demotes is False # ...so it does NOT demote (annotate-only)
    assert _r("a", False, active_from=PAST).demotes is True    # past date-gate -> active -> demotes


def test_macro_style_phase_in_is_inert_today():
    """A check date-gated to the future (like macro's 2026-06-29) must not demote today."""
    assert _r("macro_confound", False, reason="x", active_from="2026-06-29").demotes is False


# ---------------------------------------------------------------- single-sourced demotion
def test_first_failing_check_wins_and_short_circuits():
    """Matches the legacy `if stage1_pass and ...` short-circuit: only the FIRST active hard-fail's
    reason is recorded; later failing checks add nothing."""
    results = [_r("beta", True), _r("regime", False, "REGIME-FRAGILE: ..."),
               _r("deploy", False, "DEPLOY: ...")]
    sp, reasons = apply_demotions(True, ["holdout note"], results)
    assert sp is False
    assert reasons == ["holdout note", "REGIME-FRAGILE: ..."]   # deploy reason NOT appended


def test_pass_through_when_all_ok():
    results = [_r("beta", True), _r("regime", None), _r("deploy", True)]
    sp, reasons = apply_demotions(True, ["x"], results)
    assert sp is True and reasons == ["x"]


def test_multi_reason_list_is_extended():
    """Deployability returns a LIST of reasons; apply_demotions must extend, not nest."""
    results = [_r("deploy", False, ["borrow infeasible", "liquidity cost"])]
    sp, reasons = apply_demotions(True, [], results)
    assert sp is False and reasons == ["borrow infeasible", "liquidity cost"]


def test_inactive_failing_check_does_not_demote_but_later_active_one_does():
    results = [_r("macro", False, "MACRO", active_from=FUTURE),   # inactive -> skipped
               _r("deploy", False, "DEPLOY")]                      # active -> demotes
    sp, reasons = apply_demotions(True, [], results)
    assert sp is False and reasons == ["DEPLOY"]


def test_already_failed_stage1_is_unchanged():
    """If stage1 already False (e.g. tier!=PROMOTE), demotion adds no reasons."""
    sp, reasons = apply_demotions(False, ["tier FAIL"], [_r("beta", False, "BETA")])
    assert sp is False and reasons == ["tier FAIL"]


# ---------------------------------------------------------------- report shape
def test_gates_report_shape():
    results = [_r("beta", True, fm=5), _r("regime", None, fm=4, evaluated=False)]
    results[0].metrics = {"beta_to_universe": 0.9}
    rep = gates_report(results)
    assert set(rep) == {"beta", "regime"}
    assert rep["beta"]["failure_mode"] == 5 and rep["beta"]["passed"] is True
    assert rep["beta"]["beta_to_universe"] == 0.9            # metrics flattened in
    assert rep["regime"]["evaluated"] is False and rep["regime"]["active"] is True


def test_run_checks_executes_in_order():
    calls = []
    def mk(name):
        def c(ctx):
            calls.append(name)
            return _r(name, True)
        return c
    ctx = GateContext(spec=None, panel=None, price_matrix=None, search=None,
                      search_trades=[], holdout_pass=True, deploy_candidate=True)
    run_checks([mk("a"), mk("b"), mk("c")], ctx)
    assert calls == ["a", "b", "c"]
