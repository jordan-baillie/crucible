"""Uniform gate-check contract for the forge rails (design-gate-system-unification.md).

A demotion overlay used to be a hand-copied `if stage1_pass and <fail>: stage1_pass=False;
h_reasons += [...]` block in run_experiment — five copies, each with its own inline comparison
(the copy-paste that birthed the `(p or 1.0)` footgun). This module single-sources that pattern:

  * a Check is a callable (GateContext) -> CheckResult,
  * run_checks() executes them (compute lives in the check, run OUTSIDE the FDR lock),
  * apply_demotions() applies the demotion ONCE, in registry order (INSIDE the lock; cheap),
  * gates_report() renders one uniform verdict["gates"] block (single source of truth, fixes #33).

Phase-in is the `active_from` field: a check annotates (records metrics) but does not DEMOTE until
its date passes — so a new gate ships inert, is calibrated, then auto-activates. Pure module: imports
only pandas, so harness can import it with no cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

# The 5 orthogonal failure modes a rigorous pipeline must each control (deep research 2026-06-15;
# design-gate-system-unification.md §1). Tagging each check makes coverage legible.
FAILURE_MODES = {
    1: "selection/overfitting", 2: "multiplicity/data-snooping", 3: "metric-validity",
    4: "temporal-stability", 5: "economic-realizability",
}


@dataclass
class CheckResult:
    name: str
    failure_mode: int
    evaluated: bool                       # was the check evaluable? (coverage/obs guards)
    passed: bool | None                   # True=ok · False=hard-fail · None=not-evaluated
    reason: Any = None                    # demotion reason: str OR list[str] (appended only if it demotes)
    metrics: dict = field(default_factory=dict)
    active_from: str | None = None        # YYYY-MM-DD date-gate; None = always active

    @property
    def active(self) -> bool:
        """Does this check DEMOTE today, or only annotate (phase-in)?"""
        if self.active_from is None:
            return True
        return pd.Timestamp.now().strftime("%Y-%m-%d") >= self.active_from

    @property
    def demotes(self) -> bool:
        """Revoke a PASS iff the check is active AND hard-failed. not_evaluated (None) never demotes."""
        return bool(self.active and self.passed is False)


@dataclass
class GateContext:
    """Everything a demotion check might need, assembled ONCE per verdict."""
    spec: Any
    panel: Any
    price_matrix: Any
    search: Any                           # search-window return series (date-indexed)
    search_trades: list
    holdout_pass: bool
    deploy_candidate: bool                # cleared holdout + deployment-sanity (gates the costly re-price)


Check = Callable[[GateContext], CheckResult]


def run_checks(checks: list[Check], ctx: GateContext) -> list[CheckResult]:
    """Execute every check (heavy compute lives here; call OUTSIDE the FDR lock)."""
    return [c(ctx) for c in checks]


def apply_demotions(stage1_pass: bool, h_reasons: list, results: list[CheckResult]):
    """SINGLE-SOURCED demotion. Walk results in registry order; the first active hard-fail revokes the
    PASS and appends its reason(s) — matching the legacy short-circuit (`if stage1_pass and ...`) where
    only the first failing check's reason was recorded. Returns (stage1_pass, h_reasons)."""
    reasons = list(h_reasons)
    for r in results:
        if stage1_pass and r.demotes:
            stage1_pass = False
            if r.reason:
                reasons.extend(r.reason if isinstance(r.reason, list) else [r.reason])
    return stage1_pass, reasons


def gates_report(results: list[CheckResult]) -> dict:
    """One uniform gate block for the verdict (single source of truth; downstream renders from this)."""
    return {r.name: {"failure_mode": r.failure_mode, "evaluated": r.evaluated, "passed": r.passed,
                     "active": r.active, "reason": r.reason, **r.metrics} for r in results}


def gate_metric(verdict: dict, name: str, key: str, default=None, flat: str | None = None):
    """Read a gate value from the single-sourced verdict['gates'][name][key], with BACK-COMPAT
    fallback to the legacy flat verdict key (old records pre-date verdict['gates']). `flat` defaults
    to `key`. This is how all consumers read gate verdicts post-unification."""
    g = (verdict.get("gates") or {}).get(name)
    if isinstance(g, dict) and key in g:
        return g[key]
    return verdict.get(flat or key, default)
