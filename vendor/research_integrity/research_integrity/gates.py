"""Hard-gate evaluation for the Cross-OOS harness (Midas #102).

Encodes the Plan §2 gate table. A candidate must satisfy EVERY gate that is defined in
`thresholds`; a missing measurement is a FAIL (you cannot pass a battery you did not run).

Pure: takes a results bundle (dict of measured values) → structured pass/fail report.
"""
from __future__ import annotations

from dataclasses import dataclass

# Plan §2 defaults. Each entry: (bundle_key, comparator, threshold, human description).
# comparator is one of: '>=', '>', '<=', '<', 'is_true'.
DEFAULT_GATES: dict[str, tuple[str, str, float, str]] = {
    "median_cpcv_sharpe": ("median_cpcv_sharpe", ">=", 0.8, "Median CPCV net Sharpe >= 0.8"),
    "frac_paths_positive": ("frac_paths_positive", ">=", 0.60, ">=60% of CPCV paths net-positive"),
    "pbo": ("pbo", "<=", 0.50, "PBO <= 0.50"),
    "dsr": ("dsr", ">=", 0.95, "Deflated Sharpe Ratio significant at 95%"),
    "top_asset_frac": ("top_asset_frac", "<", 0.35, "Top asset < 35% of net PnL (cross-asset LOO)"),
    "loo_venue_ok": ("loo_venue_ok", "is_true", 1.0, "Cross-venue LOO: positive on >=1 held-out venue"),
    "min_regime_sharpe": ("min_regime_sharpe", ">=", -0.5, "No catastrophic regime (min regime Sharpe >= -0.5)"),
    "max_regime_pnl_frac": ("max_regime_pnl_frac", "<=", 0.50, "<=50% of PnL from any one regime"),
    "cost_stress_sharpe": ("cost_stress_sharpe", ">=", 0.5, "10 bps/side stress: median path Sharpe >= 0.5"),
    "forward_net": ("forward_net", ">", 0.0, "Forward holdout net > 0"),
}

# High-profit targets (informational, not pass/fail).
HIGH_PROFIT_TARGETS = {
    "median_cpcv_sharpe": 1.2,
    "frac_paths_positive": 0.70,
}


@dataclass(frozen=True)
class GateResult:
    name: str
    description: str
    value: float | None
    threshold: float
    comparator: str
    status: str  # 'pass' | 'fail' | 'missing'


def _cmp(value: float, comparator: str, threshold: float) -> bool:
    if comparator == ">=":
        return value >= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "is_true":
        return bool(value)
    raise ValueError(f"unknown comparator {comparator!r}")


def evaluate_gates(bundle: dict, thresholds: dict | None = None) -> dict:
    """Evaluate a results bundle against the gate table.

    Returns dict: {overall_pass: bool, n_pass, n_fail, n_missing, gates: [GateResult...]}.
    A gate whose bundle key is absent or non-finite is 'missing' and fails overall.
    """
    gates = thresholds or DEFAULT_GATES
    results: list[GateResult] = []
    for name, (key, comparator, threshold, desc) in gates.items():
        if key not in bundle or bundle[key] is None:
            results.append(GateResult(name, desc, None, threshold, comparator, "missing"))
            continue
        val = bundle[key]
        try:
            fval = float(val)
        except (TypeError, ValueError):
            results.append(GateResult(name, desc, None, threshold, comparator, "missing"))
            continue
        if comparator != "is_true" and not _is_finite(fval):
            results.append(GateResult(name, desc, fval, threshold, comparator, "missing"))
            continue
        status = "pass" if _cmp(fval, comparator, threshold) else "fail"
        results.append(GateResult(name, desc, fval, threshold, comparator, status))

    n_pass = sum(1 for r in results if r.status == "pass")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_missing = sum(1 for r in results if r.status == "missing")
    return {
        "overall_pass": (n_fail == 0 and n_missing == 0),
        "n_pass": n_pass, "n_fail": n_fail, "n_missing": n_missing,
        "gates": results,
    }


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def format_report(report: dict) -> str:
    """Human-readable gate report."""
    lines = [f"Cross-OOS gates: {'PASS' if report['overall_pass'] else 'FAIL'} "
             f"({report['n_pass']} pass / {report['n_fail']} fail / {report['n_missing']} missing)"]
    icon = {"pass": "✅", "fail": "❌", "missing": "⬜"}
    for r in report["gates"]:
        v = "—" if r.value is None else f"{r.value:.4g}"
        lines.append(f"  {icon[r.status]} {r.description}  (value={v})")
    return "\n".join(lines)


__all__ = ["DEFAULT_GATES", "HIGH_PROFIT_TARGETS", "GateResult", "evaluate_gates", "format_report"]
