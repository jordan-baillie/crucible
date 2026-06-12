"""Harness integration tests — the gate stack must COMPLETE on every verdict path.

This codebase has had two production gate-wiring breakages that were invisible until
audited (the stage1_pass wiring bug; the SCREEN_FAIL FileLock UnboundLocalError).
These tests drive synthetic, deterministic, no-network StrategySpecs through the REAL
run_experiment() against a tmp wiki and assert each path completes with a stable verdict.

Run:  python3 -m pytest tests/ -x -q
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture()
def tmp_env(monkeypatch, tmp_path):
    """Isolated wiki + research-integrity state so tests never touch production memory."""
    wiki = tmp_path / "wiki"
    for d in ["experiments", "patterns", "decisions", ".queue", ".locks", ".registry", ".elite"]:
        (wiki / d).mkdir(parents=True)
    (wiki / "log.md").write_text("# log\n")
    (wiki / "index.md").write_text("# index\n")
    ri_state = tmp_path / "ri_state"
    ri_state.mkdir()
    monkeypatch.setenv("CRUCIBLE_WIKI", str(wiki))
    monkeypatch.setenv("CRUCIBLE_DEPLOY", "")
    monkeypatch.setenv("RESEARCH_INTEGRITY_DIR", str(ri_state))
    # force re-import against the tmp env
    for m in list(sys.modules):
        if m.startswith(("sdk", "crucible_paths", "research_integrity")):
            sys.modules.pop(m)
    yield wiki


def _panel(n_days=2200, n_assets=8, seed=7, drift=0.0):
    """Deterministic random-walk price panel, business days ending ~now."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-01", periods=n_days)
    rets = rng.normal(drift, 0.01, (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _make_spec(harness, signal, spec_id, grid=None):
    return harness.StrategySpec(
        id=spec_id, family=f"test_{spec_id}", title=f"test {spec_id}", markets=["test"],
        data_desc="synthetic", pre_registration="FROZEN synthetic test spec",
        load_data=lambda: _panel(),
        signal=signal,
        default_params={}, grid=grid or {},
        holdout_start="2024-01-01", deploy_max_positions=8, scope="local",
    )


def _trades_from(w, panel):
    out = []
    for dt, row in w.resample("ME").last().dropna(how="all").iterrows():
        for tkr, wt in row.dropna().items():
            if abs(wt) > 0.01:
                out.append({"ticker": tkr, "entry_date": str(dt.date()), "exit_date": str(dt.date()),
                            "position_value": float(wt) * 1e5, "pnl": 0.0, "sector": tkr})
    return out


def test_screen_fail_path_completes(tmp_env):
    """Pure-noise signal -> SCREEN_FAIL verdict must complete AND reach the wiki.
    Regression: FileLock used at the screen-fail branch before its function-level
    import -> UnboundLocalError crashed every weak hypothesis silently."""
    from sdk import harness

    def noise_signal(panel, **_):
        rets = panel.pct_change()
        w = pd.DataFrame(1.0 / panel.shape[1], index=panel.index, columns=panel.columns).shift(1)
        daily = (w * rets).sum(axis=1) - (w * rets).sum(axis=1)  # exactly zero -> Sharpe 0
        return daily + 1e-9 * np.random.default_rng(0).normal(size=len(daily)), _trades_from(w, panel)

    v = harness.run_experiment(_make_spec(harness, noise_signal, "t-screenfail"),
                               write_wiki=True, alert=False)
    assert v["tier"] == "SCREEN_FAIL"
    assert (Path(tmp_env) / "experiments" / "t-screenfail.md").exists(), \
        "SCREEN_FAIL verdict must be recorded — negative knowledge is the product"
    assert v.get("registry_recorded") is not False


def test_fail_path_completes_with_stable_verdict_keys(tmp_env):
    """A signal with in-sample edge that passes tier-0 must run the FULL stack and
    emit the stable verdict schema downstream consumers rely on."""
    from sdk import harness

    def momo_signal(panel, lookback=60, **_):
        rets = panel.pct_change()
        mom = panel.pct_change(lookback)
        w = mom.clip(-1, 1)
        w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
        daily = (w * rets).sum(axis=1)
        # inject mild in-sample edge so tier-0 passes deterministically
        boost = pd.Series(0.0006, index=daily.index)
        boost[daily.index >= "2024-01-01"] = 0.0
        return daily + boost, _trades_from(w, panel)

    spec = _make_spec(harness, momo_signal, "t-fullstack",
                      grid={"default": {}, "lb120": {"lookback": 120}})
    v = harness.run_experiment(spec, write_wiki=True, alert=False)

    assert v["tier"] != "SCREEN_FAIL"           # tier-0 passed; full stack ran
    for k in ["id", "family", "tier", "promote_bar", "n_families", "dsr", "median_cpcv",
              "pbo", "holdout_sharpe", "holdout_pass", "stage1_pass", "PASSED_ALL_GATES"]:
        assert k in v, f"verdict schema regression: missing {k}"
    assert v["PASSED_ALL_GATES"] in (True, False)
    assert isinstance(v["stage1_pass"], bool)
    assert (Path(tmp_env) / "experiments" / "t-fullstack.md").exists()


def test_soft_expectations_recorded_and_never_blocking(tmp_env):
    """Pre-registered soft expectations (tranched_v3 lesson): a FALSIFIED claim and a
    CRASHING check must both be recorded on the verdict + wiki page, never alter the
    gate outcome, and never kill the run."""
    from sdk import harness

    def momo_signal(panel, lookback=60, **_):
        rets = panel.pct_change()
        mom = panel.pct_change(lookback)
        w = mom.clip(-1, 1)
        w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
        daily = (w * rets).sum(axis=1)
        boost = pd.Series(0.0006, index=daily.index)
        boost[daily.index >= "2024-01-01"] = 0.0
        return daily + boost, _trades_from(w, panel)

    spec = _make_spec(harness, momo_signal, "t-softexp",
                      grid={"default": {}, "lb120": {"lookback": 120}})
    spec.expectations = [
        {"name": "always_true", "claim": "search Sharpe is finite",
         "check": lambda ctx: {"pass": bool(np.isfinite(ctx["search"].mean())),
                               "observed": round(float(ctx["search"].mean()), 6)}},
        {"name": "falsified_claim", "claim": "search mean return > 100%/day (absurd)",
         "check": lambda ctx: {"pass": bool(ctx["search"].mean() > 1.0),
                               "observed": float(ctx["search"].mean())}},
        {"name": "broken_check", "claim": "this check raises",
         "check": lambda ctx: (_ for _ in ()).throw(ValueError("boom"))},
    ]
    v = harness.run_experiment(spec, write_wiki=True, alert=False)

    soft = v["soft_expectations"]
    assert soft is not None and len(soft) == 3
    by = {r["name"]: r for r in soft}
    assert by["always_true"]["pass"] is True
    assert by["falsified_claim"]["pass"] is False and by["falsified_claim"]["status"] == "FALSIFIED"
    assert by["broken_check"]["pass"] is None and by["broken_check"]["status"] == "error"
    assert v["soft_expectations_pass"] is False
    # NEVER blocking: gate fields are computed identically (this spec FAILs on merit anyway,
    # but the schema-level guarantee is that soft results live OUTSIDE the gate booleans).
    assert isinstance(v["stage1_pass"], bool)
    assert v["PASSED_ALL_GATES"] in (True, False)
    page = (Path(tmp_env) / "experiments" / "t-softexp.md").read_text()
    assert "FALSIFIED" in page and "falsified_claim" in page and "check-error" in page


def test_no_expectations_yields_none_not_failure(tmp_env):
    """Legacy specs (no expectations field) must record soft_expectations=None and a
    'prose-only' wiki line — not an empty-list false pass."""
    from sdk import harness

    def momo_signal(panel, lookback=60, **_):
        rets = panel.pct_change()
        w = panel.pct_change(lookback).clip(-1, 1)
        w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
        daily = (w * rets).sum(axis=1)
        boost = pd.Series(0.0006, index=daily.index)
        boost[daily.index >= "2024-01-01"] = 0.0
        return daily + boost, _trades_from(w, panel)

    v = harness.run_experiment(_make_spec(harness, momo_signal, "t-noexp",
                                          grid={"default": {}, "lb120": {"lookback": 120}}),
                               write_wiki=True, alert=False)
    assert v["soft_expectations"] is None
    assert v["soft_expectations_pass"] is None
    page = (Path(tmp_env) / "experiments" / "t-noexp.md").read_text()
    assert "prose-only" in page


def test_registry_append_failure_is_loud(tmp_env, monkeypatch):
    """The shared FDR bar depends on every run being appended. A failed append must
    be flagged in the verdict, never silently swallowed."""
    from sdk import harness
    import research_integrity as ri

    def boom(*a, **k):
        raise OSError("disk full (simulated)")
    monkeypatch.setattr(ri.registry, "append_run", boom)

    def noise(panel, **_):
        w = pd.DataFrame(1.0 / panel.shape[1], index=panel.index, columns=panel.columns).shift(1)
        z = pd.Series(1e-9, index=panel.index)
        return z, _trades_from(w, panel)

    v = harness.run_experiment(_make_spec(harness, noise, "t-regfail"),
                               write_wiki=False, alert=False)
    assert v.get("registry_recorded") is False, \
        "registry append failure must surface in the verdict (FDR-bar integrity)"


def test_write_once_holdout_enforced(tmp_env):
    """Invariant #4: the SAME frozen config may read the holdout exactly once.
    A second run must be refused (holdout gate forced FAIL, violation surfaced)."""
    from sdk import harness

    def momo_signal(panel, lookback=60, **_):
        rets = panel.pct_change()
        w = panel.pct_change(lookback).clip(-1, 1)
        w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
        daily = (w * rets).sum(axis=1)
        boost = pd.Series(0.0006, index=daily.index)
        boost[daily.index >= "2024-01-01"] = 0.0
        return daily + boost, _trades_from(w, panel)

    spec = _make_spec(harness, momo_signal, "t-writeonce")
    v1 = harness.run_experiment(spec, write_wiki=False, alert=False)
    assert v1["holdout_burned"] is False, "first look must be fresh"

    v2 = harness.run_experiment(spec, write_wiki=False, alert=False)
    assert v2["holdout_burned"] is True, "second look must be detected"
    assert v2["holdout_pass"] is False, "second look must force holdout FAIL"
    assert any("WRITE-ONCE" in r for r in v2["holdout_reasons"])


def test_parallel_paths_match_serial(tmp_env, monkeypatch):
    """E1 regression: parallel MCPT/grid must produce IDENTICAL verdicts to serial."""
    from sdk import harness

    def momo_signal(panel, lookback=60, **_):
        rets = panel.pct_change()
        w = panel.pct_change(lookback).clip(-1, 1)
        w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
        daily = (w * rets).sum(axis=1)
        boost = pd.Series(0.0006, index=daily.index)
        boost[daily.index >= "2024-01-01"] = 0.0
        return daily + boost, _trades_from(w, panel)

    grid = {"default": {}, "lb40": {"lookback": 40}, "lb120": {"lookback": 120}}
    spec_p = _make_spec(harness, momo_signal, "t-par", grid=grid)
    v_par = harness.run_experiment(spec_p, write_wiki=False, alert=False)

    monkeypatch.setenv("CRUCIBLE_MCPT_WORKERS", "1")
    spec_s = _make_spec(harness, momo_signal, "t-ser", grid=grid)
    v_ser = harness.run_experiment(spec_s, write_wiki=False, alert=False)

    for k in ["tier", "dsr", "median_cpcv", "pbo", "search_sharpe", "holdout_sharpe", "stage1_pass"]:
        assert v_par[k] == v_ser[k], f"parallel/serial divergence on {k}: {v_par[k]} != {v_ser[k]}"
