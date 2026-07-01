"""Advisory night-planner (agent/night_plan.py) + director blend. Pure — no LLM, no network.

Locks the Stage-4 safety contract from tasks/FABLE5_ORCHESTRATION_PLAN.md:
  - _normalize() always yields a valid distribution over the 4 arms (garbage-tolerant).
  - read_plan() rejects a STALE plan (fail-open to pure bandit).
  - director._blend_night_plan() MODULATES the bandit but RE-APPLIES the floors — the 25% explore floor
    always holds, the result is a valid distribution, and no plan / a bad plan is a clean no-op.
"""
import json
from datetime import datetime, timedelta

from agent import bandit, director, night_plan


def test_normalize_yields_a_valid_distribution():
    p = night_plan._normalize({"arm_bias": {"refine": 3, "explore": 1}})
    b = p["arm_bias"]
    assert set(b) == set(bandit.ARMS)
    assert abs(sum(b.values()) - 1.0) < 1e-9
    assert b["refine"] > b["explore"] > 0.0          # relative order preserved
    assert b["orthogonal"] == 0.0 and b["crossover"] == 0.0


def test_normalize_falls_back_to_uniform_on_garbage():
    p = night_plan._normalize({"arm_bias": {"nonsense": "x"}})
    b = p["arm_bias"]
    assert abs(sum(b.values()) - 1.0) < 1e-9
    assert all(abs(v - 0.25) < 1e-9 for v in b.values())   # uniform when no usable signal


def test_read_plan_rejects_stale(tmp_path, monkeypatch):
    pf = tmp_path / "night_plan.json"
    monkeypatch.setattr(night_plan, "PLAN_FILE", pf)
    stale = {"arm_bias": {a: 0.25 for a in bandit.ARMS},
             "generated_at": (datetime.now() - timedelta(hours=night_plan.MAX_AGE_H + 1)).isoformat()}
    pf.write_text(json.dumps(stale))
    assert night_plan.read_plan() is None                  # too old -> ignored (fail-open)
    fresh = {**stale, "generated_at": datetime.now().isoformat()}
    pf.write_text(json.dumps(fresh))
    assert night_plan.read_plan() is not None              # fresh -> honoured


def test_read_plan_missing_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(night_plan, "PLAN_FILE", tmp_path / "does_not_exist.json")
    assert night_plan.read_plan() is None


def test_blend_preserves_explore_floor_and_is_a_distribution(monkeypatch):
    # a plan that wants ALL refine must still be floored: explore >= 0.25, valid distribution.
    plan = {"arm_bias": {"explore": 0.0, "refine": 1.0, "orthogonal": 0.0, "crossover": 0.0},
            "rationale": "attack PBO"}
    # director does a lazy `from agent import night_plan` inside the blend, so patch the real module.
    monkeypatch.setattr(night_plan, "read_plan", lambda: plan)
    blended = dict(director._blend_night_plan(bandit.FIXED_SPLIT))
    assert abs(sum(blended.values()) - 1.0) < 1e-9
    assert blended["explore"] >= bandit.EXPLORE_FLOOR - 1e-9      # floor can never be starved
    assert blended["refine"] > dict(bandit.FIXED_SPLIT)["refine"] # the steer actually raised refine
    assert tuple(a for a, _ in director._blend_night_plan(bandit.FIXED_SPLIT)) == bandit.ARMS


def test_blend_is_noop_without_a_plan(monkeypatch):
    monkeypatch.setattr(night_plan, "read_plan", lambda: None)
    assert director._blend_night_plan(bandit.FIXED_SPLIT) == bandit.FIXED_SPLIT


def test_blend_fails_open_on_error(monkeypatch):
    def boom():
        raise RuntimeError("registry wedged")
    monkeypatch.setattr(night_plan, "read_plan", boom)
    assert director._blend_night_plan(bandit.FIXED_SPLIT) == bandit.FIXED_SPLIT
