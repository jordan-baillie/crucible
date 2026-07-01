"""agent/night_plan.py — the advisory Fable-5 night-planner (tasks/FABLE5_ORCHESTRATION_PLAN.md, Stage 4).

ONE tool-less Fable-5 call (via the SINGLE config.llm_cmd()-style seam — here config.planner_cmd(),
still --no-tools PURE generation) that reads the joint crucible+atlas snapshot (agent.joint_state) and
emits an advisory `arm_bias` over the four proposal arms. director.top_up() BLENDS it into the Thompson
bandit weights and RE-APPLIES the bandit floors, so the planner MODULATES the empirically-fit allocation
— it never replaces it (RD-Agent(Q)'s ablation: an LLM scheduler that overrides the bandit is *worse*).

Confined to O(nights): one call per forge night, routed to the 'planner' MODEL_POLICY tier so Fable-5
cost is opt-in. Gated behind NIGHT_PLANNER=1. Fail-open everywhere: a missing/garbled/stale plan (or an
unset flag) leaves director on the pure bandit — the planner is strictly additive, never a dependency.

It emits a search-priority HINT only: it cannot touch a gate, the FDR bar, the holdout, or capital.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from crucible_paths import WIKI
from agent.bandit import ARMS
from agent.llm import LLMError

PLAN_FILE = WIKI / ".dashboard" / "night_plan.json"
MAX_AGE_H = 24  # a plan older than one night is stale -> ignored (fail-open to pure bandit)


def plan_enabled() -> bool:
    return os.environ.get("NIGHT_PLANNER", "0").strip().lower() in ("1", "true", "yes", "on")


def _blockers() -> list[dict]:
    """Each near-miss elite's stored BLOCKER (the wall the refine arm must attack) — the signal the
    deterministic bandit is blind to. Graceful: [] on any failure."""
    try:
        from agent import elite
        out = []
        for it in elite.top():
            s = it.get("summary") or {}
            if s.get("blocker"):
                out.append({"family": elite._family(it), "blocker": str(s["blocker"])[:200],
                            "fitness": it.get("fitness"), "pbo": s.get("pbo")})
        return out[:10]
    except Exception:
        return []


def _prompt(snap: dict) -> str:
    research, execution = snap.get("research", {}), snap.get("execution", [])
    payload = {"fdr": research.get("fdr"), "closed_families": research.get("closed_families"),
               "elite_cells": research.get("elite_cells"), "near_miss_blockers": _blockers(),
               "live_books": execution}
    return ("You are the crucible forge night-planner. Below is the JOINT state of the research forge "
            "(the FDR bar + families already charged against it, the MAP-Elites pool, closed/falsified "
            "families, and each near-miss's BLOCKER) and the Atlas execution side (live paper books + "
            "their realised P&L). Allocate TONIGHT's search budget across the four proposal ARMS to "
            "maximise PASSES-per-FDR-look:\n"
            "  explore    = brand-new hypotheses (discovery insurance)\n"
            "  refine     = mutate an elite to ATTACK its named blocker (best when BREAKABLE walls exist, e.g. PBO/overfit)\n"
            "  orthogonal = a new mechanism reusing a proven universe/construction\n"
            "  crossover  = fuse two elites from different families\n"
            "ADVISORY ONLY: your bias MODULATES a data-driven Thompson bandit that keeps a HARD 25% "
            "explore floor, and every idea still faces the full non-bypassable gate stack — you cannot "
            "weaken a gate, deploy, or move capital. Favour refine when near-miss blockers are BREAKABLE "
            "(the one historical PASS beat PBO deliberately via overlapping tranches); favour explore when "
            "the pool is thin or the families are saturated; down-weight families already live or closed.\n\n"
            f"JOINT STATE:\n{json.dumps(payload, indent=1, default=str)[:9000]}\n\n"
            'Return ONLY JSON: {"arm_bias": {"explore": <0-1>, "refine": <0-1>, "orthogonal": <0-1>, '
            '"crossover": <0-1>}, "rationale": "<=1 sentence why", '
            '"focus_hint": "optional: the single wall/family worth attacking tonight"}')


def _normalize(plan: dict) -> dict:
    raw = plan.get("arm_bias") or {}
    b = {}
    for a in ARMS:
        try:
            b[a] = max(float(raw.get(a, 0.0)), 0.0)
        except (TypeError, ValueError):
            b[a] = 0.0
    s = sum(b.values())
    b = {a: (b[a] / s if s > 0 else 1.0 / len(ARMS)) for a in ARMS}  # valid distribution (director re-floors)
    return {"schema_version": 1, "generated_at": datetime.now().isoformat(timespec="seconds"),
            "arm_bias": b, "rationale": str(plan.get("rationale") or "")[:600],
            "focus_hint": str(plan.get("focus_hint") or "")[:300]}


def generate() -> dict:
    """Build the joint snapshot, ask the planner (tool-less, planner tier), normalize, and write the
    plan atomically. Raises LLMError on an unparseable/empty result (fail-loud); main() catches it so a
    dead planner never blocks the forge."""
    from agent import joint_state
    from agent.llm import call, extract_json
    from agent.config import planner_cmd
    snap = joint_state.build()
    text = call(_prompt(snap), cmd=planner_cmd())
    plan = extract_json(text)
    if not isinstance(plan, dict) or "arm_bias" not in plan:
        raise LLMError(f"night-planner returned no parseable arm_bias (len {len(text)})")
    plan = _normalize(plan)
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLAN_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    tmp.replace(PLAN_FILE)
    return plan


def read_plan() -> dict | None:
    """The FRESH plan for director.top_up to blend, or None (missing / stale / garbled) => pure bandit."""
    try:
        if not PLAN_FILE.exists():
            return None
        plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
        if not isinstance(plan.get("arm_bias"), dict):
            return None
        ts = plan.get("generated_at")
        if ts:
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
            if age > MAX_AGE_H * 3600:
                return None  # stale -> ignore
        return plan
    except Exception:
        return None


def main() -> int:
    if not plan_enabled():
        print("[night_plan] NIGHT_PLANNER not set — skipping (director uses pure bandit).")
        return 0
    try:
        plan = generate()
    except Exception as e:  # fail-open: a dead planner must never block the night
        print(f"[night_plan] generate failed (non-fatal; director falls back to pure bandit): "
              f"{type(e).__name__}: {str(e)[:160]}")
        return 0
    print(f"[night_plan] wrote {PLAN_FILE} — arm_bias {plan['arm_bias']} — {plan['rationale'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
