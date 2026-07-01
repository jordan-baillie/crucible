"""agent/joint_state.py — the forge's BOTH-SIDES snapshot + the generator's diversity brief.

"summon sees crucible AND atlas" (tasks/FABLE5_ORCHESTRATION_PLAN.md, Stage 0). ONE read-only
aggregator over state that ALREADY exists on both sides of the forge:
  • research side — the FDR registry (bar + families), the MAP-Elites pool (occupied cells), and the
    closed/falsified families;
  • execution side — the Atlas paper books, read through the SAME file-contract seam
    morning_report/live.deploy use (config/live_strategies.json -> data/live/<name>/returns.jsonl),
    with NO cross-repo python import.

It emits joint_state.json (a machine artifact, alongside forge_state.json) and brief() — a bounded
steering string telling the GENERATOR which family-space is already spent or already live, so new
candidates land in UNSPENT regions. That is the most direct lever on discoveries-per-FDR-look: the
shared FDR bar rises with every family ever tested, so re-proposing a burned/closed/deployed family
is pure waste.

Read-only + graceful: every read degrades to empty on any failure (missing wiki, deploy disabled,
partial write), so brief() returns "" — no steer — exactly like propose._focus() when unset. It
biases what is GENERATED, never what passes or deploys (the gate stack is untouched).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from crucible_paths import WIKI, DEPLOY_TARGET

OUT = WIKI / ".dashboard" / "joint_state.json"
SCHEMA_VERSION = 1
_MAX_LIST = 12  # cap every family list in the brief so the prompt cannot grow without bound


def _closed_families() -> list[str]:
    try:
        from agent.elite import _closed_families as cf
        return sorted(cf())
    except Exception:
        return []


def _elite_families() -> list[dict]:
    """Occupied MAP-Elites cells -> [{family, universe, turnover, fitness}] (heavily-exploited space)."""
    try:
        from agent import elite
        out = []
        for it in elite.top():
            cell = elite.cell_of(it)
            fam, uni, turn = (cell.split("|") + ["", "", ""])[:3]
            out.append({"family": fam, "universe": uni, "turnover": turn,
                        "fitness": it.get("fitness")})
        return out
    except Exception:
        return []


def _fdr() -> dict:
    """FDR bar + the distinct families already charged against it (the scarcest resource)."""
    try:
        from agent.forge_state import _parse_registry
        r = _parse_registry()
        return {"bar": r.get("bar"), "n_families": r.get("n_families"),
                "families": [f.get("family") for f in r.get("families", []) if f.get("family")]}
    except Exception:
        return {"bar": None, "n_families": 0, "families": []}


def _live_books() -> list[dict]:
    """Deployed Atlas paper books via the FROZEN file-contract seam (no cross-repo import).
    None-tolerant: CRUCIBLE_DEPLOY unset/empty => [] (research-only box, like live.deploy)."""
    if DEPLOY_TARGET is None:
        return []
    reg_f = DEPLOY_TARGET / "config" / "live_strategies.json"
    live = DEPLOY_TARGET / "data" / "live"
    try:
        reg = json.loads(reg_f.read_text(encoding="utf-8")) if reg_f.exists() else []
    except Exception:
        return []
    out = []
    for s in reg:
        name = s.get("name", "")
        rets = []
        try:
            rf = live / name / "returns.jsonl"
            if rf.exists():
                rets = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            rets = []
        cum = 1.0
        for r in rets:
            try:
                cum *= 1 + float(r.get("ret") or 0)
            except (TypeError, ValueError):
                pass
        out.append({"name": name, "state": s.get("state"),
                    "family": _bucket(name), "days": len(rets),
                    "cum_return_pct": round((cum - 1) * 100, 2) if rets else None})
    return out


def _bucket(text: str) -> str:
    try:
        from agent.families import family_bucket
        return family_bucket(text or "")
    except Exception:
        return ""


def build() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "research": {
            "fdr": _fdr(),
            "elite_cells": _elite_families(),
            "closed_families": _closed_families(),
        },
        "execution": _live_books(),
    }


def _dedup(seq) -> list[str]:
    seen, out = set(), []
    for x in seq:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def brief(snap: dict | None = None) -> str:
    """A bounded, generator-only steering string (same category as propose._focus): the families that
    are already CLOSED, already LIVE, or already heavily EXPLOITED — so the generator steers AWAY from
    spent/live family-space toward orthogonal premia in different markets. Returns "" when there is
    nothing to steer against (fresh machine / no wiki / deploy disabled) => no steer, safe default."""
    snap = snap or build()
    closed = _dedup(snap["research"]["closed_families"])[:_MAX_LIST]
    live = _dedup(b["family"] or b["name"] for b in snap["execution"])[:_MAX_LIST]
    exploited = _dedup(c["family"] for c in snap["research"]["elite_cells"])[:_MAX_LIST]
    if not (closed or live or exploited):
        return ""
    lines = ["\n\n=== DIVERSITY BRIEF (steer AWAY from already-spent / already-live family-space) ==="]
    if live:
        lines.append("ALREADY DEPLOYED (live paper books — do NOT rebuild these premia): " + ", ".join(live))
    if closed:
        lines.append("CLOSED / FALSIFIED (never re-open): " + ", ".join(closed))
    if exploited:
        lines.append("HEAVILY EXPLOITED (elite pool already covers these — needs a genuinely fresh "
                     "angle or a DIFFERENT market): " + ", ".join(exploited))
    lines.append("Prefer ORTHOGONAL premia in DIFFERENT markets from the families above. (The shared "
                 "FDR bar rises with every family ever tested; a re-tread of a spent family is wasted budget.)")
    return "\n".join(lines)


def main() -> int:
    snap = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, indent=1, default=str), encoding="utf-8")
    tmp.replace(OUT)
    print(f"[joint_state] wrote {OUT} — {len(snap['execution'])} live books, "
          f"{snap['research']['fdr'].get('n_families')} FDR families, "
          f"{len(snap['research']['closed_families'])} closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
