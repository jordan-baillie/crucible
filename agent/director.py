"""Director: keep the shared work queue topped with deduped, prioritized, UNTESTED hypotheses.

Runs under the 'director' lock so only one top-up happens at a time even if several workers
invoke it when the queue runs dry. Dedup is against recorded experiments + everything in-flight
(queued/claimed), so N workers never get the same hypothesis.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

from crucible_paths import ROOT, WIKI  # central config
sys.path.insert(0, str(ROOT))

from agent.propose import propose, mutate as propose_mutate, orthogonal as propose_orthogonal, \
    crossover as propose_crossover
from agent.scout import scout
from agent import elite
from sdk import queue
from sdk.locks import FileLock, LockTimeout

TARGET = 4  # keep at least this many items queued

# Arm split (SYNTHESIS_PLAN.md Stage 1c). Fixed until run_log holds >=60 arm-labelled outcomes,
# then a Thompson bandit may be fitted (parked — data-first). Exploit arms fall back to explore
# when pool preconditions are unmet (empty pool / <2 families).
ARM_SPLIT = (("explore", 0.45), ("refine", 0.25), ("orthogonal", 0.15), ("crossover", 0.15))


def _pick_arm(rng) -> str:
    r, c = rng.random(), 0.0
    for arm, w in ARM_SPLIT:
        c += w
        if r <= c:
            return arm
    return "explore"


def _propose_via_arm(rng) -> tuple[dict, str]:
    """THE single arm-selection point. Returns (proposal, arm-actually-used)."""
    arm = _pick_arm(rng)
    if arm in ("refine", "orthogonal"):
        e = elite.sample(rng)
        if e is None:
            return propose(), "explore"  # precondition unmet -> fallback
        return (propose_mutate(e) if arm == "refine" else propose_orthogonal(e)), arm
    if arm == "crossover":
        pair = elite.sample_pair(rng)
        if pair is None:
            return propose(), "explore"  # <2 families in pool -> fallback
        return propose_crossover(*pair), arm
    return propose(), "explore"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]


def _theme(prop: dict) -> str:
    """Coarse FAMILY key (robust to LLM rewording) so the 2/theme queue cap catches sibling mutations
    (e.g. reworded value×mom variants) instead of treating each as a novel premium."""
    from agent.families import family_bucket
    return family_bucket(str(prop.get("title", "")) or str(prop.get("premium", "")))


def _tested_titles() -> set[str]:
    return {_norm(p.stem) for p in (WIKI / "experiments").glob("*.md")}


def top_up(target: int = TARGET, max_new: int = 4) -> dict:
    """Enqueue up to max_new fresh, deduped hypotheses to reach `target` queued items.
    Returns fast (added=0, skipped) if a peer already holds the director lock — the caller
    should then just retry claim_next; the peer is filling the queue."""
    try:
        lock = FileLock("director", ttl=900, wait=8).acquire()
    except LockTimeout:
        return {"added": 0, "skipped": "director busy", **queue.stats()}
    try:
        st = queue.stats()  # E8: one read (stats() rescans the whole queue file)
        if st.get("queued", 0) >= target:
            return {"added": 0, **st}
        if random.random() < 0.4:
            try:
                scout()  # occasionally pull fresh external ideas into candidates.md first
            except Exception as e:
                print(f"[director] scout failed (non-fatal): {e}")
        tested, inflight = _tested_titles(), queue.inflight_titles()
        # seed per-theme counts from what's already in-flight, so we cap clustering (max 2/theme)
        themes: dict[str, int] = {}
        for it in queue._read_all():
            if it["status"] in ("queued", "claimed"):
                th = _theme(it["proposal"])
                themes[th] = themes.get(th, 0) + 1
        added = 0
        need = target - st.get("queued", 0)
        rng = random.Random()
        for _ in range(min(need, max_new) * 3):  # extra tries to find DIVERSE ideas
            if added >= min(need, max_new):
                break
            prop, arm = _propose_via_arm(rng)
            if "error" in prop:
                continue
            key, th = _norm(prop.get("title", "")), _theme(prop)
            if th in elite._closed_families():
                continue  # HARD gate: family closed by decision (cf. decisions/CLOSED.md) — never enqueue
            if str(prop.get("retail_tradable_5k", "yes")).strip().lower().startswith("no"):
                continue  # DEPLOYABILITY gate (board 2026-06-09): stranded alpha — don't spend a slot+holdout look on it
            if not key or key in tested or key in inflight or themes.get(th, 0) >= 2:
                continue  # dedup vs recorded experiments + in-flight + theme cluster cap
            queue.enqueue(prop, arm=arm)
            inflight.add(key)
            themes[th] = themes.get(th, 0) + 1
            added += 1
        return {"added": added, **queue.stats()}
    finally:
        lock.release()


if __name__ == "__main__":
    print(top_up())
