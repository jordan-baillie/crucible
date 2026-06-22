"""Director: keep the shared work queue topped with deduped, prioritized, UNTESTED hypotheses.

Runs under the 'director' lock so only one top-up happens at a time even if several workers
invoke it when the queue runs dry. Dedup is against recorded experiments + everything in-flight
(queued/claimed), so N workers never get the same hypothesis.
"""
from __future__ import annotations

import os
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
_ARMS = {"explore", "refine", "orthogonal", "crossover"}


def _pick_arm(rng) -> str:
    # Operator-directed override (env CRUCIBLE_FORCE_ARM) for a steered run, e.g. an all-`explore`
    # commodities batch. Makes directed runs first-class through the SAME top_up() gate logic
    # (dedup/closed-family/deployability) instead of a parallel seeder that duplicates it. Default
    # (unset/invalid) = the normal weighted split. Exploit arms still fall back to explore in
    # _propose_via_arm when the elite pool can't satisfy them.
    forced = os.environ.get("CRUCIBLE_FORCE_ARM", "").strip().lower()
    if forced in _ARMS:
        return forced
    r, c = rng.random(), 0.0
    for arm, w in ARM_SPLIT:
        c += w
        if r <= c:
            return arm
    return "explore"


def _propose_via_arm(rng) -> tuple[dict, str, list[str]]:
    """THE single arm-selection point. Returns (proposal, arm-actually-used, parent_ids).
    parent_ids = elite-pool ids the exploit arms derived from — the EXPLICIT lineage record
    (research-map graph + retro-analysis); explore has no parents."""
    arm = _pick_arm(rng)
    if arm in ("refine", "orthogonal"):
        e = elite.sample(rng)
        if e is None:
            return propose(), "explore", []  # precondition unmet -> fallback
        prop = propose_mutate(e) if arm == "refine" else propose_orthogonal(e)
        return prop, arm, [str(e.get("id"))]
    if arm == "crossover":
        pair = elite.sample_pair(rng)
        if pair is None:
            return propose(), "explore", []  # <2 families in pool -> fallback
        return propose_crossover(*pair), arm, [str(p.get("id")) for p in pair]
    return propose(), "explore", []


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]


def _theme(prop: dict) -> str:
    """Coarse FAMILY key (robust to LLM rewording) so the 2/theme queue cap catches sibling mutations
    (e.g. reworded value×mom variants) instead of treating each as a novel premium."""
    from agent.families import family_bucket
    return family_bucket(str(prop.get("title", "")) or str(prop.get("premium", "")))


def _strip_auto_affixes(stem: str) -> str:
    """Drop the `auto_` prefix and the `_<agent>_<id>` suffix the worker stamps onto generated
    strategy filenames, so a recorded experiment's stem can be compared to a PROPOSAL title."""
    s = re.sub(r"^auto[_-]", "", stem)
    return re.sub(r"[_-][a-z0-9]{1,8}[_-]\d{1,6}$", "", s)  # _smith3_33634 / -omdtx1-72241


def _page_key(path: Path) -> str:
    """Dedup key for a recorded experiment. Prefer the H1 title (the human-written premium, stable
    under reruns); fall back to the affix-stripped stem. The OLD `_norm(stem)` keyed on the
    `auto_<slug>_<agent>_<id>` filename, which can NEVER equal `_norm(proposal_title)` (the `auto_`
    prefix + id suffix guarantee a miss) — so title dedup against tested experiments was a no-op and
    failed ideas recurred indefinitely."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return _norm(line[2:])
    except OSError:
        pass
    return _norm(_strip_auto_affixes(path.stem))


def _tested_titles() -> set[str]:
    return {_page_key(p) for p in (WIKI / "experiments").glob("*.md")}


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
            prop, arm, parent_ids = _propose_via_arm(rng)
            if "error" in prop:
                continue
            key, th = _norm(prop.get("title", "")), _theme(prop)
            if th in elite._closed_families():
                continue  # HARD gate: family closed by decision (cf. decisions/CLOSED.md) — never enqueue
            if str(prop.get("retail_tradable_5k", "yes")).strip().lower().startswith("no"):
                continue  # DEPLOYABILITY gate (board 2026-06-09): stranded alpha — don't spend a slot+holdout look on it
            if not key or key in tested or key in inflight or themes.get(th, 0) >= 2:
                continue  # dedup vs recorded experiments + in-flight + theme cluster cap
            queue.enqueue(prop, arm=arm, parent_ids=parent_ids)
            inflight.add(key)
            themes[th] = themes.get(th, 0) + 1
            added += 1
        return {"added": added, **queue.stats()}
    finally:
        lock.release()


if __name__ == "__main__":
    print(top_up())
