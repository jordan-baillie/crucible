"""Shared hypothesis work queue for N forge agents.

A single JSONL at research-wiki/.queue/queue.jsonl, mutated only under the 'queue' FileLock
with atomic temp-file replace. status: queued -> claimed -> done. The director enqueues;
workers claim_next() + complete(). Stale claims (dead worker) are reclaimed after CLAIM_TTL.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from sdk.locks import FileLock

from crucible_paths import QUEUE  # env-overridable via HEPH_QUEUE
CLAIM_TTL = 3600  # reclaim a claimed-but-unfinished item after 1h (worker presumed dead)


def _read_all() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]


def _write_all(items: list[dict]) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    # UNIQUE temp per writer (not a shared queue.tmp) — defends against any stale-steal lock
    # edge where two writers race on the same temp path (caused a FileNotFoundError on replace).
    tmp = QUEUE.parent / f".queue.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp.write_text("".join(json.dumps(i) + "\n" for i in items))
        tmp.replace(QUEUE)  # atomic on POSIX
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def enqueue(proposal: dict, arm: str | None = None) -> str:
    item = {"id": uuid.uuid4().hex[:12], "status": "queued", "ts": time.time(),
            "arm": arm or "explore",  # which proposal arm produced this (bandit dataset, Stage 1c)
            "proposal": proposal, "claimed_by": None, "claimed_at": None}
    with FileLock("queue"):
        items = _read_all()
        items.append(item)
        _write_all(items)
    return item["id"]


def claim_next(agent: str) -> dict | None:
    """Atomically take the oldest queued item (reclaiming stale claims first)."""
    with FileLock("queue"):
        items = _read_all()
        now = time.time()
        for it in items:
            if it["status"] == "claimed" and now - (it.get("claimed_at") or 0) > CLAIM_TTL:
                it["status"], it["claimed_by"] = "queued", None
        nxt = next((it for it in items if it["status"] == "queued"), None)
        if nxt is None:
            _write_all(items)
            return None
        nxt["status"], nxt["claimed_by"], nxt["claimed_at"] = "claimed", agent, now
        _write_all(items)
        return nxt


def complete(item_id: str, verdict: dict | None) -> None:
    with FileLock("queue"):
        items = _read_all()
        for it in items:
            if it["id"] == item_id:
                it["status"], it["done_at"] = "done", time.time()
                it["passed_all"] = bool(verdict and verdict.get("PASSED_ALL_GATES"))
        _write_all(items)


def inflight_titles() -> set[str]:
    """Normalized titles currently queued or claimed (for director dedup)."""
    return {_norm(it["proposal"].get("title", "")) for it in _read_all()
            if it["status"] in ("queued", "claimed")}


def stats() -> dict:
    c = Counter(it["status"] for it in _read_all())
    return {"total": sum(c.values()), "queued": c.get("queued", 0),
            "claimed": c.get("claimed", 0), "done": c.get("done", 0)}


def _norm(t: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40]
