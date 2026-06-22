"""Rail 2 — hypothesis registry (cross-family multiple-testing context).

The effective-N DSR (search_history.py) corrects for the config search WITHIN a strategy. It does
NOT correct for the thousands of DISTINCT ideas an industrialized loop throws at the gate. This
registry logs every battery run's FAMILY (coarse hypothesis class) so the PROMOTION bar can escalate
with the cumulative count of distinct ideas tested (see adapter.promote_dsr). Counting FAMILIES, not
configs, avoids both under- and over-counting (the grid is already DSR-deflated within a family).

Append-only JSONL, pure/deterministic except file I/O. Spec: research/INTEGRITY_RAILS_SPEC.md (Rail 2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import os
from .holdout import _LazyPath
REGISTRY = _LazyPath("hypothesis_registry.jsonl")  # resolved at use time (see holdout._state_dir)


def family_of(strategy: str) -> str:
    """Coarse hypothesis class. Default = strategy name (one idea, many grid configs)."""
    return str(strategy)


def _read_families(path: Optional[Path] = None) -> set:
    p = path or REGISTRY
    fams: set = set()
    if not p.exists():
        return fams
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            fams.add(json.loads(line).get("family"))
        except Exception:
            continue
    fams.discard(None)
    return fams


def distinct_families(extra: Optional[str] = None, path: Optional[Path] = None) -> int:
    """Number of distinct families ever tested, optionally including a not-yet-logged one.

    Always >= 1 (a single idea still carries its own bar at the base level).
    """
    fams = _read_families(path)
    if extra:
        fams.add(extra)
    return max(1, len(fams))


def append_run(record: dict, path: Optional[Path] = None) -> None:
    """Append one battery run to the registry.

    CALLER-ATOMICITY CONTRACT: the FDR invariant is `count distinct families -> compute the promote
    bar -> grade against it -> append THIS family`, and the bar/grade step is caller policy that sits
    between the count and the append — so (unlike the holdout's self-contained ledger_commit_once) the
    atomic unit cannot be encapsulated here. The caller MUST hold one lock across distinct_families()
    + append_run() so N parallel agents can't both read a too-low count, grade against a too-low bar,
    and then both append (multiplying false-discovery risk). Crucible's harness does this under
    FileLock("fdr-registry"); any other caller must serialize the same span. The write itself is a
    single append-mode line (atomic up to PIPE_BUF on POSIX for these small records)."""
    p = path or REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


__all__ = ["family_of", "distinct_families", "append_run", "REGISTRY"]
