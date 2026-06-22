"""Rail 1 — write-once holdout partition (SHARED research-integrity infra).

PROJECT-AGNOSTIC: holdout_gate/config_hash/ledger are pure + reusable. Each project writes its OWN runner that
produces holdout-period returns+trades, then calls holdout_gate(). Paths are RESEARCH_INTEGRITY_DIR-
configurable (or research_integrity.configure()). Original:

Battery SEARCH runs are quarantined to data strictly before `holdout_start` (the loop physically
cannot read holdout rows during search). A candidate that reaches PROMOTE is evaluated on the
holdout EXACTLY ONCE — enforced by an append-only single-use ledger so a
config cannot be iterated against the holdout. A PROMOTE that degrades on the holdout is downgraded
to FAIL and burned.

Spec: research/INTEGRITY_RAILS_SPEC.md (Rail 1).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import os
import sys
PROJECT = Path(__file__).resolve().parents[2]
_OVERRIDE_DIR = None  # set via research_integrity.configure(state_dir)


def _state_dir() -> Path:
    """Resolve the state dir LAZILY (per call), so import order can never silently point
    holdout/registry state at os.getcwd() — the M5 config-by-import-order trap."""
    return _OVERRIDE_DIR or Path(os.environ.get("RESEARCH_INTEGRITY_DIR", os.getcwd()))


class _LazyPath:
    """Path-like whose location is resolved at USE time (back-compat for module constants)."""
    def __init__(self, name): self._name = name
    def _p(self) -> Path: return _state_dir() / self._name
    def __getattr__(self, a): return getattr(self._p(), a)
    def __fspath__(self): return str(self._p())
    def __str__(self): return str(self._p())
    def __truediv__(self, o): return self._p() / o


HOLDOUT_CFG = _LazyPath("holdout.json")          # per-project: set RESEARCH_INTEGRITY_DIR or configure()
LEDGER = _LazyPath("holdout_ledger.jsonl")       # write-once single-use ledger (per project)

# Pre-registered holdout-gate thresholds (frozen).
MIN_HOLDOUT_SHARPE = 0.0           # must be net-positive on truly unseen data
MAX_DEGRADATION_PCT = -50.0        # holdout Sharpe may fall at most 50% vs search


def load_holdout_config() -> Optional[dict]:
    if not HOLDOUT_CFG.exists():
        return None
    try:
        return json.load(open(HOLDOUT_CFG))
    except Exception:
        return None


def holdout_start_ts() -> Optional[pd.Timestamp]:
    cfg = load_holdout_config()
    if cfg and cfg.get("holdout_start"):
        return pd.Timestamp(cfg["holdout_start"])
    return None


def config_hash(strategy: str, primary_config: Optional[dict], market: str) -> str:
    payload = json.dumps({"s": strategy, "m": market, "p": primary_config or {}},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ledger_lookup(h: str) -> Optional[dict]:
    if not LEDGER.exists():
        return None
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("config_hash") == h:
            return rec
    return None


def ledger_append(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


from contextlib import contextmanager


@contextmanager
def _ledger_lock(timeout: float = 120.0):
    """Self-contained advisory lock co-located with the ledger state dir.

    Makes the single-use guarantee INTRINSIC to the rail instead of dependent on every caller
    remembering to wrap lookup+append in its own lock (the footgun: a future or other-project
    caller does lookup() then append() with no lock, the check-then-act races, and a slice the
    holdout protects gets double-booked). flock on POSIX; non-POSIX degrades to a no-op with a
    LOUD warning (the re-check below still runs — POSIX is the supported deployment)."""
    _state_dir().mkdir(parents=True, exist_ok=True)
    lockf = _state_dir() / "holdout_ledger.lock"
    try:
        import fcntl
        import time
    except ImportError:
        sys.stderr.write("[research_integrity] WARNING: no fcntl (non-POSIX) — holdout ledger "
                         "commit is NOT cross-process race-proof on this host\n")
        yield
        return
    fh = open(lockf, "w")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"holdout ledger lock busy > {timeout}s — another writer wedged")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def ledger_commit_once(config_hash: str, rec: dict) -> Optional[dict]:
    """ATOMIC single-use commit — the write-once guarantee made INTRINSIC to the rail.

    Under the ledger lock: RE-look-up `config_hash`, and append `rec` IFF no prior look exists.
    Returns the PRIOR record if this config was already evaluated on the holdout (the caller MUST
    treat that as 'holdout already burned' and never count the read as out-of-sample), else None
    (this call booked the one and only look). Callers must NOT hand-roll lookup()+append(): doing
    so leaves a check-then-act race that silently double-books the quarantined slice. `config_hash`
    is forced into the stored record (single source of the key the ledger is indexed by)."""
    with _ledger_lock():
        existing = ledger_lookup(config_hash)
        if existing is not None:
            return existing
        ledger_append({**rec, "config_hash": config_hash})
        return None


def holdout_gate(holdout_sharpe: float, degradation_pct: Optional[float],
                 deployment_passed: bool) -> Tuple[bool, List[str]]:
    """Pure gate logic (testable without a backtest)."""
    reasons: List[str] = []
    if not (holdout_sharpe == holdout_sharpe and holdout_sharpe > MIN_HOLDOUT_SHARPE):
        reasons.append(f"holdout_sharpe {holdout_sharpe:.3f} <= {MIN_HOLDOUT_SHARPE}")
    if degradation_pct is not None and degradation_pct < MAX_DEGRADATION_PCT:
        reasons.append(f"degradation {degradation_pct:.1f}% < {MAX_DEGRADATION_PCT}%")
    if not deployment_passed:
        reasons.append("holdout deployment-sanity FAIL")
    return (len(reasons) == 0), reasons


# NOTE (S6 2026-06-10): evaluate_holdout (the original Atlas reference runner) was REMOVED as
# dead code — it lazy-imported atlas modules (research.cross_oos, scripts.validate_oos) that no
# longer exist, and had zero callers across crucible/atlas/boreas. Each project writes its OWN
# runner producing holdout returns+trades, then calls holdout_gate() (see crucible sdk/harness.py).

__all__ = ["load_holdout_config", "holdout_start_ts", "config_hash", "ledger_lookup",
           "ledger_append", "ledger_commit_once", "holdout_gate",
           "MIN_HOLDOUT_SHARPE", "MAX_DEGRADATION_PCT"]
