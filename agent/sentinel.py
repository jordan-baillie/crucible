"""Data-integrity sentinel — daily READ-ONLY checks across the research/trading estate.

The safest possible loop (alert-only, zero writes) and the TEMPLATE for new loops
(cf. tasks/LOOPS_FRAMEWORK_PLAN.md 2.2). Every check is a pure assertion against an
objective source of truth; failures are collected and sent as ONE Telegram message.
Silence = healthy. Exit code 1 if any check fails (systemd visibility).

Checks:
  S1  Sharadar freshness        SEP.zip mtime within N days (data pipeline alive)
  S2  SEP cache schema          cached parquet columns ⊇ source columns we depend on
                                (the 2026-06-10 closeunadj gap, generalized)
  S3  SEP cache staleness       cache at least as new as SEP.zip (rebuild trigger works)
  S4  Forward-paper liveness    returns.jsonl last row within N business days + finite
  S5  Forward-paper equity      book equity within sane band (fat-finger / runaway)
  S6  Wiki off-box backup       last nightly auto-push commit is fresh (off-box memory)
  S7  Queue health              research queue not empty-and-stale (forge starvation)
  S8  Run-log heartbeat         forge produced rows on the last scheduled night
  S9  Holdout ledger integrity  jsonl parses, no duplicate config_hash entries
  S10 Loop registry sync        every live crucible/atlas timer has a row in wiki loops.md
                                (comprehension-debt rule: unregistered loop = stray)
  S11 Forward-paper log scan    yesterday's cycle log carries no 'FAILED' step markers
                                (the steps are '|| echo FAILED' guarded — exit 0 lies)
  S12 Rails vendoring sync      crucible's vendored research_integrity == the INSTALLED
                                package the runtime imports (no stale rails on a standalone
                                checkout; 2026-06-17 a fix edited the vendored copy while
                                runtime used the /root/shared install — silent divergence)

Usage: python3 -m agent.sentinel
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from crucible_paths import ROOT, DATA, WIKI, QUEUE, RUN_LOG, REGISTRY

SHARADAR = DATA / "sharadar"
CACHE = DATA / "cache" / "sep_long_v2.parquet"
LIVE = Path("/root/atlas/data/live")
REGISTRY = Path("/root/atlas/config/live_strategies.json")


def _books() -> list[str]:
    """Registry-driven (#36): every deployed book, not a hardcoded name."""
    try:
        return [s["name"] for s in json.loads(REGISTRY.read_text(encoding="utf-8"))]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return []
REQUIRED_FIELDS = ("close", "closeadj", "closeunadj", "volume")

MAX_SHARADAR_AGE_D = 7        # weekly refresh cadence + slack
MAX_RETURNS_AGE_BD = 3        # forward-paper runs Mon-Fri 23:45
MAX_WIKI_PUSH_AGE_H = 36      # nightly 06:30 backup + slack
MAX_RUNLOG_GAP_D = 2          # forge runs nightly 03:30


def _age_days(p: Path) -> float | None:
    return (time.time() - p.stat().st_mtime) / 86400 if p.exists() else None


def _bdays_since(date_str: str) -> int:
    import pandas as pd
    return len(pd.bdate_range(date_str, datetime.now().strftime("%Y-%m-%d"))) - 1


def check_sharadar(fail):
    sep = SHARADAR / "SEP.zip"
    age = _age_days(sep)
    if age is None:
        fail(f"S1 Sharadar: {sep} MISSING")
    elif age > MAX_SHARADAR_AGE_D:
        fail(f"S1 Sharadar: SEP.zip is {age:.1f}d old (> {MAX_SHARADAR_AGE_D}d) — refresh pipeline dead?")


def check_sep_cache(fail):
    if not CACHE.exists():
        fail(f"S2 SEP cache: {CACHE.name} missing (first strategy run will pay the rebuild)")
        return
    try:
        import pyarrow.parquet as pq
        cols = set(pq.ParquetFile(CACHE).schema_arrow.names)
        missing = [f for f in REQUIRED_FIELDS if f not in cols]
        if missing:
            fail(f"S2 SEP cache: missing fields {missing} — strategies needing them will runtime_error")
    except Exception as e:
        fail(f"S2 SEP cache: unreadable ({type(e).__name__}: {e})")
    src = SHARADAR / "SEP.zip"
    if src.exists() and CACHE.exists() and src.stat().st_mtime > CACHE.stat().st_mtime:
        fail("S3 SEP cache: OLDER than SEP.zip — rebuild trigger failed (stale prices feeding rails)")


def check_forward_paper(fail):
    """S4/S5 per deployed book (registry-driven, #36). Equity sanity is vs the book's
    own registered capital (0.5x–3x band) — the old global band only fit val_mom."""
    books = _books()
    if not books:
        fail("S4 forward-paper: live_strategies.json missing/empty — no books to check")
        return
    try:
        caps = {s["name"]: float(s.get("capital") or 0)
                for s in json.loads(REGISTRY.read_text(encoding="utf-8"))}
    except Exception:
        caps = {}
    for book in books:
        rj = LIVE / book / "returns.jsonl"
        if not rj.exists():
            fail(f"S4 forward-paper[{book}]: {rj} missing — recorder never ran?")
            continue
        rows = [json.loads(l) for l in rj.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            # A just-restarted / re-baselined book legitimately has a baseline equity_state but
            # no returns YET — healthy-but-new, not broken (mirrors L4's 'too new to assess').
            es = LIVE / book / "equity_state.json"
            try:
                base = json.loads(es.read_text(encoding="utf-8")) if es.exists() else {}
            except (OSError, json.JSONDecodeError):
                base = {}
            if base.get("date") and _bdays_since(base["date"]) <= MAX_RETURNS_AGE_BD:
                continue  # fresh baseline; real returns accrue next cycle
            fail(f"S4 forward-paper[{book}]: returns.jsonl EMPTY and no fresh baseline — recorder dead?")
            continue
        last = rows[-1]
        gap = _bdays_since(last["date"])
        if gap > MAX_RETURNS_AGE_BD:
            fail(f"S4 forward-paper[{book}]: last return {last['date']} ({gap} bdays ago > {MAX_RETURNS_AGE_BD}) — daily cycle dead?")
        r = last.get("ret")
        if r is None or not (-0.5 < float(r) < 0.5):
            fail(f"S4 forward-paper[{book}]: insane daily return {r} on {last['date']}")
        eq, cap = last.get("equity"), caps.get(book, 0)
        if eq is not None and cap > 0 and not (0.5 * cap <= float(eq) <= 3.0 * cap):
            fail(f"S5 forward-paper[{book}]: equity ${float(eq):,.0f} outside 0.5x–3x of "
                 f"registered capital ${cap:,.0f}")


def check_wiki_pushed(fail):
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%ct %s"], cwd=str(WIKI),
                           capture_output=True, text=True, timeout=30)
        ts, _, msg = r.stdout.strip().partition(" ")
        age_h = (time.time() - int(ts)) / 3600
        if age_h > MAX_WIKI_PUSH_AGE_H:
            fail(f"S6 wiki backup: last commit {age_h:.0f}h ago (> {MAX_WIKI_PUSH_AGE_H}h) — nightly snapshot dead? ('{msg[:40]}')")
        # verify it actually landed off-box, not just committed locally
        r2 = subprocess.run(["git", "status", "-sb"], cwd=str(WIKI), capture_output=True, text=True, timeout=30)
        if "ahead" in r2.stdout:
            fail(f"S6 wiki backup: local commits NOT pushed ({r2.stdout.strip().splitlines()[0]})")
    except Exception as e:
        fail(f"S6 wiki backup: git check failed ({type(e).__name__}: {e})")


def check_queue(fail):
    if not QUEUE.exists():
        fail(f"S7 queue: {QUEUE} missing")
        return
    rows = [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    open_items = [r for r in rows if r.get("status") in ("queued", "claimed")]
    if not open_items and _age_days(QUEUE) and _age_days(QUEUE) > 2:
        fail("S7 queue: no open items and file stale >2d — director top-up failing (forge will idle)")


def check_run_log(fail):
    if not RUN_LOG.exists():
        fail(f"S8 run-log: {RUN_LOG} missing")
        return
    age = _age_days(RUN_LOG)
    if age is not None and age > MAX_RUNLOG_GAP_D:
        fail(f"S8 run-log: no forge activity for {age:.1f}d (> {MAX_RUNLOG_GAP_D}d) — nightly forge dead?")


def check_holdout_ledger(fail):
    led = REGISTRY / "holdout_ledger.jsonl"
    if not led.exists():
        return  # legitimate pre-first-run state
    seen, dupes = set(), []
    for i, l in enumerate(led.read_text(encoding="utf-8").splitlines()):
        if not l.strip():
            continue
        try:
            h = json.loads(l).get("config_hash")
        except json.JSONDecodeError:
            fail(f"S9 holdout ledger: line {i + 1} unparseable — write-once enforcement compromised")
            return
        if h in seen:
            dupes.append(h)
        seen.add(h)
    if dupes:
        fail(f"S9 holdout ledger: DUPLICATE config_hash entries {dupes[:3]} — a second OOS look was recorded; investigate")


def check_loop_registry(fail):
    """S10: every live crucible/atlas timer must be registered in wiki loops.md."""
    reg = WIKI / "loops.md"
    if not reg.exists():
        fail("S10 loop registry: research-wiki/loops.md missing")
        return
    txt = reg.read_text(encoding="utf-8")
    try:
        r = subprocess.run(["systemctl", "list-timers", "--all", "--no-legend", "--plain"],
                           capture_output=True, text=True, timeout=10)
        timers = [l.split()[-2] for l in r.stdout.splitlines()
                  if l.strip() and ("crucible-" in l or "atlas-" in l)]
    except Exception as e:
        fail(f"S10 loop registry: systemctl unavailable ({e})")
        return
    strays = [t for t in timers if t.removesuffix(".timer") not in txt]
    if strays:
        fail(f"S10 loop registry: live timers NOT in loops.md: {strays} — "
             f"register or disable (unregistered loop = stray)")


def check_forward_paper_log(fail):
    """S11: ops/forward-paper.sh guards each step with '|| echo ... FAILED' so the unit
    exits 0 even when a step dies — scan the last cycle's log block for the markers."""
    log = Path("/root/atlas/data/live/forward_paper.log")
    if not log.exists():
        return  # S4 already covers total absence via returns.jsonl
    lines = log.read_text(encoding="utf-8").splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("=== forward-paper cycle")]
    if not starts:
        return
    last_block = lines[starts[-1]:]
    failed = [l for l in last_block if "FAILED" in l]
    if failed:
        fail(f"S11 forward-paper log: last cycle had failed steps: {failed} "
             f"(unit exited 0 — this is the only place the failure shows)")


def check_rails_vendoring(fail):
    """S12: crucible's VENDORED research_integrity snapshot must match the INSTALLED package
    the runtime actually imports — else a standalone crucible checkout would run STALE rails.
    (2026-06-17: a deployment_sanity fix edited the vendored copy while the runtime imported
    the /root/shared editable install; the two silently diverged. Re-vendor: ops/sync_vendored_ri.sh.)"""
    import importlib.util, hashlib
    spec = importlib.util.find_spec("research_integrity")
    if spec is None or not spec.origin:
        fail("S12 rails sync: cannot locate the installed research_integrity package")
        return
    installed = Path(spec.origin).parent
    vendored = ROOT / "vendor" / "research_integrity" / "research_integrity"
    if not vendored.exists():
        fail(f"S12 rails sync: vendored copy missing at {vendored} — run ops/sync_vendored_ri.sh")
        return

    def _hashes(d: Path) -> dict:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.py"))}

    hi, hv = _hashes(installed), _hashes(vendored)
    if hi != hv:
        differing = sorted({k for k in set(hi) | set(hv) if hi.get(k) != hv.get(k)})
        fail(f"S12 rails sync: vendored research_integrity DRIFTED from installed ({installed}) "
             f"— files differ: {differing[:6]}. Run ops/sync_vendored_ri.sh + commit (stale rails ship otherwise)")


CHECKS = [check_sharadar, check_sep_cache, check_forward_paper,
          check_wiki_pushed, check_queue, check_run_log, check_holdout_ledger,
          check_loop_registry, check_forward_paper_log, check_rails_vendoring]


def main() -> int:
    failures: list[str] = []
    fail = failures.append
    for c in CHECKS:
        try:
            c(fail)
        except Exception as e:   # a crashed check is itself a failure (never silent)
            fail(f"{c.__name__}: check crashed ({type(e).__name__}: {str(e)[:120]})")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if failures:
        for f in failures:
            print(f"[sentinel] FAIL {f}")
        # Severity routing (operator directive: critical-only Telegram).
        # CRITICAL = money-path: forward-paper book dead/insane/failed-steps (S4/S5/S11),
        # write-once holdout integrity (S9). Everything else (data freshness, backup lag,
        # queue starvation, registry drift) waits for the morning report.
        crit_tags = ("S4 ", "S5 ", "S9 ", "S11 ")
        crit = [f for f in failures if f.startswith(crit_tags) or "crashed" in f]
        rest = [f for f in failures if f not in crit]
        if crit:
            from sdk.notify import telegram_critical
            telegram_critical("🛰 <b>Sentinel — CRITICAL</b> — " + stamp + "\n"
                              + "\n".join("❌ " + f for f in crit))
        if rest:
            from sdk.notify import notice
            for f in rest:
                notice("❌ " + f, source="sentinel")
        return 1
    print(f"[sentinel] {stamp}: all {len(CHECKS)} checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
