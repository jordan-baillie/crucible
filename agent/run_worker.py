"""A forge worker (one smith). Claim a hypothesis from the shared queue -> codegen ->
sandboxed run (the harness is registry-locked) -> complete + record. Run N in parallel; each
gets a unique agent id so strategy files never collide. Honors LOOP_DISABLED.

  python -m agent.run_worker --cycles 1      # one claim->run cycle (default)
  FORGE_AGENT=smith-2 python -m agent.run_worker --cycles 5
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from crucible_paths import ROOT, WIKI  # central config
sys.path.insert(0, str(ROOT))

from agent import codegen


def _forge_model() -> str | None:
    """O3: record WHICH model generated this run (Fable-5 vs opus nights must be separable
    in retrospectives — the 2026-06-10 routing incident made model attribution mandatory)."""
    try:
        from agent.config import MODEL
        return MODEL
    except Exception:
        return None
from agent.sandbox import scan_code, apply_rlimits
from agent.director import top_up
from sdk import queue
from sdk.locks import FileLock

RUNLOG = ROOT / "agent" / "run_log.jsonl"
MAX_RETRIES = 3
AGENT_ID = os.environ.get("FORGE_AGENT") or f"{socket.gethostname()}-{os.getpid()}"
TAG = (re.sub(r"[^a-z0-9]+", "", AGENT_ID.lower())[-6:]) or "w"


def _slug(proposal: dict) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(proposal.get("title", "auto")).lower()).strip("-")[:40]
    return f"auto-{base}-{TAG}-{int(time.time()) % 100000}"


def _run_module(mod_stem: str):
    """Run the generated SPEC through the harness in a SANDBOXED subprocess (rlimits + timeout)."""
    code = (f"import sys; sys.path.insert(0,'{ROOT}')\n"
            f"from importlib import import_module\n"
            f"m = import_module('strategies.{mod_stem}')\n"
            f"from sdk.harness import run_experiment\n"
            f"v = run_experiment(m.SPEC, write_wiki=True, alert=True)\n"
            f"import json; print('VERDICT_JSON='+json.dumps({{k:v[k] for k in v}}, default=str))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=2700, cwd=str(ROOT),
                       preexec_fn=apply_rlimits if os.name == 'posix' else None)  # headroom for the stage-2 cross-universe battery on a stage-1 pass
    m = re.search(r"VERDICT_JSON=(\{.*\})", r.stdout)
    return (json.loads(m.group(1)) if m else None), (r.stdout + "\n" + r.stderr)


def _arm_reward(verdict: dict | None) -> float:
    """Scalar reward for the proposal arm that produced this run (bandit dataset — Stage 1c).
    Gate progress, not just pass/fail, so near-misses still inform the arm comparison:
    0 didn't run/no signal; 0.25 ran clean; +stage1 0.5; +DSR-scaled; 2.0 floor on full pass."""
    if not verdict:
        return 0.0
    r = 0.25
    if verdict.get("stage1_pass"):
        r = 0.5
        try:
            r += min(max(float(verdict.get("dsr") or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    if verdict.get("PASSED_ALL_GATES"):
        r = max(r, 2.0)
    return round(r, 3)


def run_one_from_queue():
    if (ROOT / "LOOP_DISABLED").exists():
        print(f"[{AGENT_ID}] LOOP_DISABLED -- halting.")
        return None
    item = queue.claim_next(AGENT_ID)
    if item is None:
        # queue dry: try to fill it; if a PEER is already filling (director lock busy),
        # top_up returns fast and we just back off + retry-claim what the peer enqueues.
        for _ in range(15):  # ~75s budget
            try:
                top_up()  # director LLM call (propose/mutate) — guard its 300s timeout etc.
            except Exception as e:
                print(f"[{AGENT_ID}] top_up failed ({type(e).__name__}: {str(e)[:120]}); backing off.")
            item = queue.claim_next(AGENT_ID)
            if item is not None:
                break
            time.sleep(5)
        if item is None:
            print(f"[{AGENT_ID}] nothing to claim after retries.")
            return None
    prop, sid = item["proposal"], _slug(item["proposal"])
    print(f"[{AGENT_ID}] claimed {item['id']}: {str(prop.get('title'))[:48]} -> {sid}")
    verdict, log = None, ""
    # Observability: per-stage wall-clock + retry counters -> run_log.jsonl ("stages" key)
    stages = {"codegen_s": None, "codegen_attempts": None, "consistency_fix": False,
              "sandbox_rejects": 0, "run_attempts": 0, "backtest_s": None, "total_s": None}
    t_cycle = time.time()
    try:
        t0 = time.time()
        code = codegen.generate(prop)
        ok, issues = codegen.consistency_check(prop, code)  # does the code implement the claimed thesis?
        if not ok and issues:
            print(f"[{AGENT_ID}] thesis<->code mismatch: {issues[:120]}; requesting fix...")
            stages["consistency_fix"] = True
            code = codegen.fix(code, f"THESIS MISMATCH — the code must FAITHFULLY implement the proposal's "
                                     f"economic thesis. Fix these mismatches: {issues}")
        stages["codegen_s"] = round(time.time() - t0, 1)
        stages["codegen_attempts"] = codegen.LAST_GEN.get("attempts")
        for attempt in range(1, MAX_RETRIES + 1):
            bad = scan_code(code)
            if bad:
                print(f"[{AGENT_ID}] sandbox REJECT ({bad}); requesting fix...")
                stages["sandbox_rejects"] += 1
                code = codegen.fix(code, f"SANDBOX VIOLATION: {bad}. Remove it entirely; the harness "
                                         f"owns ALL I/O and data is fetched via sdk.adapters only.")
                continue
            mod = ROOT / "strategies" / f"{sid.replace('-', '_')}.py"
            mod.write_text(code)
            print(f"[{AGENT_ID}] run attempt {attempt}...")
            stages["run_attempts"] = attempt
            t_run = time.time()
            try:
                verdict, log = _run_module(mod.stem)
            except subprocess.TimeoutExpired:
                log = "TIMEOUT (>2700s)"
            if verdict is not None:
                stages["backtest_s"] = round(time.time() - t_run, 1)
                break
            tb = "\n".join(l for l in log.splitlines() if any(k in l for k in
                  ("Error", "Traceback", "Exception", "line ", "raise", "assert")))[-2500:] or log[-2500:]
            code = codegen.fix(code, tb)
    except Exception as e:  # never let one bad cycle crash the worker / strand the queue item
        log = f"WORKER EXCEPTION: {type(e).__name__}: {str(e)[:300]}"
        print(f"[{AGENT_ID}] {log}")
    stages["total_s"] = round(time.time() - t_cycle, 1)
    # O3 (schema v2): classify HOW a non-run failed — 'ran: false' alone hid whether the night
    # was lost to codegen, sandbox rejections, runtime crashes, or timeouts (different fixes).
    if verdict is not None:
        fail_reason = None
    elif log.startswith("WORKER EXCEPTION"):
        fail_reason = "worker_exception"
    elif "TIMEOUT" in log:
        fail_reason = "backtest_timeout"
    elif stages["run_attempts"] == 0:
        fail_reason = "sandbox_rejected"  # never produced acceptable code
    else:
        fail_reason = "runtime_error"     # ran out of fix-retries on tracebacks
    outcome = {"schema": 2, "ts": datetime.now().isoformat(), "agent": AGENT_ID,
               # Stage 1c: arm + verdict-derived reward = the dataset for the (parked) Thompson
               # bandit over proposal arms. Logged on EVERY outcome; fit only at N>=60.
               "arm": item.get("arm") or "explore", "arm_reward": _arm_reward(verdict),
               "queue_id": item["id"], "id": sid, "title": prop.get("title"), "proposal": prop,
               "model": _forge_model(), "thinking": os.environ.get("FORGE_THINKING") or None,
               "ran": verdict is not None, "fail_reason": fail_reason,
               "tier": (verdict or {}).get("tier"),  # top-level: greppable without parsing verdict
               "module_sha": (verdict or {}).get("module_sha"),
               "repo_sha": (verdict or {}).get("repo_sha"),
               "verdict": verdict, "stages": stages, "log_tail": (None if verdict is not None else log[-800:]),
               "passed_all": bool(verdict and verdict.get("PASSED_ALL_GATES"))}
    RUNLOG.parent.mkdir(exist_ok=True)
    with FileLock("runlog", ttl=30):  # proposals can exceed the 4KB atomic-append size
        with open(RUNLOG, "a") as f:
            f.write(json.dumps(outcome, default=str) + "\n")
    queue.complete(item["id"], verdict)
    try:
        from agent import elite
        elite.record(outcome)  # feed the MAP-Elites grid (best-per-cell by DSR) for the exploit arms
    except Exception:
        pass
    # Deploy policy (still paper-only, no real capital):
    #   PASSED_ALL (broad + stage-2 confirmed)        -> Paper Book (validated edge, accumulate live evidence)
    #   stage-1 pass with LOCAL scope                  -> Paper Book too: forward-validation IS its
    #     confirmation path, so the shadow track must start immediately (calendar time is the test).
    v = outcome.get("verdict") or {}
    # mcpt_pass None = MCPT not run (pre-MCPT verdicts / non-price panels treated as pass inside the
    # harness); False = construction artifact -> NEVER deploy, even as a local candidate.
    deploy = (outcome["passed_all"] or (isinstance(v, dict) and v.get("stage1_pass")
                                        and v.get("scope") == "local")) \
        and (not isinstance(v, dict) or v.get("mcpt_pass") is not False)
    if deploy:
        try:
            from live.deploy import deploy_to_paper
            res = deploy_to_paper(str(mod))
            why = "PASS" if outcome["passed_all"] else "LOCAL stage-1 candidate (forward-validation)"
            print(f"[{AGENT_ID}] {why} -> Paper Book: {res['name']} ({res['n_positions']} positions)")
        except Exception as e:
            print(f"[{AGENT_ID}] deploy_to_paper failed (verdict still recorded): {str(e)[:200]}")
    print(f"[{AGENT_ID}] DONE {sid} -> tier {verdict and verdict.get('tier')} | "
          f"PASSED_ALL={outcome['passed_all']}")
    return outcome


def loop(max_cycles=None):
    n = 0
    while not (ROOT / "LOOP_DISABLED").exists():
        if max_cycles is not None and n >= max_cycles:
            break
        out = run_one_from_queue()
        n += 1
        if out is None:
            time.sleep(8)  # idle backoff when the queue is empty


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1, help="claim->run cycles to do (default 1)")
    args = ap.parse_args()
    loop(max_cycles=args.cycles)
