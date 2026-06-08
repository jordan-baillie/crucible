"""One fully-autonomous research cycle: propose -> codegen -> sandboxed run-with-retry -> record.
The agent writes the signal itself; the fixed harness owns the rails. Killswitch: touch LOOP_DISABLED."""
import json, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/hephaestus")
sys.path.insert(0, str(ROOT))
from agent.propose import propose
from agent import codegen

MAX_RETRIES = 3
RUNLOG = ROOT / "agent" / "run_log.jsonl"


def _slug(proposal):
    base = re.sub(r"[^a-z0-9]+", "-", (proposal.get("title", "auto")).lower()).strip("-")[:48]
    return f"auto-{base}-{int(time.time())%100000}"


def _run_module(mod_path: str) -> tuple:
    """Run the generated module's SPEC through the harness in a SUBPROCESS (timeout-bounded, confined cwd)."""
    code = (f"import sys; sys.path.insert(0,'{ROOT}')\n"
            f"from importlib import import_module\n"
            f"m = import_module('strategies.{Path(mod_path).stem}')\n"
            f"from sdk.harness import run_experiment\n"
            f"v = run_experiment(m.SPEC, write_wiki=True, alert=True)\n"
            f"import json; print('VERDICT_JSON='+json.dumps({{k:v[k] for k in v}}, default=str))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=1800, cwd=str(ROOT))
    out = r.stdout + "\n" + r.stderr
    m = re.search(r"VERDICT_JSON=(\{.*\})", r.stdout)
    return (json.loads(m.group(1)) if m else None), out


def cycle():
    if (ROOT / "LOOP_DISABLED").exists():
        print("[loop] LOOP_DISABLED present — halting."); return None
    print("[loop] 1. proposing (reading wiki)...")
    prop = propose()
    if "error" in prop:
        print("[loop] proposal failed:", prop.get("error")); return None
    sid = _slug(prop)
    print(f"[loop] proposed: {prop.get('title')} -> {sid}")
    print("[loop] 2. codegen...")
    code = codegen.generate(prop)
    verdict, log = None, ""
    for attempt in range(1, MAX_RETRIES + 1):
        mod = ROOT / "strategies" / f"{sid.replace('-', '_')}.py"
        mod.write_text(code)
        print(f"[loop] 3. run attempt {attempt}...")
        try:
            verdict, log = _run_module(str(mod))
        except subprocess.TimeoutExpired:
            log = "TIMEOUT (>1800s)"
        if verdict is not None:
            break
        tb = "\n".join(l for l in log.splitlines() if any(k in l for k in
              ("Error", "Traceback", "Exception", "line ", "raise", "assert")))[-2500:] or log[-2500:]
        print(f"[loop] run failed; fixing (attempt {attempt})...")
        code = codegen.fix(code, tb)
    outcome = {"ts": datetime.now().isoformat(), "id": sid, "title": prop.get("title"),
               "proposal": prop, "ran": verdict is not None,
               "verdict": verdict, "passed_all": bool(verdict and verdict.get("PASSED_ALL_GATES"))}
    RUNLOG.parent.mkdir(exist_ok=True)
    with open(RUNLOG, "a") as f:
        f.write(json.dumps(outcome, default=str) + "\n")
    if verdict is None:
        print(f"[loop] FAILED after {MAX_RETRIES} retries. logged.")
    else:
        print(f"[loop] DONE: {sid} -> tier {verdict.get('tier')} | holdout {verdict.get('holdout_pass')} | "
              f"PASSED_ALL={verdict.get('PASSED_ALL_GATES')}")
    return outcome


if __name__ == "__main__":
    cycle()
