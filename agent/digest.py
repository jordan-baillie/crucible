"""Daily Telegram digest: summarize recent autonomous cycles (what was tested + verdicts)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/hephaestus")
from sdk.notify import telegram_msg
ROOT = Path("/root/hephaestus")

def digest(n=5):
    rl = ROOT / "agent" / "run_log.jsonl"
    if not rl.exists(): return
    runs = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()][-n:]
    if not runs: return
    lines = ["🔨 <b>Hephaestus digest</b> — last %d cycles\n" % len(runs)]
    for r in runs:
        v = r.get("verdict") or {}
        mark = "🟢 PASS" if r.get("passed_all") else ("✗ fail" if r.get("ran") else "⚠ codegen-failed")
        lines.append(f"{mark} | {r.get('title','?')[:60]}\n   tier {v.get('tier','-')} holdout {v.get('holdout_pass','-')}")
    npass = sum(1 for r in runs if r.get("passed_all"))
    lines.append(f"\n{npass} passed all gates (human review needed)" if npass else "\nNo full-gate passes — base rate holding.")
    telegram_msg("\n".join(lines))

if __name__ == "__main__":
    digest()
