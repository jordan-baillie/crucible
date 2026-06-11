"""Post-forge digest. Severity-routed (2026-06-12): a full-gate PASS is CRITICAL
(immediate Telegram — human review unlocks deployment); an ordinary night is NOT a
phone buzz — the 07:00 morning report already covers it in full. No-PASS nights
log to stdout (forge log) only."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sdk.notify import telegram_critical
from crucible_paths import ROOT, WIKI  # central config

def digest(n=5):
    rl = ROOT / "agent" / "run_log.jsonl"
    if not rl.exists(): return
    runs = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()][-n:]
    if not runs: return
    passes = [r for r in runs if r.get("passed_all")]
    for r in runs:
        v = r.get("verdict") or {}
        mark = "PASS" if r.get("passed_all") else ("fail" if r.get("ran") else "codegen-failed")
        print(f"[digest] {mark} | {r.get('title','?')[:60]} | tier {v.get('tier','-')}")
    if passes:
        lines = [f"🟢 <b>{len(passes)} FULL-GATE PASS</b> — human review required"]
        for r in passes:
            v = r.get("verdict") or {}
            lines.append(f"• {r.get('title','?')[:70]}\n  holdout {v.get('holdout_sharpe','?')} "
                         f"DSR {v.get('dsr','?')} — wiki/experiments/{r.get('id','?')}.md")
        telegram_critical("\n".join(lines))
    else:
        print("[digest] no full-gate passes — base rate holding (details in morning report)")

if __name__ == "__main__":
    digest()
