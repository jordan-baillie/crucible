# Hephaestus controls
- **Run one cycle now:** `systemctl start hephaestus-cycle.service` (logs: /tmp/heph_cycle.log)
- **Enable nightly autonomy:** `systemctl enable --now hephaestus-cycle.timer` (03:30 daily, nice'd)
- **KILLSWITCH (halt the loop):** `touch /root/crucible/LOOP_DISABLED` (run_one checks this first)
- **Daily digest:** `python3 agent/digest.py` (Telegram). Alerts fire ONLY on full-gate PASS.
- Safety: rails non-bypassable (harness owns gates), no capital/config write scope, human review before any deploy.

## Multi-agent (Phase 3)
- **Run N smiths** (self-coordinate via shared queue + FDR-registry lock — no dup work, correct FDR bar):
  `systemctl start hephaestus-worker@1 hephaestus-worker@2 hephaestus-worker@3`  (each does `--cycles 3`, then exits)
- **Pre-fill the queue** (optional; workers self-fill when dry): `python3 -m agent.director`
- **Queue state:** `python3 -c "import sys;sys.path.insert(0,'.');from sdk import queue;print(queue.stats())"`
- **Halt ALL agents** (single + workers): `touch /root/crucible/LOOP_DISABLED`
- Worker logs: `/tmp/heph_worker_<i>.log`. Locks: `/root/research-wiki/.locks/`. Queue: `/root/research-wiki/.queue/`.
- worker@ units are **installed DISABLED**; the nightly `hephaestus-cycle` remains the live path until N is enabled.

## LIVE nightly path (2026-06-08): 3-smith forge
- `hephaestus-forge.service` (3 parallel smiths --cycles 3 + digest) runs nightly via `hephaestus-forge.timer` (03:30). The single-agent `hephaestus-cycle.timer` is DISABLED (superseded).
- Run now: `systemctl start hephaestus-forge.service` | logs: `/tmp/heph_forge.log`
- Halt all smiths: `touch /root/crucible/LOOP_DISABLED`
