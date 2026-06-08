# Hephaestus controls
- **Run one cycle now:** `systemctl start hephaestus-cycle.service` (logs: /tmp/heph_cycle.log)
- **Enable nightly autonomy:** `systemctl enable --now hephaestus-cycle.timer` (03:30 daily, nice'd)
- **KILLSWITCH (halt the loop):** `touch /root/hephaestus/LOOP_DISABLED` (run_one checks this first)
- **Daily digest:** `python3 agent/digest.py` (Telegram). Alerts fire ONLY on full-gate PASS.
- Safety: rails non-bypassable (harness owns gates), no capital/config write scope, human review before any deploy.
