# Forge cadence: fix morning-chain coupling (A) + drip-mode prototype (C)

Operator chose option 2 (2026-06-17): fix the latent roll-up coupling bug AND stand up a
distributed-drip schedule behind a flag for an honest A/B vs the nightly batch.

## Key facts (verified)
- Live units are real files in /etc/systemd/system (repo systemd/ has drifted copies). forge
  uses a drop-in (focus.conf) — idiomatic, reversible. systemd 255 → ExecCondition supported.
- Forge is LATENCY-bound, $0 compute. Wall-clock 26min–3h10m; on 2026-06-16 it ran to 06:40,
  PAST the 06:15 triage timer → roll-up read mid-write state. That is the bug A fixes.
- Binding constraint is the FDR family ratchet (registry.py): the promotion bar escalates with
  cumulative DISTINCT families ever tested, permanently. So drip must NOT double-count vs batch.

## Design
ONE mode flag (FORGE_MODE, default `batch`) switches cadence; ExecCondition on BOTH forge and
drip means flipping the flag is the only switch — no double FDR spend, fully reversible.

### A — make forge→rollup dependency REAL (not assumed-by-clock)
- ops/wait_for_research_quiescent.sh: ExecStartPre guard. Polls ActiveState of forge+drip
  (configurable via QUIESCENT_UNITS for testing), waits while running, bounded (120min ceiling),
  proceeds with a loud log on timeout (a hung forge fires its own OnFailure independently).
- Drop-in wait-quiescent.conf on triage/state-backup/morning-report (+ sentinel/soak for
  uniformity) adding ExecStartPre + explicit generous TimeoutStartSec.

### C — drip mode behind the flag
- crucible_paths.forge_mode() single-sources mode resolution (default batch).
- ops/forge_mode_is.sh <mode>: ExecCondition gate, exit 0 iff current mode==arg (clean skip else).
- crucible-drip.service: ExecCondition=mode==drip (+ LOOP_DISABLED); runs FORGE_AGENT=drip
  run_worker --cycles 1; same dated log; NO per-run digest (digest is the daily roll-up).
- crucible-drip.timer: every 3h at {02,05,08,11,14,17,20,23}:30 (8/day ≈ today's 9; avoids the
  06:00–08:00 roll-up window).
- forge.service drop-in mode-gate.conf: ExecCondition=mode==batch (batch no-ops in drip mode).
- Attribution for the A/B is free: run_log records FORGE_AGENT (smith-* vs drip), and only one
  mode writes at a time, so batch vs drip periods are cleanly separable by date.

## Steps
- [ ] 1. crucible_paths.py: FORGE_MODE_FILE + forge_mode(); .gitignore FORGE_MODE
- [ ] 2. ops/forge_mode_is.sh (+chmod +x); standalone exit-code tests
- [ ] 3. ops/wait_for_research_quiescent.sh (+chmod +x); functional test vs transient sleep unit
- [ ] 4. crucible-drip.service + .timer (live + repo copy)
- [ ] 5. forge mode-gate drop-in; rollup wait-quiescent drop-ins (live + repo mirror)
- [ ] 6. daemon-reload; systemd-analyze verify; list-timers; throwaway-unit ExecCondition proof
- [ ] 7. Leave mode=batch so tonight's 03:30 batch runs unchanged + robust. Enable drip.timer.

## Safety
- Next forge 03:30 (~2h). Must end mode=batch (default) → batch runs as today; drip dormant;
  guards are no-ops when forge idle. Prove the ExecCondition mechanics WITHOUT triggering a real
  forge run (throwaway echo unit) — never burn a real forge cycle just to test (it spends budget).

## Out of scope (tracked separately)
- Single-source systemd install.sh + reconcile repo↔live drift (touches all 22 units — not safe
  to do 2h before the live forge). New task.
- Extend measure_scout.py to auto-split batch vs drip gate-progress for the A/B readout.
