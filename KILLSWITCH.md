# Crucible controls (operator emergency card)

## KILLSWITCH — halt the loop
```bash
touch /root/crucible/LOOP_DISABLED      # checked before EVERY cycle by every smith
```
Re-enable: `rm /root/crucible/LOOP_DISABLED`

## The live nightly path: 3-smith forge
- `crucible-forge.timer` → `crucible-forge.service` at **03:30** (3 parallel smiths, `--cycles 3` each, then digest)
- Run now: `systemctl start crucible-forge.service`
- Logs: `/root/crucible/logs/forge-YYYY-MM-DD.log` (symlinked at `/tmp/crucible_forge.log`)
- Stop a running forge: `systemctl stop crucible-forge.service` (then touch LOOP_DISABLED to prevent restart at next timer fire)

## Cadence mode — batch vs drip (one flag, no double FDR spend)
`FORGE_MODE` (repo root, gitignored; **absent ⇒ batch**) selects which cadence is live. Both timers
are always enabled; each unit self-gates via `ExecCondition` so only the chosen mode does work.
```bash
cat /root/crucible/FORGE_MODE                 # current mode (batch|drip)
echo drip  > /root/crucible/FORGE_MODE         # -> 1 smith every 3h (crucible-drip, 8 slots/day)
echo batch > /root/crucible/FORGE_MODE         # -> nightly 3-smith burst at 03:30 (default)
```
- `batch`: `crucible-forge` runs; `crucible-drip` cleanly skips at each 3h slot.
- `drip`:  `crucible-drip` runs (02,05,08,11,14,17,20,23:30); `crucible-forge` cleanly skips at 03:30.
- Mode is single-sourced in `crucible_paths.forge_mode()`; the gate is `ops/forge_mode_is.sh`.
- **A/B caveat:** before a drip A/B, mirror the forge's active focus (`crucible-forge.service.d/focus.conf`)
  onto `crucible-drip.service`, else cadence is confounded with focus. Attribution is free —
  run_log tags batch rows `smith-*` vs drip rows `drip`, and only one mode writes at a time.
- Roll-up steps (triage/backup/report) now `ExecStartPre`-wait for research to be quiescent
  (`ops/wait_for_research_quiescent.sh`), so a slow/overrunning forge is never read mid-write.

## Other units
| Unit | When | What |
|---|---|---|
| `crucible-morning-report.timer` | 07:00 | Telegram morning report |
| `crucible-state-backup.timer` | 06:30 | wiki + live-state tarball |
| `crucible-lint.timer` | Sun 04:30 | wiki hygiene + strategy archival |
| `crucible-bab-forward.timer` | Mon 06:00 | defensive-BAB forward track |
| `crucible-worker@.service` | manual | extra smiths: `systemctl start crucible-worker@4` |

## Quick state checks
```bash
python3 -c "import sys;sys.path.insert(0,'/root/crucible');from sdk import queue;print(queue.stats())"   # queue
tail -5 /root/crucible/agent/run_log.jsonl                                                                # last cycles
systemctl list-timers 'crucible-*' --no-pager                                                             # schedules
```
- Locks: `$CRUCIBLE_WIKI/.locks/` · Queue: `$CRUCIBLE_WIKI/.queue/` · FDR registry: `$CRUCIBLE_WIKI/.registry/`

## Safety
Rails non-bypassable (harness owns every verdict) · no autonomous capital · Telegram alert only on full-gate PASS · human review before any real-capital action.
