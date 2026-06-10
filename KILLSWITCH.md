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
