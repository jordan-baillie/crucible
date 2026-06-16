#!/bin/bash
# systemd ExecStartPre guard: block until the research layer is QUIESCENT, so the daily
# roll-up (triage / state-backup / morning-report / sentinel / soak) never reads mid-write
# state. Makes the forge->rollup dependency REAL instead of assumed-by-clock.
#
# Why this exists: the forge is latency-bound and variable (26min..3h10m observed). On
# 2026-06-16 it ran to 06:40 — PAST the 06:15 triage timer — so triage/backup read state the
# forge was still writing. Fixed-clock ordering cannot express "wait until research is done".
#
# Behaviour: polls ActiveState of the watched units (oneshots are 'activating' while their
# ExecStart runs, so is-active is NOT sufficient — we check ActiveState explicitly). Bounded
# by a ceiling; on timeout we proceed with a loud log line rather than block the morning report
# forever — a genuinely hung forge fires its own OnFailure=loop-alert independently.
#
# Tunables (env): QUIESCENT_UNITS (default forge+drip; overridable for tests),
#                 QUIESCENT_MAX_WAIT_SEC (default 7200), QUIESCENT_POLL_SEC (default 30).
set -u
UNITS="${QUIESCENT_UNITS:-crucible-forge.service crucible-drip.service}"
MAX_WAIT="${QUIESCENT_MAX_WAIT_SEC:-7200}"
POLL="${QUIESCENT_POLL_SEC:-30}"

_running() {  # 0 if ANY watched unit is starting/running/stopping
  local u st
  for u in $UNITS; do
    st="$(systemctl show -p ActiveState --value "$u" 2>/dev/null)"
    case "$st" in active|activating|deactivating|reloading) return 0 ;; esac
  done
  return 1
}

waited=0
while _running; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "[quiescent-guard] WARNING: research still active after ${MAX_WAIT}s ($UNITS); proceeding" >&2
    exit 0
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done
[ "$waited" -gt 0 ] && echo "[quiescent-guard] waited ${waited}s for research to settle"
exit 0
