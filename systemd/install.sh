#!/bin/bash
# Single source of truth for crucible systemd units = this repo's systemd/ dir.
# Deterministically installs the repo's unit files + drop-ins to /etc/systemd/system and
# daemon-reloads. Drift between repo and live is a footgun: a fix to one silently leaves the
# other stale (hit 2026-06-17 with the vendored rails). Workflow: edit units HERE, run this,
# commit. `--check` reports drift without writing (sentinel S13 calls it daily; exit 1 = drift).
#
# Usage:  systemd/install.sh            install repo -> /etc, daemon-reload
#         systemd/install.sh --check    report drift only, no writes (exit 1 if drift)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../crucible/systemd
ETC=/etc/systemd/system
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
changed=0; drift=0

_sync() {  # $1=src(repo) $2=dst(etc) — install if content differs (or report in --check)
  if ! diff -q "$1" "$2" >/dev/null 2>&1; then
    if [ "$CHECK" -eq 1 ]; then echo "DRIFT: $2 differs from repo (or missing)"; drift=1
    else install -D -m 0644 "$1" "$2"; echo "installed: ${2#$ETC/}"; changed=1; fi
  fi
}

# 1) unit files (services + timers, incl. @ templates: crucible-worker@, loop-alert@)
for f in "$REPO"/*.service "$REPO"/*.timer; do
  [ -e "$f" ] || continue
  _sync "$f" "$ETC/$(basename "$f")"
done

# 2) drop-ins (systemd/dropins/<unit>.service.d/*.conf -> /etc/.../<unit>.service.d/)
for d in "$REPO"/dropins/*/; do
  [ -d "$d" ] || continue
  unit="$(basename "$d")"
  for c in "$d"*.conf; do
    [ -e "$c" ] || continue
    _sync "$c" "$ETC/$unit/$(basename "$c")"
  done
done

# 3) drift report — live state NOT represented in the repo (added out-of-band, would be lost)
for f in "$ETC"/crucible-*.service "$ETC"/crucible-*.timer; do
  [ -e "$f" ] || continue
  [ -f "$REPO/$(basename "$f")" ] || { echo "DRIFT: live $(basename "$f") is NOT in the repo — commit or remove it"; drift=1; }
done
for d in "$ETC"/crucible-*.service.d; do
  [ -d "$d" ] || continue
  unit="$(basename "$d")"
  for c in "$d"/*.conf; do
    [ -e "$c" ] || continue
    [ -f "$REPO/dropins/$unit/$(basename "$c")" ] || { echo "DRIFT: live drop-in $unit/$(basename "$c") is NOT in the repo"; drift=1; }
  done
done

if [ "$CHECK" -eq 1 ]; then
  [ "$drift" -eq 0 ] && echo "systemd in sync ✓" || echo "systemd DRIFT detected (run systemd/install.sh to push repo, or commit the live change)"
  exit "$drift"
fi
if [ "$changed" -eq 1 ]; then systemctl daemon-reload && echo "daemon-reloaded"; else echo "already in sync (no changes)"; fi
[ "$drift" -eq 1 ] && echo "WARNING: live units/drop-ins not in repo (see above) — reconcile before relying on the repo"
exit 0
