#!/bin/bash
# systemd ExecCondition gate: exit 0 iff the current forge cadence mode == $1, else exit 1.
#
# ExecCondition semantics (systemd): a clean non-zero (1-254) makes the unit SKIP without
# failing (no OnFailure/loop-alert) — exactly right for "this cadence isn't active right now".
#
# Mode resolution is single-sourced in crucible_paths.forge_mode() (NO duplicated default
# logic here): 'batch' nightly burst vs 'drip' every-3h. Absent/garbage FORGE_MODE -> 'batch'.
#
# Usage (in a drop-in / unit):  ExecCondition=/root/crucible/ops/forge_mode_is.sh batch
set -u
want="${1:?usage: forge_mode_is.sh <batch|drip>}"
cd /root/crucible || exit 255   # 255 = real failure (can't resolve repo), not a clean skip
mode="$(python3 -c 'from crucible_paths import forge_mode; print(forge_mode())' 2>/dev/null)"
[ "$mode" = "$want" ] && exit 0 || exit 1
