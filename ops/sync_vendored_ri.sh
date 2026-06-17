#!/bin/bash
# Re-vendor research_integrity: mirror the INSTALLED/canonical package (what the runtime
# actually imports — the shared research-integrity repo via the editable install) into
# crucible's vendored snapshot, so a STANDALONE crucible checkout runs byte-identical rails.
#
# Single source of truth = the installed package. Workflow when you change the rails:
#   1. edit the canonical repo (e.g. /root/shared/research_integrity) + commit/push it
#   2. run THIS script   3. `git -C /root/crucible add vendor/research_integrity && commit`
# Sentinel S12 fails loudly if the two ever drift, so step 2 can never be silently skipped.
set -euo pipefail
DST=/root/crucible/vendor/research_integrity/research_integrity
SRC="$(python3 -c 'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("research_integrity").origin))')"
[ -d "$SRC" ] || { echo "ERROR: cannot locate the installed research_integrity package"; exit 1; }
echo "canonical (installed) : $SRC"
echo "vendored  (snapshot)  : $DST"
mkdir -p "$DST"
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' "$SRC"/ "$DST"/
echo "--- post-sync diff (empty => in sync) ---"
if diff -rq "$SRC" "$DST" | grep -vE '__pycache__|\.pyc'; then
  echo "WARNING: differences remain (see above)"; exit 1
else
  echo "in sync ✓"
fi
