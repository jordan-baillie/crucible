# Vendored dependencies

## research_integrity (snapshot)
The research-integrity rails (CPCV/DSR/PBO, FDR-aware promote bar, write-once holdout,
deployment-sanity). **Canonical source on the origin box is `/root/shared/research_integrity`
(pip-installed editable, shared with other projects).** This snapshot makes Crucible
self-contained on a fresh clone:

    pip install -e vendor/research_integrity

Re-snapshot when the canonical copy changes:
    cp -r /root/shared/research_integrity/research_integrity vendor/research_integrity/
