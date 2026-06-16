# Rails hardening (operator 2026-06-16) — 3 items, simple + surgical

## 1. Harden the write-once holdout (PRIORITY — underwrites the zero-false-PASS guarantee)
`sdk/harness.py` holdout commit (the `else: try ledger_append except WARN` block):
- [ ] **Fail CLOSED**: if the look can't be recorded, force-fail (`h_pass=False` + reason) instead of WARN-and-continue. A holdout that silently stops enforcing is how a zero-false-PASS record erodes invisibly.
- [ ] **Lock + re-check**: wrap the commit in `FileLock("holdout-ledger", ttl=120)` (mirrors the FDR-registry lock) and **re-lookup inside the lock** so a concurrent smith that claimed the same `config_hash` is caught (write-once preserved under N smiths). The expensive holdout compute stays OUTSIDE the lock (no serialization).
- [ ] Test: append-failure → forced FAIL; concurrent-claim → forced FAIL.

## 2. CPCV purge ≥ holding horizon (close label-overlap leakage for multi-day holds)
`cpcv_path_sharpes` calls `cpcv_splits` WITHOUT purge → hard-coded `purge=1` (only correct for 1-bar holds).
- [ ] `adapter.cpcv_path_sharpes`: add `purge: int = 1`, forward to `cpcv_splits(..., purge=purge)`.
- [ ] `adapter.assemble_bundle`: add `purge: int = 1`, pass to `cpcv_path_sharpes`.
- [ ] `harness`: `_holding_horizon_bars(trades)` = median trade hold in return-series bars (≥1); pass as `purge` to `assemble_bundle`. Conservative (purge ≥ typical horizon). Methodology tightening, not result-tuning; only affects FUTURE runs (no retroactive re-score).
- [ ] Test: purge wired through; a 5-bar-hold strategy gets purge≈5; backward-compatible default=1.

## 3. macro_confound phase-in watch (date-gated active_from=2026-06-29 — "watch", not new building)
- [x] Calibration recorded: `research-wiki/methodology/results-macro-neutralization-calibration.md` (thresholds confirmed, would-demote=ust_rolldown).
- [ ] **Canary gap**: no canary covers macro_confound. A faithful canary needs FRED macro factors → NETWORK → would break the hermetic battery. Decision: do NOT add a network-flaky canary; coverage rests on the recorded calibration + a one-time activation check.
- [ ] **Watch task** for 2026-06-29: confirm clean activation (the `ust_rolldown_regime_carry` would-demote case actually demotes; no spared-high-r2 strategy wrongly demoted) + re-confirm calibration §5 was recorded before the date-gate. Track as a project task.

VERIFY: targeted tests for #1/#2 + full suite green. Changes are STRICTER-only (fail-closed, more purge) — never looser.
