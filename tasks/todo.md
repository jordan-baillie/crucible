# TODO — Macro-neutralization gate (annotate-only ship)

Operator approved option 1: freeze the pre-reg + ship the gate ANNOTATE-ONLY (computes & records
macro-R²/residual-Sharpe on every verdict; **demotes nothing** — behaviour-neutral). Demotion is a
separate later step gated on calibration + MACRO_DEMOTES_FROM.

Pre-reg: research-wiki/methodology/prereg-macro-neutralization-gate.md (FROZEN this session).

## Plan
- [ ] 1. Freeze pre-reg: stamp §7 (timestamp + pre-stamp commit a036d3e), Status DRAFT→FROZEN. Fix crypto history note to match binance_klines (perp 2019+/spot 2017+). Commit in research-wiki.
- [ ] 2. adapters.py:
      - [ ] MARKET_OBSERVED_FRED allowlist + REVISED_FRED_RELEASES denylist
      - [ ] `_check_fred_ids(ids, allow_revised)` pure guard (warn on revised; raise under strict if not allowlisted)
      - [ ] `fred_series(..., allow_revised=True)` — default unchanged (no caller breaks); calls guard
      - [ ] `macro_factor_returns(start, include_crypto, crypto_market)` → 8-factor df (+btc/eth)
- [ ] 3. harness.py (ANNOTATE-ONLY — no demotion branch):
      - [ ] MACRO_R2_HI / MACRO_SEL_FLOOR / MACRO_MIN_OBS / MACRO_COVERAGE_FLOOR / MACRO_DEMOTES_FROM consts
      - [ ] `_macro_decomp(search_ret, macro_mx)` — numpy lstsq, R²/residual-Sharpe/betas(diagnostic)/F-pvalue; not_evaluated paths
      - [ ] call site after _beta_decomp (~L646); include_crypto via cost_model.is_crypto(spec.markets)
      - [ ] verdict dict: add `macro_neutral` sub-dict + flattened macro_r2/macro_residual_sharpe. stage1_pass UNTOUCHED.
- [ ] 4. tests/test_macro_neutral.py: orthogonal / confounded / not_evaluated(short) / not_evaluated(coverage) / guard-raises / guard-warns. All synthetic — NO network.
- [ ] 5. Run full test suite — prove no regression + new tests green.
- [ ] 6. Commit crucible. Update task #41, project note.

## Invariant
ANNOTATE-ONLY = no PASSED_ALL_GATES can change. Verify by construction (no demotion branch added) + suite green.

## REVIEW (done 2026-06-15)
- [x] 1. Pre-reg FROZEN (research-wiki 803daa1), §7 stamped, crypto history note corrected pre-stamp.
- [x] 2. adapters.py: MARKET_OBSERVED_FRED + REVISED_FRED_RELEASES + `_check_fred_ids` guard + `fred_series(allow_revised=True)` (default unchanged → zero callers break) + `macro_factor_returns()`.
- [x] 3. harness.py: MACRO_* consts + `_macro_decomp` (numpy lstsq, R²/residual-Sharpe/F-pvalue/diagnostic betas) + call site (crypto-aware, network-guarded) + 3 verdict fields. NO demotion branch → behaviour-neutral.
- [x] 4. tests/test_macro_neutral.py — 10 tests (orthogonal/confounded/2×not_evaluated/unavailable/5×guard).
- [x] 5. Suite: 106 passed, 12 skipped (was 96) — no regression. Verdict-key-stability test green.
- [x] 6. Live smoke: equity block ~100% coverage; orthogonal r2=0.004 survives, confounded p=0.0 flagged; crypto BTC/ETH present.

### Verdict / honesty notes
- Behaviour-neutral confirmed: no demotion code added; verdict gains 3 additive keys; suite + key-stability test green.
- Latency: cold-cache crypto first-verdict pays ~60-90s for the binance BTC/ETH pull (day-cached after; equity ~1s; network-guarded → never blocks/crashes). FOLLOW-UP: pre-warm the factor cache in the forge entrypoint instead of lazily per-verdict (tracked).
- Demotion thresholds (MACRO_R2_HI=0.50 / MACRO_SEL_FLOOR=0.40) are FROZEN but INERT until the §5 calibration on the real corpus confirms them — NOT eyeballed from the smoke numbers.
