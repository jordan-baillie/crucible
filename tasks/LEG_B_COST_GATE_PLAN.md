# Leg B — Cost-Aware Deployability Gate (board 2026-06-15, task #38)

**Goal:** replace the untested flat-8bps frictionless cost assumption with a liquidity-and-borrow-aware
cost model + a non-bypassable deployability pre-filter, so an "un-deployable PASS" becomes structurally
unrepresentable (First Principle). Re-score the 96-corpus to separate **calibration** (fixable in code)
from **physics** (structural), and produce the honest survivor ledger that feeds Leg D.

**Discipline:** pre-registration is sacred. Freeze the spec BEFORE building the gate change or running the
re-score. No tuning ladder levels to make any strategy survive. Report sensitivity bands, never a single
favorable number. The contaminated live-fill slippage data is NOT a calibration target.

---

## Phase 0 — Recon (DONE 2026-06-15)
- Cost model entry point: `sdk/signal_kit.net_of_cost(W, rets, cost_bps=8.0)` — flat, per-unit-turnover,
  applied INSIDE each strategy. The kit is CONTRACT-mandated → re-pricing the kit re-prices most of the corpus.
- Corpus not stored as ledgers → re-score = re-run each `signal()` (111 modules exist) with a cost override.
- Data present: `sep_panel` carries close×volume (→ ADV / dollar-volume per name-date);
  `atlas/data/cache/alpaca_tradable_assets.json` carries the shortable set (5155/13029).
- **FINDING (reshapes the plan):** live val_mom slippage is measured vs STALE IEX decision prices
  (median 146bps, max 1795bps) — contaminated, n≈1 book, pre-OPG. NOT a usable calibration target.
  Amihud borrow evidence is clean + decisive (92% order cancellation, 47 short-sells killed on borrow).

## Phase 1 — PRE-REGISTRATION (frozen spec) — `research-wiki/methodology/prereg-cost-aware-deployability-gate.md`
- Component 1 (borrow-feasibility filter): HIGH confidence, clean data now.
- Component 2 (liquidity slippage model): microstructure-PRIOR dollar-volume ladder; live-fill ±20%
  validation DEFERRED behind clean fills; provisional; sensitivity band.
- Deployability pre-filter integration (non-bypassable, demotes PASSED_ALL like beta-confound).
- Re-score protocol (deterministic; new artifact `cost_rescore.jsonl`, NOT the production registry).
- Falsification / calibration-vs-physics decision rules.
- **GATE: operator review of the frozen spec before the re-score (keystone, irreversible result).**

## Phase 2 — Clean slippage measurement (prerequisite for Component 2 validation)
- Fix `atlas/execution/record_fills` to measure slippage vs OFFICIAL OPEN, not stale IEX decision_px
  (journal-flagged 2026-06-11). Keep val_mom + amihud shadow running as the $0 clean-fill accumulator.
- Target: ≥100 clean post-OPG steady-state fills across ≥2 books before validating the ladder.

## Phase 3 — Build the cost model — `sdk/cost_model.py`
- `liquidity_cost_bps(panel, tickers, dates)` — ADV-decile ladder from priors (frozen levels).
- `borrow_feasible(tickers)` — shortable-set membership (+ point-in-time caveat).
- Validation harness: predict clean val_mom slippage; report central vs realized (±20% check) when clean
  fills exist. Until then: PROVISIONAL.

## Phase 4 — Bake into the gate (non-bypassable)
- `research_integrity.deployment` (canonical + vendored, byte-identical) gains a cost/borrow deployability
  filter; harness demotes PASSED_ALL on failure. RED-first test (a known un-deployable book must fail).

## Phase 5 — Re-score the 96-corpus (the settling experiment + Leg D artifact)
- Deterministic re-run with the frozen cost model; survivor count at central AND conservative ladders;
  death-cause breakdown (borrow / liquidity / other). Publish the honest ledger.
- State the calibration-vs-physics verdict with its confidence.

## Falsifiable success (4wk, ~2026-07-13)
- Borrow filter shipped + re-score complete + survivor ledger published.
- Clean-fill slippage measurement shipped + accumulating.
- Calibration-vs-physics verdict stated honestly with confidence bounds.
