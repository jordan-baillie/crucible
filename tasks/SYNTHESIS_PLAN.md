# Crucible Synthesis Plan — research findings → one clean system

Date: 2026-06-12. Inputs: competitive-landscape research + source dives (RD-Agent(Q), QuantaAlpha,
QuantEvolve at /tmp/rdagent-study) + prompt-port spec (tasks/prompt-ports-quantaalpha.md).

## Design principle

We do NOT bolt findings on individually. Each subsystem is brought to its *intended final shape*
once, replacing accreted patches where a principled mechanism subsumes them. Deep structural change
only where it deletes complexity (Stage 1) or completes a missing lifecycle leg (Stage 3).

The lifecycle: GENERATE → GATE → DEPLOY → MONITOR → (ALLOCATE).
Findings map to: Stage 1 = generation, Stage 2 = gate, Stage 3 = monitor/exit, Stage 4 = loop hygiene.

---

## Stage 1 — One Evolutionary Core (deep structural; REPLACES accreted machinery)

Today's exploit machinery is three reactive patches (top-K pool, family-downweighted sampling,
2/theme queue cap) written after the value×mom-hammering incident. Replace the pool+weights with a
single quality-diversity design; keep the queue-level theme cap (different layer, still useful).

**1a. `agent/elite.py` → MAP-Elites grid** (from QuantEvolve `database.py`)
- Cell = family_bucket × turnover-band(low/med/high) × universe(smallcap/largecap/etf/multi).
- Best-per-cell, fitness=DSR unchanged, beta-confound→0 unchanged, closed-family ban unchanged.
- DELETE family-downweighted sampling (subsumed: grid structure guarantees what weights approximated).
- sample() = uniform over occupied cells, then the cell's occupant. sample_pair() = two distinct cells,
  different families (for crossover).
- Migrate existing elite.jsonl (re-bin entries; collisions keep higher DSR).

**1b. `agent/propose.py` → one operator set, four arms**
- explore() (unchanged), refine(elite) (today's mutate, unchanged), NEW orthogonal(elite),
  NEW crossover(eliteA, eliteB).
- Prompts per tasks/prompt-ports-quantaalpha.md §1–2 (4-axis orthogonality + orthogonality_reason
  field; crossover per-parent summaries + fusion_logic + must-beat-both-parents; parents from
  different families is a HARD precondition enforced by sample_pair(), not the prompt).
- All arms emit the SAME proposal JSON contract; all existing director gates (deployability,
  closed-family, theme cap, dedup) apply identically.

**1c. `agent/director.py` → single arm-selection point + uniform arm/reward logging**
- Fixed split: explore 45 / refine 25 / orthogonal 15 / crossover 15 (crossover+orthogonal require
  pool preconditions; fall back to explore when unmet).
- Every queue item carries `arm`; run_worker records arm + verdict-derived reward into run_log.jsonl.
  This is the dataset for the future bandit — we log NOW, fit LATER (parked, N≥60).

Acceptance: pool can never hold 2 entries in one cell; arms recorded end-to-end; tests for
record/sample/sample_pair/migration; one nightly forge run observed clean before stage close.
Subsumes former tasks #20/#21/#22.

## Stage 2 — Gate stack: regime burner + memorization invariant (small, additive; pre-registered)

**2a. Regime-robustness burner.** Pre-register THEN implement: split the search window by trailing
~200d realized-vol median into calm/turbulent halves; the candidate's edge must be non-negative in
BOTH halves. Runs INSIDE the search phase (before holdout unlock) — costs zero extra holdout looks.
Placement and exact rule frozen in the wiki before the code lands. One-off retro AUDIT (report-only)
on deployed strategies + elite pool; no retroactive kills without human review.

**2b. Memorization-immunity invariant (documentation, not machinery).** Wiki patterns entry: WHY the
crucible is structurally immune to the "profit mirage" failure mode (smiths emit hypotheses/code,
never security-level judgments; gates test on raw price data; holdout is post-cutoff and write-once).
Define the blindfold-check TRIGGER: required only if a future hypothesis embeds issuer-specific LLM
views (e.g. "LLM rates company quality") — then re-run with anonymized tickers before gates.

## Stage 3 — Lifecycle completion: evidence → decay → retire (structural; completes the missing leg)

Deployed strategies currently have an entry path (shadow + evidence accumulation, forward/evidence.py
G1–G7) but NO exit path. Alpha decays hyperbolically (α(t)=K/(1+λt)); a forge that only ever adds
books eventually runs a graveyard. Completion, not addition:

**3a. Explicit lifecycle states** in the deployed-strategy registry: shadow → evidence →
(real_capital_candidate | decaying) → retired. evidence.py owns transitions; states surface in the
daily digest.

**3b. Decay tracker**: rolling live-vs-modeled comparison (realized rolling-60d mean return & IC vs
the frozen holdout expectation already stored at deploy time) + a slow-drift detector (CUSUM-style)
so gradual decay is caught, not just acute divergence (which track-vs-expectation already flags).

**3c. PRE-REGISTERED retirement rule** (frozen in wiki before implementation, like every gate):
e.g. rolling-60d realized mean < 25% of modeled mean for 2 consecutive windows AND drift detector
fired → state=decaying + Telegram; human confirms retirement. No auto-liquidation.

This defines the input contract for Stage D (#7 capital allocator: allocate over strategies in
evidence/candidate states, never decaying/retired). #7 itself stays board-deferred (AUM gate).

## Stage 4 — Loop hygiene: codegen quality (small, opportunistic)

**4a. Severity levels in the consistency check** (QuantaAlpha regulator pattern): none/minor/major/
critical; only major+ triggers regeneration ("minor window/normalization differences acceptable");
corrected construction returned in the SAME call. Directly attacks the slow consistency-fix tail
(median 347s vs 211s clean) that drove the 900s-timeout fix.

**4b. Fail→success error-pair memory (LOOPS 2.2).** Add error-class key to triage records; codegen
retry queries triage history for the same class and injects the past fail→success pair. Connects two
loops we already run (triage ↔ codegen) instead of building anything new.

## Parked — explicit non-goals (revisit triggers stated)

- Thompson bandit over arms: fit only when run_log has ≥60 arm-labelled outcomes (Stage 1c logs them).
- LLM-judge orthogonality pre-enqueue gate: only if reworded-sibling leakage past family buckets is
  actually observed in queue audits.
- Island populations: pointless at 3 smiths; revisit if smith count ≥8.
- Diversified planning init (QuantaAlpha): scout + queue + theme cap already cover it.
- Any further gate additions: the stack is frozen except 2a. Rigor is the moat; churn erodes it.
- Blindfold check as standing machinery: trigger-based only (see 2b).

## Sequencing & rationale

1 → 2 → 3 → 4. Stage 1 first: generation diversity compounds nightly — every night of redundant
candidates spends shared FDR budget (the scarcest resource) on near-duplicates. Stage 2 second:
smallest diff, protects everything downstream; new gate applies going forward (no retroactive kills).
Stage 3 third: val_mom/tranched_v3 evidence is accumulating regardless; decay matters on a weeks
horizon. Stage 4 opportunistic — can interleave any time, touches no contracts.

Each stage = one reviewable unit (plan → implement → tests → one observed nightly run → close).
No stage starts before the previous one's forge run is observed clean.
