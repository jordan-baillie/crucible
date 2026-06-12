# Prompt ports from QuantaAlpha (+ RD-Agent) — reference for tasks #21/#22

Source: /tmp/rdagent-study/QuantaAlpha/quantaalpha/{pipeline/prompts/*.yaml, factors/regulator/consistency_prompts.yaml, factors/coder/prompts.yaml}
Reviewed 2026-06-12. Adapt, don't copy blind — our proposal JSON contract and gate language must stay intact.

## 1. Orthogonal-mutation prompt block (→ task #22, agent/propose.py)

Their operational definition of "orthogonal" is the part worth taking verbatim — it converts a vague
"be different" instruction into 4 checkable axes, and forces a written justification field:

```
"Orthogonal" means the new strategy must:
1. Explore a completely different market hypothesis from the parent
2. Use different data dimensions or feature types
3. Be based on different investment logic or market perspective
4. Produce signals with LOW correlation to the parent's

You will be judged on differentiation, not on resemblance to the parent.
```

Adapted insert for our `mutate_orthogonal(elite)`:
- Keep our context (META-LESSONS, DATA_CATALOG) and full proposal JSON contract.
- Add parent block: hypothesis + frozen construction + verdict metrics + WHY it passed/failed gates
  (their `parent_feedback` slot — we have richer feedback than they do; use the verdict dict).
- Require an extra JSON field: `"orthogonality_reason": "why this is near-independent of the parent
  on data / logic / horizon / market-state axes"` (mirrors their 4-axis check; complements our
  existing why_not_duplicate).
- Their fallback_templates idea (canned directions when LLM parse fails): we don't need it —
  our error path re-tries; canned hypotheses would pollute the queue.

## 2. Crossover prompt block (→ task #21)

Two verbatim-worthy instructions:

```
When fusing strategies:
- identify each parent's strengths AND weaknesses
- AVOID inheriting weaknesses common to both parents
- look for synergy: the hybrid should have a reason to beat BOTH parents, not just average them
```

and their per-parent summary template (we should render one per elite):

```
### Parent {i}
Hypothesis: ...
Construction: ...
Verdict metrics: (DSR, holdout Sharpe, turnover, universes passed)
Gate feedback: (what was strong / what was marginal)
```

Required extra JSON fields for crossover proposals: `"fusion_logic"` and `"expected_benefit_over_parents"`.
Hard rule to add (ours, not theirs): parents MUST come from different family_bucket cells, and the
hybrid keeps ALL our gates (deployability, closed-families, theme cap, data catalog).

## 3. LLM-as-judge orthogonality check (optional pre-enqueue gate)

They run a second cheap call scoring orthogonality 1-10 on 4 axes (data / logic / time-scale /
market-state) with an overall score. Port as an OPTIONAL director check for arm∈{orthogonal, crossover}:
if overall_score < 6 vs the parent(s) → discard before spending a queue slot. This is stronger than
family-bucket dedup for catching reworded siblings. Cost: one extra short LLM call per candidate.
(Defer until #21/#22 land and we see sibling leakage in practice — don't add latency speculatively.)

## 4. Consistency-regulator pattern (already partially ours — note the deltas)

Their regulator checks the chain hypothesis → description → formula → expression with SEVERITY
levels (none/minor/major/critical) and returns a corrected expression. Our codegen consistency-fix
exists but is binary. Worth adopting:
- severity levels (only block on major/critical; stop burning re-gen cycles on window-size nitpicks
  — their explicit rule: "minor window/normalization differences are acceptable")
- returning the corrected construction in the SAME call instead of a separate fix round.

## 5. Error fail→success pair memory (RD-Agent + QuantaAlpha coder prompts)

Their codegen retries inject: (a) the latest failed code + feedback, (b) PAST similar errors with
their eventual successful fixes (fail→success pairs). We have triage producing root-cause records but
codegen retries don't consume them. Future loop (LOOPS 2.2 candidate): on codegen failure, query
triage history for same error class and inject the fixed example. Needs an error-class key in
triage records first.

## What NOT to port
- Their hypothesis_gen "feel free to reuse a similar hypothesis if you agree with it" — directly
  contradicts our dedup discipline.
- RD-Agent's "factors that achieve high IC e.g. machine-learning factors" escalation — IC-chasing
  is the overfit path our gates exist to kill.
- Canned fallback hypotheses on parse failure (queue pollution).
