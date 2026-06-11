# Loops as a Core Framework — Plan
*2026-06-10. Source: 0xCodez/Lev Deviatkin "loop engineering" article (14-step roadmap, itself sourced from Anthropic eng docs + Addy Osmani). Mapped against the Crucible/Atlas estate.*
*2026-06-11: read Osmani's original (x.com/addyosmani/status/2064127981161959567) — confirmed no material delta vs the digest; the digest is a SUPERSET for our purposes (4-condition test, Ralph-Wiggum failure mode, security tax, cost-per-accepted-change KPI are all digest-side additions). One vocabulary nugget kept: harness vs loop layering — the harness is the environment one agent runs inside (= sdk/harness.run_experiment); the loop sits one floor above, on a timer, spawning helpers, feeding itself (= the forge). Also: Claude Code's /goal uses a FRESH model to grade the stop condition — our equivalent is stronger (deterministic statistical gates, not a second LLM opinion).*

## 0. Framing

The article's thesis: leverage moved from *writing prompts* to *designing systems that prompt* — a loop = **automation (heartbeat) + skill (context) + state file (memory) + objective gate (verifier) + subagent split (maker≠checker)**, guarded by hard stops, with a human gate on anything irreversible.

**We already run this.** The nightly forge is a textbook (better-than-textbook) loop:

| Article building block | Forge implementation |
|---|---|
| Automation/heartbeat | systemd timers (03:30 forge, 23:45 forward-paper, 06:30 backup, 07:00 report, weekly lint/BAB) |
| State file | research-wiki (experiments/decisions/patterns) + run_log.jsonl + queue.jsonl + elite pool + FDR registry + write-once holdout ledger |
| Skill / standing spec | wiki AGENTS.md + DATA_CATALOG + codegen CONTRACT (reread every run = the article's anti-goal-drift fix) |
| Objective gate | the gate stack: sandbox → thesis↔code consistency → CPCV/DSR/PBO/FDR/MCPT/beta-confound/holdout. *Not* "a second optimist" — statistical, non-bypassable |
| Maker≠checker | smith codegen vs independent consistency checker vs deterministic verdict code; elite._fitness defense-in-depth |
| Hard stops | --cycles 3, MAX_RETRIES 3, sandbox rlimits + 2700s timeout, claim TTL |
| Connectors | Telegram (alerts/digest/report), GitHub (off-box push), Alpaca (forward-paper) |
| Human gate on irreversible | PASS → human review before capital; plan-approval gates in Atlas |
| Token economics | $0 Max OAuth — we're in the article's "winner" quadrant (condition 3 trivially satisfied) |

So the plan = **close the gaps in existing loops (Phase 1), then promote "loop" to the default unit of work for everything still manual (Phase 2-3), with a meta-loop that watches loop health (Phase 4).**

The 4-condition test, our version (apply before building ANY new loop):
1. Recurs ≥ weekly. 2. Objective machine gate exists (test/backtest/reconciliation — never LLM opinion). 3. $0-Max budget can absorb retries (true until quota; respect the 5h window). 4. The agent has the senior tools (logs, repro env, can run what it writes). **Plus our 5th: anything touching capital, config promotion, or live orders keeps a human approval gate. Non-negotiable.**

---

## Phase 1 — Harden the loops we have (gaps found in the audit)

**1.1 Gate canary (the "gates rot" spot-check, automated).** The article's sharpest warning we don't yet implement: *verify the gate actually catches the failure mode you care about*. We learned this the hard way (I3: regime gates passed vacuously for months). Build `agent/canary.py`: a frozen battery of known-bad strategies — (a) lookahead leak, (b) pure long-only beta clone, (c) overfit grid-mined noise, (d) holdout double-dip attempt — run through the full stack weekly (piggyback crucible-lint.timer). **Every canary must FAIL its designated gate; any canary that PASSES = loud Telegram alert.** This is mutation testing for the gate stack. ~Highest value item in this plan.

**1.2 Cost-per-accepted-change metric.** Article's loop-health KPI. Ours: hypotheses-run per PROMOTE-or-better, fix-retry rate per smith night, and wall-clock per hypothesis — added to the morning report from run_log (schema v2 already carries stages timings + fail_reason; pure aggregation, no new instrumentation). Catches regressions like the Fable-5 "empty code" pattern *quantitatively* (retry rate spike) instead of by eyeball.

**1.3 Ralph-Wiggum sweep of secondary loops.** The forge's stops are solid; audit the *other* timers for soft completion: does forward-paper fail loudly if Alpaca rejects orders? Does backup verify the wiki push actually landed (git ls-remote check) vs "command exited 0"? Does the morning report alert if run_log shows 0 rows (forge silently dead)? One pass, add hard assertions + Telegram-on-violation to each.

**1.4 Security tax pass.** Unattended loops = unattended attack surface. We're decent (sandbox rejects net imports, no community skills auto-installed). Add: (a) secret-scan on the nightly wiki auto-push (the off-box repo must never carry tokens), (b) 30-day permission re-audit reminder as a lint line-item (Alpaca keys scope, GitHub deploy key scope).

## Phase 2 — New loops that pass the 4-condition test (ranked)

**2.1 Runtime-error self-triage loop.** Today's SEP-cache fix was me-in-the-chair; the pattern recurs (fail_reason taxonomy exists precisely because errors repeat). Nightly after forge: collect `fail_reason=runtime_error` rows → debugger subagent gets log_tail + module + SDK source → drafts a fix as a *branch + diff posted to Telegram* (never auto-merge to the SDK — gate = full test suite green + my approval; SDK feeds frozen designs, byte-exactness rules apply). Re-enqueues the casualty hypothesis on merge. Gate: pytest + the equivalence checks. Passes all 4 conditions + the human gate.

**2.2 Data-integrity sentinel.** Daily: Sharadar freshness, SEP cache schema vs source columns (today's bug, generalized), Alpaca account reconciliation (positions file vs broker truth), Atlas portfolio staleness (currently flagged manually in my context — automate it). Gate = pure assertions. Alert-only loop (no writes) → safest possible first new loop; build it first as the template.

**2.3 Forward-paper evidence accumulator.** The val_mom track gate (≥40-50 trades, +ve net expectancy, ≥2 regimes) is currently checked when *I* think of it. Loop: weekly, compute the gate verdict from returns.jsonl + fills, write to wiki `forward/` page, Telegram the trajectory ("23/40 trades, expectancy +4.1bps, on track for ~Aug-15"). State = the wiki page; gate = the pre-registered thresholds. Turns the board's go-live gate into a self-reporting instrument.

**2.4 Dependency/infra bump loop (article's classic).** Weekly: pip-audit + outdated check on crucible/atlas envs, run full test suite in a scratch venv against bumped pins, open a branch + Telegram diff if green. Low value-per-run, but it's the canonical "good first loop" and exercises the branch-PR-gate machinery 2.1 needs.

**2.5 NOT loops (the article's "bad first loops" — keep human-in-chair):** strategy architecture decisions, risk-policy changes, config promotion to live, capital allocation, anything where "done" is a judgment call. Board + human gates stay. The forge generates *hypotheses* inside frozen rails — that's the only place generation runs unattended, and only because the gate stack is statistical.

## Phase 3 — Loops as the default unit of work (process change)

**3.1 The loop registry.** `research-wiki/loops.md`: one row per loop — heartbeat, state location, gate, hard stops, human-gate points, last-canary-pass, owner. The article's "comprehension debt" mitigation applied to loops themselves: if a loop isn't in the registry, it shouldn't be running. Seed with the 7 existing timers.

**3.2 New-work triage rule (added to CEO AGENTS.md routing).** When work arrives: run the 30-second loop check. Recurring + machine-gateable → build/extend a loop (small: one automation, one skill-file, one state file, one gate — in that order, manual run reliable FIRST, then schedule). One-off or judgment-call → do it directly, as now.

**3.3 Read-the-diffs discipline (comprehension-debt control).** Morning report already surfaces near-misses; add: every SDK/rails diff produced by any loop (2.1, 2.4) lands as Telegram diff + requires explicit ack; monthly "comprehension audit" — I pick one loop-produced artifact at random and explain it in the journal. If I can't, the loop gets narrowed.

## Phase 4 — The meta-loop (watch the watchers)

**4.1 Loop-health report.** Weekly (lint timer): for each registry loop — last run time vs expected heartbeat, exit status, gate-canary status (1.1), cost-per-accepted-change trend (1.2), state-file growth (wiki bloat = the selective-forgetting problem). One Telegram line per unhealthy loop; silence = healthy. The morning report covers *research output*; this covers *loop infrastructure*.

---

## Sequencing & effort
1. **1.1 gate canary** + **2.2 data sentinel** first (highest safety value; sentinel is the new-loop template). ~1 session.
2. **1.2 metrics** + **1.3 Ralph-Wiggum sweep** + **3.1 registry**. ~1 session.
3. **2.3 forward-paper accumulator** (directly serves the Phase-4 capital gate). ~1 session.
4. **2.1 self-triage** + **2.4 dep bump** (need the branch-PR-gate machinery). ~1-2 sessions.
5. **3.2/3.3/4.1** process items ride along.

All on $0 Max OAuth (--system-prompt rule applies to every loop's LLM calls). Mind the June-22 Fable revert — loops must read the model policy, never hardcode (already true for forge; enforce in every new loop).
