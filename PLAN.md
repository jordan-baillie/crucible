# Hephaestus — Industrialized Autonomous Strategy Research

**Goal:** automate the loop we ran by hand this session (hypothesis → Gate-0 → pre-register → build → run through the rails → verdict), coordinated through a **shared LLM-maintained wiki** that compounds all learnings, with a **Telegram alert only when a strategy passes ALL gates**. One agent now; designed to scale to N.

**Why this is the right thing to build:** the board concluded the portfolio's genuine product is the *research-integrity methodology + accumulated negative knowledge*, not any single alpha. Hephaestus **operationalizes exactly that asset** — it turns our hard-won discipline into an autonomous, compounding, $0-marginal-cost machine. Low-ops once built; free compute (Claude Max OAuth).

---

## The 3 components

### 1. The Shared Research Wiki (karpathy / llm-wiki-agent pattern)
A git repo of LLM-owned markdown at **`/root/research-wiki/`** — the coordination substrate + compounding memory for ALL agents and projects. Adapted from the generic pattern to *our* domain (strategy research, not papers).

```
research-wiki/
├── AGENTS.md            schema: how agents read/write/maintain the wiki (the config)
├── index.md            catalog of every page (updated every experiment)
├── log.md              append-only chronicle: "## [date] experiment|decision|ingest | <title>"
├── overview.md         LIVING synthesis: the current portfolio thesis + what's validated/dead
├── experiments/        ONE page per tested hypothesis (pre-reg + frozen design + verdict + metrics)
│                       — the FDR registry made human-readable + interlinked
├── premia/             concept pages: carry, trend, value, VRP, momentum, basis... priors + cross-market results
├── markets/            entity pages: crypto, equities, futures, sports, FX... data sources, what works/fails
├── patterns/           CONFIRMED patterns + ANTI-patterns ("standalone prediction dies", "diversification IS the edge", "DM carry dead", "meta-overfitting trap", "verify-data-before-build")
├── decisions/          closed board decisions + kill verdicts (so agents NEVER re-open them)
├── methodology/        the rails, the gates, pre-registration discipline (the HOW)
└── graph/              optional knowledge graph (wikilinks → graph.html)
```
- **Reads** before generating (avoid re-testing, use priors, respect anti-patterns). **Writes** after every experiment (record result, update premia/markets/patterns, flag contradictions).
- `index.md` + `log.md` are the navigation spine (grep-able; no embedding RAG needed at our scale).
- A small **search CLI** (BM25 over the markdown) added in Phase 2 when it grows.

### 2. The Autonomous Research Agent (the loop)
Generalizes Atlas's paused `AUTOMATED_LOOP_SPEC` (director/runner/discovery — ~80% built) to cross-asset/cross-project research. A systemd-scheduled `pi` agent (Claude Max OAuth, $0) running:

| step | what it does | reuses |
|---|---|---|
| **a. Read wiki** | load overview + experiments index + open priors + anti-patterns | wiki |
| **b. Generate** | propose ONE *untested* hypothesis — economically-motivated (risk premium / structural / **combination of validated legs**), data-feasible, not in `experiments/`, not in `decisions/` | pi CLI |
| **c. Gate-0** | auto data-feasibility probe (free? coverage? point-in-time?) — KILL if infeasible | our Gate-0 pattern |
| **d. Pre-register** | freeze the design → `experiments/<id>.md` BEFORE running | discipline |
| **e. Build+run** | fill a **strategy TEMPLATE** (signal fn + data spec), harness runs it through `research_integrity` (CPCV/DSR/PBO + write-once holdout + deployment-sanity + FDR bar) | rails + Strategy SDK (below) |
| **f. Verdict+record** | evaluate vs frozen gates; write verdict to wiki + FDR registry + `log.md` | rails |
| **g. Gate** | **if PASS ALL gates → Telegram alert** (human reviews before ANY capital) | telegram |

### 3. Telegram alerts + multi-agent scaling
- **Alert on full-gate-pass only** (rare by design): "🟢 PASSED ALL GATES: <name> | DSR x, holdout PASS, deploy x, tier PROMOTE — review experiments/<id>." Uses existing `telegram_bot_token` + telegram-compose skill. Optional daily digest + heartbeat.
- **Scale to N agents** (designed now, 1 first): shared **hypothesis queue** + `atlas_state` **locks** (no collisions) + the **shared FDR registry** in the wiki. The wiki IS the coordination layer — agents see each other's results. A meta "director" sets priorities + the human reviews passes.

---

## The hard problem (and the solution): automating build+run
This session I wrote + debugged each pipeline by hand. To automate reliably, the agent must NOT write the whole pipeline each time. Solution = a **Strategy SDK / template** so the agent only writes the *novel* part:
- A fixed **harness** (`hephaestus/sdk/`) that owns: data ingest adapters (yfinance/FRED/Binance-Vision/Sharadar already built), walk-forward split, the rails wiring, holdout, tail/combination analysis, verdict, wiki-write, Telegram. (We already built all of these this session in Boreas/Midas/Hermes — extract + generalize them.)
- The agent generates ONLY: `signal(panel, params) -> (daily_returns, trades)` + a `data_spec` + `pre_registration`. Everything else is the fixed, tested harness.
- Sandboxed execution + error-recovery loop (agent reads the traceback, fixes the signal fn, retries — bounded retries).
- This makes generation tractable AND keeps the rails non-bypassable (the harness, not the agent, runs the gates).

---

## Critical discipline — what makes autonomous search SAFE
Free unlimited LLM search against a FIXED gate → a guaranteed overfitting machine. Against the rails it's safe — and this is the WHOLE point:
1. **Write-once holdout** (single-use ledger) — the only incorruptible arbiter; the harness owns it, the agent can't peek.
2. **FDR-aware promote bar that RISES with every experiment** (`promote_dsr(n_families)`) — and is **SHARED across all agents** (N parallel searchers multiply false-discovery risk; the shared registry is what prevents it). This is non-negotiable for multi-agent.
3. **Deployment-sanity** auto-FAIL (no 1-2 name mirages).
4. **No autonomous capital, ever** — PASS only triggers a Telegram alert; a human + the charter gates decide deployment.
5. **Anti-patterns in the wiki** stop re-testing closed sets (meta-overfitting) — the agent reads `patterns/` + `decisions/` before generating.

---

## Phased build plan

**Phase 0 — Foundation (the wiki + migrate all knowledge).** Stand up `/root/research-wiki/` (structure + `AGENTS.md` schema). **Migrate ALL existing learnings** into it: Atlas `memory/SUMMARY.md` + brain/ + hypotheses; Midas notes + carry/trend findings; Hermes retirement; Cronus 48 edges; Boreas pre-regs + the carry+trend validation; the CEO journal; every board memo; the research_integrity methodology. Output: a queryable, structured knowledge base. *(Highest value, lowest risk — do first regardless.)*

**Phase 1 — Single-agent, human-in-the-loop.** Build the Strategy SDK (extract the harness from this session's code). Agent: read wiki → generate hypothesis → **human approves** → build+run through rails → record to wiki → Telegram on pass. Proves the loop end-to-end with a human gate on generation.

**Phase 2 — Single-agent autonomy.** Agent runs generate→Gate-0→pre-reg→build→run→record fully autonomously; only a full-gate PASS requires human review. systemd-scheduled (nightly, bounded cores, nice'd, never starves live ops). Telegram alerts + daily digest + heartbeat. Add the wiki search CLI.

**Phase 3 — Multi-agent scale-out.** Shared hypothesis queue + `atlas_state` locks + shared cross-agent FDR registry + wiki as coordination. Scale 1→N. A director role sets research priorities.

---

## Risks & safeguards
- **Overfitting at scale** → shared FDR bar + write-once holdout + forward confirmation (the rails); this is solved-by-design.
- **Agent writes bad/unsafe code** → sandboxed exec, fixed harness owns the gates, write scope limited (no capital/config/live paths), bounded retries.
- **Wiki rot/contradiction** → periodic `lint` pass (orphans, stale claims, contradictions) per the karpathy pattern.
- **Cost** → Claude Max OAuth ($0 marginal); compute nice'd/bounded.
- **False alerts** → alert ONLY on full-gate-pass (rails make this rare + meaningful).
- **Auth** → every `pi`/`claude` subprocess MUST include `--system-prompt` (Claude Max routing, per /root/AGENTS.md).

## Naming
Codename **Hephaestus** (the forge god — hammers out + tests tools). The wiki = the shared memory; the agents = the smiths; the rails = the quality gate; Telegram = the bell that rings only when something real comes off the anvil.
