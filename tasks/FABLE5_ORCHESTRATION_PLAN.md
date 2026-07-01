# Fable-5 Orchestration Plan — a cost-routed scout/planner tier, not a rails rewrite

*2026-07-01. Trigger: Fable 5 redeployed globally today (US export controls lifted 2026-06-30;
`anthropic.com/news/redeploying-fable-5`). Goal: re-introduce Fable 5 to discover profitable,
**passing** strategies faster — by making the **search** smarter, never by touching the frozen judge.*

> Method note: the research + three independent design drafts + an adversarial invariant-critique
> behind this plan were produced by a fan-out agent workflow and then re-verified against the code.
> Every load-bearing fact below (`model-policy.json` absent, `summon` flags unverifiable here, the
> bandit constants, the Atlas read seam) was checked in this repo, not taken on faith.

---

## 0. Honest grounding on the premise (read this first)

The ask says Fable 5 is *"exceptional at agent orchestration."* That superlative is **not grounded**.
What Anthropic's primary docs (model-intro, the *Prompting Claude Fable 5* guide, the Bedrock card)
actually support:

- **Real model** `claude-fable-5`, launched 2026-06-09. 1M context, up to 128K output, adaptive-thinking-
  only, Jan-2026 cutoff. List price **$10/M in, $50/M out** — **2× Opus 4.8** ($5/$25), ~3× Sonnet 5.
- Built *"for the most demanding reasoning and long-horizon agentic work."* Turns *"run for many
  minutes"*; autonomous runs *"can extend for hours."* (This is exactly why `agent/scout.py` and
  `agent/llm.py` already carry a **900 s** timeout — that comment is grounded; keep it.)
- *"Significantly more dependable at dispatching and sustaining **parallel subagents**… prefer
  asynchronous communication over blocking,"* and the guide advises restructuring harnesses to
  **check runs asynchronously via scheduled jobs** — which maps onto the forge's existing nightly/drip
  cadence.
- **Integration cost:** refusal rate *"materially higher than previous Claude models"* — refusals come
  back as HTTP 200 `stop_reason:'refusal'` across cyber / life-sciences / **reasoning_extraction**
  categories (the last is tripped by prompts that ask the model to echo its own reasoning). Anthropic
  recommends an **Opus-4.8 fallback**. Mandatory **30-day data retention** (Covered Model, no ZDR).

What is **asserted but not grounded**, and must not be bet on:

- *"Exceptional / best-in-class orchestration"* — the word appears nowhere in Anthropic's materials;
  every claim is **comparative vs Opus 4.8**, with no head-to-head vs competitors. Independent
  (third-party, unverified) benchmarks put Fable 5 ahead mainly on the **hardest long-horizon** tasks
  while **Opus 4.8 is competitive-or-better on many everyday coding/tool tasks**. → *escalate to Fable 5
  for the rare long-horizon synthesis job; keep cheaper models as the default.*
- The **redeployment announcement** the ask was framed on is **entirely safety/export-controls** and
  explicitly says Fable 5 *"provides no such unique offensive capabilities."* It has **zero**
  orchestration content and cannot be cited for this thesis. (The safeguard blocks the reported
  jailbreak **93%** of the time per CNBC/Axios — not the "99%" sometimes quoted.)
- Cost window: *"included for up to 50% of weekly usage limits through July 7"* is **provisional**
  (third-party dates conflict; KYC is being added). Today is 2026-07-01 → **paid credits are imminent.**

**Verdict.** The *direction* — Fable 5 as a **scout / night-planner tier**, escalated for the rare
long-horizon job — is defensible. The superlative is not. So this plan uses Fable 5 **only to re-weight
which families the deterministic search attacks — never what passes or deploys.**

---

## 1. The core insight (where orchestration is safe vs forbidden)

**The scarcest resource is the shared FDR bar**, which rises with *every family ever tested*
(`vendor/research_integrity`). The dominant waste is spending a backtest+holdout look on a **dupe or a
near-miss of something already burned — or already deployed**. A re-proposed known factor either dies at
`director` dedup (a spent LLM call) or clears dedup and fails a bar it can no longer beat (a spent look).
So "faster discovery" = **more distinct-real families per FDR look**, *not* more raw idea throughput. The
SOTA literature agrees: RD-Agent(Q)'s ablation shows a Thompson bandit **beats an LLM-as-scheduler**
(extra LLM calls cut iteration count), and the *Agentic Trading* survey (19 studies; 0/19 reproducible)
finds **no evidence** multi-agent swarms beat simpler baselines on risk-adjusted return. A researcher/
coder/critic swarm here would be snake-oil that just **burns the FDR bar faster**.

**SAFE + additive (the generation lane, upstream of every gate):**
- The **scout** (`agent/scout.py`) is the *only* step that injects external orthogonality, and it runs
  **tool-lessly today** — two `--no-tools` calls hand-stitch pre-fetched Brave/Firecrawl/FinTwit blocks.
  The model cannot chase a citation, read the actual paper, or check whether a mechanism is
  already-published *before* proposing it. That is precisely the long-horizon, multi-tool job Fable 5 is
  documented for — and it is the *"future agentic X-gather scout"* that `mcp/README.md` already names as
  a deliberate follow-on.
- The scout's entire output contract is **append-lines to `WIKI/candidates.md`** (consumed by
  `propose._read_tail`, attributed via the measured `seeded_by` field). Everything downstream —
  `director.top_up`'s dedup / `elite._closed_families()` / theme-cap / `retail_tradable_5k` gate, then
  the full non-bypassable stack — is unchanged. A scout upgrade is a **pure swap behind a frozen
  interface**: better candidates in, same gates out.
- **"summon sees both crucible AND atlas"** needs *zero new machinery*: the forge already reads
  research-side state (`forge_state.build()`) and execution-side state
  (`morning_report.forward_paper_section()` → `<atlas>/config/live_strategies.json` → per-book
  `returns.jsonl`/`book.json`/`runs.jsonl`). Feeding the scout *"what's already burned + what's already
  live"* steers new candidates into **unspent** family-space.

**FORBIDDEN by the rails (do not go here):**
- **Making `codegen` agentic.** `config.llm_cmd()` keeps `--no-tools` byte-for-byte — the comment
  documents that tools once made codegen *"run the entire backtest in a bash tool loop until timeout →
  crash, plus ~2× compute"* (the issuance-factor run died this way). **The scout is safe to make agentic
  precisely because it emits candidate *text* and has no strategy module to execute; codegen is not.**
- Weakening/bypassing/resetting the **gate stack, FDR bar, or write-once holdout**.
- A **second uncontrolled LLM seam** — all new LLM code lives in `agent/config.py` and reuses
  `_policy_model()` / `_thinking_args()`.
- **Cross-repo Python imports** — the Atlas coupling stays file-shaped, read via existing callers only.
- **Autonomous capital** — PASS → paper book only; real money stays human-gated.

> **Rails-change flag (AGENTS.md):** `agent/config.py::llm_cmd()` is the frozen **model seam**. Stage 2
> below adds a *sibling* builder to that file; per AGENTS.md this is a rails-adjacent change and
> **requires explicit human sign-off**, not a silent edit. This document is the flag.

---

## 2. Verified preconditions (checked in-repo 2026-07-01)

| Fact | Status | Consequence |
|---|---|---|
| `/root/.pi/model-policy.json` (`crucible_paths.MODEL_POLICY`) exists? | **NO** (dir absent too) | `_policy_model()` fails safe to `claude-opus-4-8` for **everything** → all stages ship **dark at $0**; Fable 5 is opted in by adding **one JSON key**. |
| `summon` / `pi` on PATH in this box? | **NO** | The Stage-2 flags (`--mcp-config`, `--max-turns`, tool-allowlist) are **unverifiable here** → verifying them on the prod box is a hard Stage-2 prerequisite. |
| `agent/bandit.py` constants | `EXPLORE_FLOOR=0.25`, `ARM_EPS=0.03`, `N_MIN=60`, `FIXED_SPLIT`, `_apply_floors`, `arm_weights` | A planner `arm_bias` must **modulate** these, never replace them. |
| `morning_report.forward_paper_section()` Atlas read | `live_strategies.json` → `runs/returns.jsonl` + `book.json`; **None-tolerant** when `DEPLOY_TARGET` unset | Reuse this exact pattern; degrade to research-only when deploy is disabled. |
| `director.top_up` scout gate | `if random.random() < 0.4` | Agentic-scout Fable-5 spend is **O(nights)**, gated at ~40% of top-ups — not O(strategies). |

---

## 3. Staged levers (ranked by value-per-effort × invariant-safety)

Everything ships **dark at $0 first**. `[ADOPT]` = clean; `[ADOPT*]` = adopt-with-blocking-changes.

### Stage 0 — Joint-truth reader + diversity brief  ·  `[ADOPT]` · **S** · $0, no LLM
- **What:** ONE pure-function reader emitting `joint_state.json` (schema v1, atomic tmp-replace).
  Research side reuses `forge_state.build()` (closed/heavily-tested families, occupied elite cells via
  `elite.top()`/`elite._closed_families()`, FDR bar). Execution side reuses the **exact**
  `morning_report.forward_paper_section()` pattern. A few-hundred-char steering brief derived from that
  one artifact — *"these families are closed / heavily-tested / already-deployed → propose ORTHOGONAL
  premia and DIFFERENT markets"* — is injected into the scout/propose context.
- **Files:** `agent/joint_state.py` (or extend `forge_state.py`; piggyback its existing invocation).
  **Build ONE reader, not two** — the two design drafts overlapped here; a parallel reader would violate
  subtract-before-you-add.
- **Invariant safety:** read-only over frozen file shapes; **zero cross-repo imports**; biases what is
  *generated*, never what passes/deploys.
- **Speed (FDR terms):** naming the already-spent/already-live family set up front concentrates
  candidates in **unspent** regions — the single most direct lever on discoveries-per-look, and it helps
  **even with the tool-less scout**. This is the "summon sees both sides" close-the-loop piece.
- **Fable-5 routing / cost:** none. $0. **Ship this first.**

### Stage 1 — `scout` tier in `MODEL_POLICY` + docs  ·  `[ADOPT]` · **S** · $0 until a key is added
- **What:** create `/root/.pi/model-policy.json` with the **full** `frontier`/`standard`/`fast` set (all
  → the current $0 failsafe) **plus** a `scout` tier (and optional `planner` tier) → `claude-fable-5`.
  Add the one-line tier read in Stage 2. Document `FORGE_SCOUT_MODEL`, `FORGE_SCOUT_TIER`, `SCOUT_AGENTIC`
  next to the existing `FORGE_MODEL`/`MODEL_POLICY` rows in `README.md`/`AGENTS.md`.
- **Invariant safety:** uses the **existing** tiers seam ("no new machinery to route models"). Pure config.
- **⚠ Load-bearing landmine:** when you create the file you must write the `frontier` tier too —
  **do NOT point `frontier` at Fable 5.** That silently moves the entire **O(strategies)** propose/codegen
  load onto paid credits (worst-case blast radius). One-line JSON review discipline is the control.
- **Fable-5 routing / cost:** **this is the control surface.** Absent the key ⇒ $0 failsafe. Present ⇒
  only the scout role pays.

### Stage 2 — `scout_cmd()` sibling in `config.py` (tools ON, MCP wired, turn-capped)  ·  `[ADOPT*]` · **S** · $0 by default
- **What:** a sibling to `llm_cmd()` returning the summon invocation **without** `--no-tools`, **with**
  `--mcp-config` at the existing `crucible-research` MCP (`x_search`/`web_search`/`research_search`/
  `scrape_url`/`extract_url`), a **read-only tool allowlist**, a **`--max-turns` cap**, and
  `MODEL = FORGE_SCOUT_MODEL env or _policy_model(tier='scout', failsafe=MODEL)`. `llm_cmd()` is
  byte-for-byte unchanged.
- **Files:** `agent/config.py` only (the single seam) — **rails-adjacent, human sign-off required.**
- **Invariant safety:** `llm_cmd()` stays `--no-tools` (Inv2/Inv4); the scout only **reads**; the
  codegen-bash-loop failure mode **cannot recur** (no strategy module to execute; turn-capped).
- **Blocking changes:**
  1. **VERIFY the summon flags are real** on the prod box (`summon --help`). `summon` is **not installed
     here**, so the *"turn-capped, cannot loop unboundedly"* guarantee — the exact thing preventing a
     repeat of the timeout crash — is **asserted, not proven.** This is the #1 prerequisite.
  2. Wire the **Opus-4.8 fallback** and treat `stop_reason:'refusal'` like `LLMError` (**fail loud**).
  3. Keep opt-in: absent the tier/env, run on the $0 failsafe.
- **Cost:** confines Fable 5 to the scout role; fires at ~40% of top-ups (O(nights)).

### Stage 3 — Agentic scout turn behind the unchanged `candidates.md` contract  ·  `[ADOPT*]` · **M** · default OFF
- **What:** in `agent/scout.py`, add a path (behind `SCOUT_AGENTIC`) that replaces the manual
  `_brave`/`_research`/`_fintwit`/`_deep_dive` stitching with ONE Fable-5 turn via `scout_cmd()`: same
  wiki context **+ the Stage-0 brief**, instruct it to search/read/cross-check via the MCP, and return
  the **identical** distill JSON (`{summary, candidates[...], premia_updates, contradictions}`).
  `_ingest()`, `candidates.md`, the sources page, `log.md` all unchanged.
- **Invariant safety:** reuses `_ingest` + the existing **fail-loud `LLMError`** discipline; scout still
  cannot promote; no code execution, no capital.
- **Blocking changes:**
  1. **Default OFF** — the tool-less two-call path stays bit-for-bit the default.
  2. **Preserve fail-loud** — parse/timeout/**refusal** must RAISE, never coerce to `candidates:[]`
     (this is the 2026-06-22 silent-0-candidate-storm lesson already baked into `scout.py`; a Fable-5
     200-refusal is a *new* way to trigger it).
  3. Ship **only after** Stages 0–2.
  4. **Gate acceptance on a measured distinct-novel-family lift via `seeded_by`**, not vibes — kill it
     if it just re-proposes near-misses.
- **Speed (FDR terms):** one agentic turn can follow the citation trail (search → `scrape_url`/
  `extract_url` the real arXiv/SSRN methodology → confirm distinctness vs the wiki/brief) **before**
  emitting a candidate — raising novel-family yield per candidate.
- **Cost:** one multi-tool Fable-5 turn per scout invocation (~40% of top-ups); MCP backends unchanged
  (twitterapi.io sub-cent, Firecrawl ~8 credits/run). `SCOUT_AGENTIC=0` reverts to $0 instantly.

### Stage 4 (optional) — Tool-less Fable-5 night-planner  ·  `[ADOPT*]` · **M** · DEFER until 2/3 prove value
- **What:** ONE `--no-tools` Fable-5 call via the **existing** `config.llm_cmd()` path, consuming
  `joint_state.json` and emitting an advisory `arm_bias` JSON hint that `director.top_up` blends into
  the bandit weights before sampling.
- **Invariant safety:** reuses the existing seam verbatim (no summon-flag risk, no second seam); emits a
  research-lane **bias only** — cannot touch gate/holdout/FDR/capital.
- **Blocking changes:**
  1. `arm_bias` must **MODULATE, not replace**, the data-driven Thompson bandit — re-apply
     `bandit._apply_floors` **after** the blend so `EXPLORE_FLOOR=0.25` + `ARM_EPS` always hold (the
     plan can never zero an arm). Replacing the bandit regresses an empirically-fit allocation into LLM
     vibes — RD-Agent(Q)'s ablation says that is *worse*.
  2. Strictly advisory + idempotent: missing/garbled/stale plan → degrade to pure bandit, never block a
     night (same discipline as the forge pre-flight healthcheck).
  3. **One Fable-5 orchestration call per night** — planner *or* agentic-scout, not both — until measured
     value justifies two.

**Deferred / rejected:** any second LLM path outside `config.py`; any per-smith or codegen Fable-5 use;
any Mythos-5 use (Glasswing-only); an agentic **fleet-scaling actuator** (keep fleet size human/systemd-
gated, hard-capped `[3..6]`); anything touching the gate stack, FDR registry, or holdout.

---

## 4. Cost & model-routing policy

- **Baseline today is $0** (`SUMMON_FORCE_OAUTH_ROUTING=1`, no `ANTHROPIC_API_KEY`; `MODEL_POLICY` file
  absent → `claude-opus-4-8` everywhere). All stages ship dark; Fable 5 is opted in by **one tier key**.
- **Keep codegen/propose on $0-Max** — the O(strategies) path (up to 12 propose + N codegen/fix per
  top-up) stays on `llm_cmd()` → `frontier` failsafe. **Never repoint `frontier` at Fable 5.**
- **Fable 5 only for O(nights) roles** — the agentic scout (~40% of top-ups) and/or the planner (1/night),
  confined via the `scout`/`planner` tiers + `FORGE_SCOUT_MODEL`.
- **Kill paths (any one, all cheap):** `SCOUT_AGENTIC=0` (instant $0 revert) · remove/repoint the `scout`
  tier key (one-line JSON, $0) · `--max-turns` bounds per-call burn (**must be CLI-verified**) ·
  Opus-4.8 refusal fallback treated as fail-loud · the one-call-per-night rule.
- **Compliance:** Fable 5 has mandatory 30-day retention (no ZDR). The scout consumes wiki gaps + public
  queries + the diversity brief — **not** proprietary strategy source. Keep it that way; do not send
  strategy code through the scout turn.

---

## 5. Non-goals / what stays frozen

- **Gate stack, FDR bar, write-once holdout** (`sdk/harness.py`, `sdk/gates.py`, `vendor/research_integrity`):
  untouched. Faster discovery comes from smarter search, never weaker gates.
- **`config.llm_cmd()` stays `--no-tools`** byte-for-byte; codegen never becomes agentic.
- **No autonomous capital**; **one LLM seam** (`config.py`); **no cross-repo Python imports**.
- **The deterministic orchestration stays authoritative** — Thompson bandit + explore floor, MAP-Elites,
  director dedup/theme-cap/deployability. Fable 5 only re-weights *inputs to search*.
- **No re-proposing the done roadmap** (MAP-Elites, 4-arm operators, Thompson bandit, regime burner,
  lifecycle/decay, self-triage, data sentinel, gate canary — all already shipped).

---

## 6. Acceptance evidence per stage

- **Stage 0:** `joint_state.json` (schema v1, atomic) carries both the research snapshot and the Atlas
  paper-book snapshot with **zero cross-repo imports**; brief renders < a few hundred chars naming the
  actually-closed/deployed families. **Proof:** the fraction of top-up candidates hitting director dedup
  / `_closed_families()` **drops** vs the pre-brief baseline (from director logs). No gate/FDR file touched.
- **Stage 1:** file present with `frontier`/`standard`/`fast` all → `claude-opus-4-8`; a dry run confirms
  `frontier` still resolves to the $0 model (codegen stays $0) and flipping only `scout` routes only the
  scout. Docs updated.
- **Stage 2:** on the prod box, `summon --help` **confirms** `--mcp-config`/`--max-turns`/tool-allowlist;
  a turn-capped call visibly stops at the cap; the Opus-4.8 fallback fires + is logged on an induced
  refusal; `llm_cmd()` output is byte-identical to before.
- **Stage 3:** default-OFF verified (unset ⇒ tool-less path bit-for-bit). ON ⇒ identical distill-JSON
  shape into `candidates.md`; a parse/timeout/refusal RAISES (no false 0-candidate night in `log.md`).
  **Primary acceptance:** measured **distinct-novel-family lift via `seeded_by`** over a defined window
  vs the tool-less baseline. No lift ⇒ kill.
- **Stage 4 (if pursued):** `arm_bias` demonstrably **modulates** (never zeroes) the weights and the 25%
  explore floor holds; a missing/garbled plan degrades to pure bandit with the night still running;
  measured discovery-rate improvement without arm-reward regression.

**Sequencing rule:** Stage 0 → 1 → (verify summon) → 2 → 3; one Fable-5 orchestration call per night;
every stage reversible to $0 by an env flag or a one-line JSON diff. Stage 4 only if 2/3 prove value.

**Key files:** `agent/config.py`, `agent/scout.py`, `agent/director.py`, `agent/bandit.py`,
`agent/elite.py`, `agent/forge_state.py`, `agent/morning_report.py`, `live/deploy.py`,
`crucible_paths.py`, `mcp/server.py`, `/root/.pi/model-policy.json` (to be created).
