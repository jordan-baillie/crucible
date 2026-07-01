# Crucible

**An autonomous quant-research pipeline.**

LLM agents ("smiths") generate trading-strategy hypotheses around the clock — generation is cheap
and unlimited. The value is the **crucible**: a non-bypassable stack of statistical gates that
burns away everything that isn't a real, harvestable market premium. Overfitting, lucky universes,
hidden beta, construction artifacts — each has a gate that kills it, and the agents cannot grade
their own work.

39 production cycles in, the system has produced **zero false PASSes**: every near-miss was killed
by a later gate for a documented, distinct reason — and every kill became a recorded lesson the
agents must obey on the next cycle.

> A strategy that survives the crucible earns a Telegram alert and a paper-trading book.
> Real capital is **always human-gated** — the machine never touches money on its own.

---

## How it works

```
                        ┌──────────────────────────────────────────────┐
                        │            RESEARCH WIKI (git repo)          │
                        │  experiments · lessons · closed decisions ·  │
                        │  hypothesis queue · locks · FDR registry     │
                        └───────▲──────────────────────────┬───────────┘
                          write │ every outcome       read │ before generating
                                │                          ▼
   ┌─────────┐   propose   ┌────┴────┐   codegen   ┌──────────────┐   verdict
   │ smith 1 │ ──────────► │  queue  │ ──────────► │   sandbox    │ ─────────┐
   │ smith 2 │  (LLM, one  │ (shared,│  (LLM writes│ (AST denylist│          │
   │ smith N │  hypothesis │  locked)│   the code) │  + rlimits)  │          │
   └─────────┘  at a time) └─────────┘             └──────────────┘          ▼
                                                              ┌───────────────────────┐
                                                              │   THE GATE STACK      │
                                                              │  (sdk/harness.py —    │
                                                              │  agents can't touch)  │
                                                              └───────────┬───────────┘
                                                                          │ PASS only
                                                   Telegram alert ◄───────┤
                                                                          ▼
                                                              ┌───────────────────────┐
                                                              │ paper book (optional  │
                                                              │ execution host) — no  │
                                                              │ real capital          │
                                                              └───────────────────────┘
```

### The gate stack

```
STAGE 1 — statistical                          STAGE 2 — adversarial
  ├ tier-0 screen (|search Sharpe| ≥ 0.3)        ├ MCPT permutation test FIRST   p ≤ 0.05
  ├ CPCV + PBO + Deflated Sharpe                 │   (permute prices, re-run the frozen
  ├ FDR-aware promote bar — RISES with           │    signal: a real edge dies on noise;
  │   every hypothesis family ever tested,       │    a construction artifact doesn't)
  │   shared across ALL agents                   └ cross-universe breadth battery
  ├ write-once HOLDOUT (single-use ledger —          (same frozen signal on pre-declared
  │   you can never test on it twice)                 untouched universes)
  ├ deployment-sanity (a real, diversified book)
  └ beta-confound (long-only books must beat     PASSED_ALL_GATES = stage1 ∧ MCPT ∧ breadth
      the equal-weight universe, not ride it)
```

Why **two** adversarial gates: breadth catches *lucky-universe* overfits; MCPT catches
*construction artifacts* — edges manufactured by the strategy construction itself, which replicate
on every universe and are therefore invisible to breadth testing. (Confirmed twice in production;
the strongest red flag in the toolkit is a permutation mean ≥ the real Sharpe.)

### The discipline

- **Pre-registration**: the design (universe, signal, sizing, costs, PASS/KILL criteria) is
  **frozen before the first run**. No tuning to rescue a failed experiment — the parameter grid
  exists only to make the multiple-testing burden honest.
- **Compounding negative knowledge**: every kill is recorded as a closed decision or meta-lesson;
  agents must read these before generating, so the same dead idea is never paid for twice.
- **Shared FDR bar**: the Deflated-Sharpe bar to promote *rises* with every distinct hypothesis
  family any agent has ever tested. N parallel agents cannot multiply false-discovery risk.

---

## Quick start

### Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| Python ≥ 3.10, `pip` | everything | |
| [`pi` CLI](https://github.com/getpi/pi) authenticated to Claude | the autonomous loop | any billing mode: API key, subscription OAuth, or gateway. **Not needed for manual experiments** |
| `yfinance` (`pip install yfinance`) | free market data | enough for the example + futures/ETF research |
| Sharadar SEP/SF1 zips (paid, optional) | survivorship-clean US-equity research | drop into `$CRUCIBLE_DATA/sharadar/` |
| Telegram bot token (optional) | PASS alerts + morning report | silently skipped if absent |

### 1 · Install

```bash
git clone <this-repo> crucible && cd crucible
pip install -e vendor/research_integrity   # the statistical rails (vendored)
pip install -e . && pip install yfinance
```

### 2 · Bootstrap your research wiki

The wiki is the system's memory — a plain directory (make it a git repo) of markdown + jsonl that
every agent reads before generating and writes after every run:

```bash
python3 scripts/bootstrap_wiki.py ~/research-wiki     # idempotent; seeds overview/catalog/lessons
export CRUCIBLE_WIKI=~/research-wiki
```

Then edit `~/research-wiki/DATA_CATALOG.md` to match the data you actually have — agents only
propose ideas buildable on what's listed there.

### 3 · Run your first experiment (no LLM needed)

```bash
CRUCIBLE_DEPLOY="" python3 examples/first_experiment.py
```

This pushes a hand-written ETF momentum strategy through the **full gate stack** — CPCV, deflated
Sharpe, PBO, holdout, deployment-sanity — and writes a verdict page into your wiki. Expect an
honest **FAIL** (`holdout_sharpe ≈ 0.1`): that's the machine working. Read the verdict at
`$CRUCIBLE_WIKI/experiments/example-tsmom-etf.md`.

To test your own idea, copy `examples/first_experiment.py`, fill in the `StrategySpec`
(`load_data`, `signal`, frozen `pre_registration` text, a small honest `grid`), and run it.
**Write the pre-registration before you look at results.**

### 4 · Run an autonomous cycle (LLM required)

```bash
pi --version                                   # verify the pi CLI is installed + authenticated
python3 -m agent.run_worker --cycles 1         # propose -> codegen -> sandbox -> gates -> wiki
```

The worker claims a hypothesis from the shared queue (topping it up via the director LLM if
empty), has the LLM write the strategy code, runs it sandboxed through the gates, and records the
verdict. Watch the wiki grow.

### 5 · Go nightly (optional)

```bash
# systemd templates: 3 parallel smiths at 03:30 + morning report + state backup + wiki lint
# Point ALL hardcoded paths at your environment (repo, wiki, data, backup dir):
sed -i "s|/root/crucible|$(pwd)|g; s|/root/research-wiki|$CRUCIBLE_WIKI|g; \
        s|/root/atlas/data|$CRUCIBLE_DATA|g; s|/root/backups|$HOME/backups|g" \
    systemd/*.service systemd/*.timer
sudo cp systemd/crucible-*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crucible-forge.timer crucible-morning-report.timer

touch LOOP_DISABLED        # KILLSWITCH — halts the loop instantly (checked before every cycle)
```

---

## Configuration

Every external coupling lives in **one file**, `crucible_paths.py`, all env-overridable:

| Env var | Default | Purpose |
|---|---|---|
| `CRUCIBLE_WIKI` | `/root/research-wiki` | the research wiki (memory, queue, locks, FDR registry) |
| `CRUCIBLE_DATA` | `/root/atlas/data` | market-data root (`sharadar/`, `cache/`) |
| `CRUCIBLE_SECRETS` | `~/.atlas-secrets.json` | JSON with `telegram_bot_token`, `telegram_chat_id`, `fred_api_key` (all optional) |
| `CRUCIBLE_DEPLOY` | `/root/atlas` | paper-trading execution host; **`""` = research-only mode** (recommended start) |
| `CRUCIBLE_BROKER` | `alpaca` | broker label passed to the execution host |
| `FORGE_MODEL` | policy file → failsafe | LLM model for propose/codegen/scout (the O(strategies) path — keep on the $0 model) |
| `FORGE_THINKING` | pi default | LLM effort: `low` `medium` `high` `xhigh` `max` `ultracode` |
| `MODEL_POLICY` | `/root/.pi/model-policy.json` | optional central model-tier JSON; absent → safe failsafe. Reference: `examples/model-policy.json` |
| `SCOUT_AGENTIC` | `0` (off) | `1` = the scout runs **agentically** (tools ON, drives the crucible-research MCP itself) instead of the tool-less two-call path |
| `FORGE_SCOUT_MODEL` / `FORGE_SCOUT_TIER` | `scout` tier → `FORGE_MODEL` | model for the agentic scout; absent tier → the $0 forge model (opt Fable-5 in via the policy `scout` tier) |
| `NIGHT_PLANNER` | `0` (off) | `1` = run the advisory night-planner pre-forge; `director` blends its `arm_bias` into the bandit (floors preserved, fail-open) |
| `FORGE_PLANNER_MODEL` / `FORGE_PLANNER_TIER` | `planner` tier → `FORGE_MODEL` | model for the night-planner (one O(nights) call); absent tier → the $0 forge model |
| `FRED_API_KEY` | from secrets file | FRED macro-data adapter |
| `BOREAS_RESEARCH` | `/root/boreas/research` | optional external TSMOM hedge-leg (`trend_returns` adapter) |

The **agentic scout** (`SCOUT_AGENTIC=1`) and **night-planner** (`NIGHT_PLANNER=1`) are the two
cost-routed **orchestration** roles from `tasks/FABLE5_ORCHESTRATION_PLAN.md`: Fable-5 is confined to
them via the `scout`/`planner` tiers in `MODEL_POLICY`, while `propose`/`codegen` stay on the `frontier`
($0) tier. Both are **off by default and reversible** — unset the flag (or repoint the tier to the $0
model) to revert. Codegen never becomes agentic (`llm_cmd()` stays `--no-tools`).

### LLM backend

All LLM calls go through the `pi` CLI as a subprocess — `agent/config.py::pi_cmd()` is the single
canonical invocation. **How you pay for Claude is your choice**: API key, subscription plan
(e.g. Claude Max OAuth), or a gateway — whatever your `pi` install is authenticated with.
(`pi_cmd()` always sends an explicit system prompt; some subscription billing routers key off its
presence, and it's correct practice for reproducible generation calls regardless.)

---

## Repository layout

| Path | What |
|---|---|
| `sdk/harness.py` | **The gate stack.** `StrategySpec` (what an experiment fills in) + `run_experiment()` (everything else). Agents cannot modify it |
| `sdk/adapters.py` | Tested data loaders (yfinance, FRED, Sharadar) the generated code composes |
| `sdk/` (rest) | `wiki.py` (knowledge writer), `queue.py`/`locks.py` (multi-agent coordination), `notify.py` (Telegram) |
| `agent/` | The autonomous loop: `propose` → `codegen` → `sandbox` → `run_worker`; `director` (queue strategy), `elite` (evolutionary pool), `digest`/`morning_report`, `lint` (wiki hygiene) |
| `live/deploy.py` | PASS → paper-book bridge (pluggable host, disableable) |
| `forward/` | Forward-validation tracks + stage-2 battery tools (`mcpt.py`, `generalize.py`) |
| `strategies/` | Generated strategy modules — kept as **experiment evidence** (wiki verdicts reference them) |
| `examples/` | `first_experiment.py` — your entry point |
| `scripts/` | `bootstrap_wiki.py` — fresh-machine setup |
| `vendor/research_integrity/` | The statistical rails package (CPCV, DSR, PBO, holdout ledger, FDR bar), vendored |
| `systemd/` | Nightly-autonomy unit templates |

---

## Plugging in an execution host (optional)

Crucible researches; a separate execution host paper-trades survivors. The seam is a three-file
contract (Atlas is the reference implementation), pluggable via `CRUCIBLE_DEPLOY`:

| Direction | Interface |
|---|---|
| crucible → host | writes `<host>/data/live/<name>/target.json` (today's target weights) |
| crucible → host | calls `<host>/execution/providers.py::deploy_pass(name, capital, broker, expectation, strategy_path)` (Atlas: `from atlas.execution.providers import deploy_pass`) |
| crucible ← host | reads `<host>/config/live_strategies.json` (registry) + `<host>/data/live/<name>/{book.json,returns.jsonl}` (paper books) for the morning report |

The host owns execution truth (brokers, fills, books); crucible owns research truth (verdicts,
wiki, FDR registry). Neither writes the other's state. With `CRUCIBLE_DEPLOY=""` the system is
research-only and verdicts still record normally.

---

## Safety invariants

1. **Rails non-bypassable** — `sdk/harness.py` owns every verdict; agents cannot modify frozen
   pre-registrations or grade their own work.
2. **No autonomous capital** — PASS → paper book only; real money requires explicit human action.
3. **Shared FDR bar** — the promote threshold rises with every family ever tested, across all agents.
4. **Write-once holdout** — a single-use ledger makes the quarantined slice incorruptible.
5. **Sandboxed codegen** — generated code runs under an AST denylist + resource limits.
6. **Killswitch** — `touch LOOP_DISABLED` halts the loop before the next cycle.
7. **No silent overwrites** — wiki id collisions version with a hash suffix; `agent/run_log.jsonl`
   is the append-only reconstruction source of truth.
