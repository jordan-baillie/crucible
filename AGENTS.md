# Crucible — agent guide

Autonomous strategy-discovery **forge**. LLM "smiths" generate trading-strategy hypotheses
cheaply; a non-bypassable statistical **gate stack** burns away everything that isn't real.
Survivors deploy to Atlas (paper → human-gated capital). Branch: `master`.

How to work here: see [`.pi/APPEND_SYSTEM.md`](.pi/APPEND_SYSTEM.md) — **simplest form, subtract
before you add, leave the tree more findable than you found it.** This file is the repo map and
the rules you must not break.

## Where things live
| Path | What |
|---|---|
| `sdk/harness.py` | **The gate stack — the JUDGE.** `StrategySpec` (what an experiment fills in) + `run_experiment()` (everything else). FROZEN: agents fill a spec; the harness owns every verdict. |
| `sdk/gates.py` | Demotion `Check`s, the 5 tagged failure modes, and the `active_from` phase-in. |
| `sdk/adapters.py` | Tested data loaders (Sharadar / FRED / yfinance) generated code composes. |
| `sdk/` (rest) | `wiki.py`, `queue.py` + `locks.py` (multi-agent `FileLock` coordination), `notify.py`, `cost_model.py`. |
| `vendor/research_integrity/` | The statistical rails package: CPCV, Deflated Sharpe, PBO, the write-once holdout ledger, the rising FDR bar. |
| `agent/` | The loop: `propose` → `codegen` → `sandbox` → `run_worker`; plus `director` (queue), `elite` (MAP-Elites pool), `triage` (runtime-error debugger), `sentinel` / `canary` (monitors), `scout`, `digest` / `morning_report`, `lint`. |
| `agent/config.py` | **The model seam** — `pi_cmd()` is the single LLM invocation. |
| `live/deploy.py` | PASS → Atlas paper-book bridge (the cross-repo seam; see its docstring). |
| `forward/` | Forward-validation tracks + stage-2 tools (`mcpt.py`, `generalize.py`). |
| `strategies/` | **Generated** `auto_*.py` strategy modules, kept as experiment evidence. Never hand-edit or hand-add. |
| `examples/first_experiment.py` | Entry point — copy this to hand-write a `StrategySpec`. |
| `scripts/bootstrap_wiki.py` | Fresh-machine wiki setup. |
| `crucible_paths.py` | Every external coupling, in one env-overridable place. |
| `tests/` | pytest suite (CI runs it on Linux **and** Windows). |

This repo is **not** GitNexus-indexed — navigate with this map + grep/glob.

## Commands
```bash
# install (rails first, then the package)
pip install -e vendor/research_integrity && pip install -e . && pip install pytest
# test (exactly what CI runs)
pytest tests/ -q
pytest tests/ -q -m "not network"      # skip live-network tests
# run one experiment, research-only (no deploy, no capital)
CRUCIBLE_DEPLOY="" python3 examples/first_experiment.py
```

## Configuration
All external coupling is in `crucible_paths.py`, env-overridable. The ones you'll touch:
- `FORGE_MODEL` — LLM for propose/codegen/scout (else `MODEL_POLICY` tiers → failsafe).
- `CRUCIBLE_DEPLOY` — execution host path; **`""` = research-only (start here).**
- `CRUCIBLE_WIKI` — the research wiki (default `/root/research-wiki`). **External: not in this repo.**

## Never break these (invariants)
- **The gate stack is the judge and is non-bypassable.** Never weaken, bypass, or "help"
  `sdk/harness.py`, `sdk/gates.py`, or `vendor/research_integrity`. Changing a threshold *is*
  changing the science — flag it, never do it silently or to make a test pass.
- **The write-once holdout and the rising FDR bar are sacred.** A config is evaluated once. Do
  not add a re-test path, reset the registry, or reuse a holdout slice.
- **`agent/config.py::pi_cmd()` is the only LLM invocation.** It deliberately sends
  `--system-prompt` and `--no-tools` / `--no-context-files` (pure generation, billing routing).
  Don't add a second LLM call path or strip those flags.
- **The cross-repo seam is frozen** (`live/deploy.py` docstring): `deploy_pass(...)` into
  `atlas.execution.providers`; `data/live/<name>/target.json`; reads back `config/live_strategies.json`.
  Both sides depend on these file shapes — don't change one without the other.
- **Capital is human-gated downstream.** Nothing here moves real money; never add an
  auto-promote-to-live path.
- `LOOP_DISABLED` (`agent/forge_state.py`) halts the forge — leave the check intact.

## Conventions
- `strategies/auto_*.py` are **generated** research evidence, loaded dynamically by id. Don't
  delete them in a cleanup and don't hand-author there — write a `StrategySpec` and let `codegen`
  emit the module.
- New cross-process state goes through `sdk/locks.py` `FileLock` (the pattern the FDR registry uses).
- Prefer **survivorship-clean Sharadar SEP/SF1** for US equities; never yfinance (a wiki anti-pattern).
- The queue, FDR registry, and elite pool are append-only + file-locked — respect that for parallel smiths.

## Gotchas
- The **wiki, `agent/run_log.jsonl`, and the elite pool are NOT in this repo** — they live in the
  external `$CRUCIBLE_WIKI` dir (or are gitignored). Don't assume you can read them here.
- `rlimits` in `agent/sandbox.py` are **POSIX-only** (no-op on Windows); the AST denylist applies everywhere.
- The forge runs under systemd (`systemd/crucible-*.{service,timer}`), not by hand.
