# Crucible

**Industrialized autonomous strategy research.** LLM agents ("smiths") generate trading-strategy
hypotheses for ~$0; the value is the **crucible** — a non-bypassable gate stack that burns away
everything that isn't a real, harvestable premium. 39 cycles in, the system has correctly produced
**zero false PASSes**: every near-miss was killed by a later gate for a documented, distinct reason.

> A strategy that survives the crucible earns a Telegram alert and a paper-trading book.
> Real capital is **always human-gated** — the machine never touches money on its own.

## The gate stack

```
propose ─► Gate-0 data feasibility ─► pre-register (FROZEN design) ─► build (sandboxed codegen)
   │
   ▼
STAGE 1 — statistical                      STAGE 2 — adversarial (runs same night)
  ├ tier-0 screen (|Sharpe| ≥ 0.3)           ├ MCPT permutation test FIRST  p ≤ 0.05
  ├ CPCV + PBO + Deflated Sharpe             │   (long-biased books: benchmark-relative null)
  ├ FDR-aware promote bar (rises with        └ cross-universe breadth battery
  │   every family ever tested, all agents)        (frozen signal, untouched universes,
  ├ write-once HOLDOUT (single-use ledger)          holdout-window-only scoring)
  ├ deployment-sanity (real diversified book)
  └ beta-confound (long-only books must beat   PASSED_ALL_GATES = stage1 ∧ MCPT ∧ breadth
      the equal-weight universe, not ride it)
```

Why two adversarial gates: **breadth catches lucky-universe overfits; MCPT catches construction
artifacts** (edges manufactured by the construction itself — they replicate on *every* universe,
so breadth alone is structurally blind to them; confirmed twice in production).

## Layout

| Path | What |
|---|---|
| `agent/` | The forge loop: `propose` → `codegen` → `sandbox` → `run_worker` (per-smith), `director` (queue/promotion decisions), `elite` (evolutionary pool), `digest`/`morning_report` (Telegram), `lint` (wiki hygiene) |
| `sdk/` | `harness.py` (the gate stack — owns every verdict), `adapters.py` (tested data loaders), `wiki.py` (knowledge-base writer), `queue.py`/`locks.py` (multi-agent coordination), `notify.py` |
| `live/` | `deploy.py` — PASS/local-candidate → paper-book deploy (pluggable target, see `CRUCIBLE_DEPLOY`) |
| `forward/` | Forward-validation tracks + stage-2 battery tools (`mcpt.py`, `generalize.py`) |
| `strategies/` | Generated strategy modules — **experiment evidence**, kept for reproducibility (verdicts in the wiki reference them) |
| `vendor/research_integrity/` | Vendored snapshot of the rails package (`pip install -e vendor/research_integrity`) |
| `systemd/` | Deploy templates: nightly forge (3 smiths), morning report, state backup, wiki lint |
| `crucible_paths.py` | **Every external coupling, one file, all env-overridable** |

## Shared state: the research wiki

All cross-agent state lives in a separate git repo (the "research wiki", `CRUCIBLE_WIKI`):
experiment pages, the hypothesis queue, locks, the elite pool, closed decisions/families, and the
**shared FDR registry** — the single most important multi-agent safeguard (N parallel searchers
share one rising promote bar, so false-discovery risk doesn't multiply with agent count).

## Configuration

All external paths resolve through `crucible_paths.py`, env-overridable:

| Env var | Default | Purpose |
|---|---|---|
| `CRUCIBLE_ROOT` | repo dir | repo root |
| `CRUCIBLE_WIKI` | `/root/research-wiki` | shared research wiki (queue/locks/registry/pages) |
| `CRUCIBLE_DATA` | `/root/atlas/data` | market data root (`sharadar/`, `cache/`, `live/`) |
| `CRUCIBLE_SECRETS` | `~/.atlas-secrets.json` | JSON: `telegram_bot_token`, `telegram_chat_id`, `fred_api_key` |
| `CRUCIBLE_DEPLOY` | `/root/atlas` | paper-deploy target (Atlas-style `live/providers.deploy_pass`) |
| `MODEL_POLICY` | `/root/.pi/model-policy.json` | central model policy (tiers + effort levels) |
| `FORGE_MODEL` / `FORGE_THINKING` | policy / pi default | per-run LLM model + effort (`low`…`xhigh`, `max`, `ultracode`) |
| `FRED_API_KEY` | from secrets file | FRED adapter override |
| `BOREAS_RESEARCH` | `/root/boreas/research` | validated TSMOM hedge-leg source (`trend_returns` adapter) |

LLM calls go through the `pi` CLI (Claude Max OAuth, $0 marginal). **Every subprocess call must
carry `--system-prompt`** — see `agent/config.py::pi_cmd()` for the canonical invocation.

## Running

```bash
pip install -e vendor/research_integrity && pip install -e .

python3 -m agent.run_worker --cycles 1        # one supervised cycle
python3 agent/digest.py                       # Telegram digest

# nightly autonomy (3 smiths, 03:30): see systemd/crucible-forge.{service,timer}
touch LOOP_DISABLED                           # KILLSWITCH — halts the loop (checked first)
```

## Safety invariants

1. **Rails non-bypassable** — `sdk/harness.py` owns every verdict; smiths cannot modify frozen
   pre-registered experiments or grade their own work.
2. **No autonomous capital** — PASS → paper book only; real capital needs explicit human action.
3. **Shared FDR bar** — prevents N-agent false-discovery inflation.
4. **Write-once holdout** — single-use ledger; the only incorruptible arbiter.
5. **Wiki pages never silently overwritten** — id collisions version with a title-hash suffix;
   `agent/run_log.jsonl` is the reconstruction source of truth.
