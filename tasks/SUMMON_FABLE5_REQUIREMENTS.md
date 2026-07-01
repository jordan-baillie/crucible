# summon requirements for the Fable-5 agentic scout (crucible cross-repo contract)

*2026-07-01. Hand-off doc: what `summon` must support for `tasks/FABLE5_ORCHESTRATION_PLAN.md`
Stage 2/3 (the agentic scout) to work. Written from crucible because summon is a separate repo not
in this session's scope. `atlas` needs **no** change — crucible reads its paper books through the
existing read-only file-contract seam (`live/deploy.py`), no summon/atlas edit required for that.*

The forge already drives summon in production via `agent/config.py::llm_cmd()` (pure `--no-tools`
generation). This doc covers ONLY the **new** surface the agentic scout (`scout_cmd()`) and the
night-planner (`planner_cmd()`) rely on. All of it is opt-in and default-OFF in crucible, so nothing
here is urgent — but Stage 2/3 cannot be enabled until summon satisfies it.

## What crucible emits today

`agent/config.py::scout_cmd()` (the ONE tools-on invocation) builds, verbatim:

```
summon -p --model <SCOUT_MODEL> [--thinking <lvl>] \
       --mcp-config <FORGE_SCOUT_MCP or repo/mcp/run.sh> \
       --tools x_search,web_search,research_search,scrape_url,extract_url \
       --max-turns <FORGE_SCOUT_MAX_TURNS, default 12> \
       --no-context-files --system-prompt <SYS> --mode json
```

`planner_cmd()` (tool-less, Stage 4) is just `llm_cmd()` with a different model — it uses **only
already-supported flags**, so it needs nothing new from summon beyond model routing:

```
summon -p --model <PLANNER_MODEL> [--thinking <lvl>] --no-tools \
       --no-context-files --system-prompt <SYS> --mode json
```

Prompt is fed on **stdin**. Output is consumed by `agent/llm.py` (see "Stream contract" below).

## The three flags summon must support (Stage 2/3)

The pre-existing flags (`-p`, `--model`, `--thinking`, `--no-tools`, `--no-context-files`,
`--system-prompt`, `--mode json`) already work in prod. The **new** requirements are:

1. **`--mcp-config <path>`** — load an MCP server so the model can call its tools. crucible points
   this at the crucible-research stdio MCP (`mcp/run.sh` → `mcp/server.py`, tools: `x_search`,
   `web_search`, `research_search`, `scrape_url`, `extract_url`, `x_user_tweets`, `x_user_info`,
   `x_balance`, `extract_url`). If summon expects a **JSON config file** rather than the server
   command directly, that's fine — set crucible's `FORGE_SCOUT_MCP` env to that file's path and it
   passes through unchanged (no crucible code edit needed).
2. **`--tools <csv>`** — restrict the agentic turn to an **allowlist** of tool names (crucible passes
   the five read-only research tools above). This is a safety boundary: the scout must NOT get
   write/exec/bash tools. If summon's allowlist flag has a different name/shape, that's the one thing
   to reconcile (see "If summon's flags differ").
3. **`--max-turns <n>`** — a HARD cap on agentic turns so a tool loop cannot run unbounded. This is
   the backstop for the exact failure the `--no-tools` comment in `config.py` documents (an agentic
   codegen turn once ran a backtest in a bash loop until timeout → crash). The scout has no backtest
   to run, but the cap is required belt-and-braces before enabling it.

## Model / OAuth routing

- crucible resolves the model string itself (via `MODEL_POLICY` tiers) and passes a concrete
  `--model claude-fable-5` to summon. So summon needs only to **accept an arbitrary `--model`** and
  route it through its `anthropic-oauth` extension under `SUMMON_FORCE_OAUTH_ROUTING=1` (no
  `ANTHROPIC_API_KEY`). If summon already routes any `--model` via OAuth, **no summon change is
  needed for routing** — only the three flags above.
- Fable-5 is a paid/usage-credit tier (see the plan doc). Confirm the OAuth path can actually reach
  `claude-fable-5` on the subscription in use before enabling.

## Stream contract (must hold for AGENTIC turns too)

`agent/llm.py` parses summon's `--mode json` stream and must keep working unchanged:

- `assistant_text()` returns the **longest** text candidate across the stream, reading:
  `delta.text`; `message.content[].text` where `message.role == "assistant"`; and top-level
  `text`/`content` string fields. → **The final assistant message must contain the distill JSON as
  plain text** (the scout prompt asks for "ONLY the distill JSON"). Tool-call events may interleave,
  but the terminal answer must be assistant text, not buried inside a tool payload.
- `stream_error()` treats `message.stopReason == "error"` with an `errorMessage` as a hard failure.
  A Fable-5 **refusal** should surface as either that (→ crucible raises `LLMError`, fail-loud) or as
  an empty completion (→ crucible's parse guard raises). Either way crucible must never see a
  refusal as a silent empty success. If summon emits refusals with a *different* stop reason, tell me
  and I'll extend `stream_error()` to catch it.

## If summon's flags differ (the one crucible-side change)

Everything summon-specific is isolated to **three constants + one function** in `agent/config.py`:
`SCOUT_MCP`, `SCOUT_TOOLS`, `SCOUT_MAX_TURNS`, and `scout_cmd()`. If summon spells these flags
differently (e.g. `--mcp <file>`, `--allow-tools`, `--turn-limit`), paste `summon --help` and I'll
adjust `scout_cmd()` to match — no other crucible file changes. `test_fable5_config.py` pins the
current shape, so any change is caught by a test diff.

## Acceptance (before flipping SCOUT_AGENTIC=1 on the forge box)

1. `summon --help` shows `--mcp-config`, a tool-allowlist flag, and a turn cap (record the real names).
2. A manual `SCOUT_AGENTIC=1 python -m agent.scout` run: the model calls the MCP tools, stops at the
   turn cap, and returns parseable distill JSON into `candidates.md` (fail-loud on an empty/refused
   result — no false 0-candidate night).
3. `--model claude-fable-5` routes via OAuth without an API key.
4. Only then set the `scout` tier in `MODEL_POLICY` to `claude-fable-5` (keep `frontier` on the $0
   model — never move codegen onto Fable-5 credits).
