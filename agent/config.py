"""Shared forge-agent config — single source of truth for the LLM invocation all smiths use."""
import json
import os


def _policy_model(tier: str = "frontier", failsafe: str = "claude-opus-4-8") -> str:
    """Read the central model policy (MODEL_POLICY path). Failsafe = a safe default model."""
    try:
        import crucible_paths
        with open(crucible_paths.MODEL_POLICY) as fh:
            return json.load(fh)["tiers"][tier]
    except Exception:
        return failsafe


# The model every smith uses for propose / codegen / scout.
# Resolution order: FORGE_MODEL env (per-run override) > central policy > failsafe.
MODEL = os.environ.get("FORGE_MODEL") or _policy_model()
SYS = "You are Claude Code, Anthropic's official CLI for Claude."

# Optional effort level for forge LLM calls (FORGE_THINKING env).
# Accepts the Anthropic API effort vocabulary (low/medium/high/xhigh/max) plus pi's native
# levels (off/minimal) and the 'ultracode'/'ultrathink' aliases. 'max' maps to xhigh — pi's
# ceiling — until pi exposes the API's max tier. Unset = pi default.
_THINKING_ALIASES = {"ultracode": "xhigh", "ultrathink": "xhigh", "max": "xhigh"}
_VALID_THINKING = {"off", "minimal", "low", "medium", "high", "xhigh"}


def _thinking_args() -> list[str]:
    lvl = (os.environ.get("FORGE_THINKING") or "").strip().lower()
    if not lvl:
        return []
    lvl = _THINKING_ALIASES.get(lvl, lvl)
    if lvl not in _VALID_THINKING:
        return []  # never crash a smith over a typo'd env var
    return ["--thinking", lvl]


def llm_cmd() -> list[str]:
    """The summon invocation for ALL forge LLM calls.

    Migrated pi -> summon (2026-06-22): the fleet's $0-Max OAuth now lives in summon
    (anthropic-oauth extension); the legacy pi auth path expired and `pi login` no longer
    works on this box. summon accepts the identical flag set and its JSON stream is parsed
    by agent.llm.assistant_text unchanged. Routing is fail-closed to the subscription via
    SUMMON_FORCE_OAUTH_ROUTING=1 (set in the systemd unit; no ANTHROPIC_API_KEY present).

    --no-tools is critical: these are PURE generation calls (given context -> return JSON/code).
    Without it the agent runs AGENTICALLY and codegen ran the entire backtest itself in a bash
    tool loop until timeout -> crash, plus ~2x compute and heavy quota burn (the issuance-factor
    run died exactly this way). --no-context-files skips AGENTS.md/CLAUDE.md discovery."""
    return ["summon", "-p", "--model", MODEL, *_thinking_args(), "--no-tools", "--no-context-files",
            "--system-prompt", SYS, "--mode", "json"]


# Back-comat alias: some callers/tests still import the historical name. Single source of truth
# is llm_cmd(); this alias must never diverge from it.
pi_cmd = llm_cmd


# ── Fable-5 orchestration tiers (tasks/FABLE5_ORCHESTRATION_PLAN.md) ────────────────────────────────
# The forge's O(strategies) path (propose/codegen) stays on MODEL above (frontier failsafe = $0 Max).
# Fable-5 is confined to two O(nights) ORCHESTRATION roles, each routed via its own MODEL_POLICY tier
# so cost is opt-in by a single JSON key and reversible by removing it. Absent the tier/env, both fall
# back to MODEL (the same $0 model as everything else) — the code ships DARK at $0.
#   scout   -> the agentic (tools-ON) scout turn (Stage 2/3), FORGE_SCOUT_MODEL / tier 'scout'
#   planner -> the tool-less night-planner (Stage 4),        FORGE_PLANNER_MODEL / tier 'planner'
SCOUT_MODEL = os.environ.get("FORGE_SCOUT_MODEL") or _policy_model(
    os.environ.get("FORGE_SCOUT_TIER", "scout"), failsafe=MODEL)
PLANNER_MODEL = os.environ.get("FORGE_PLANNER_MODEL") or _policy_model(
    os.environ.get("FORGE_PLANNER_TIER", "planner"), failsafe=MODEL)

# The agentic scout READS external sources via the existing crucible-research MCP (mcp/server.py) and
# emits candidate TEXT — it has NO strategy module to execute, so the --no-tools codegen-crash rationale
# does not apply. Tools are ALLOWLISTED to the read-only research surface and the turn is hard-capped.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOUT_MCP = os.environ.get("FORGE_SCOUT_MCP") or os.path.join(_REPO_ROOT, "mcp", "run.sh")
SCOUT_TOOLS = ("x_search", "web_search", "research_search", "scrape_url", "extract_url")
SCOUT_MAX_TURNS = int(os.environ.get("FORGE_SCOUT_MAX_TURNS", "12"))


def scout_cmd() -> list[str]:
    """summon invocation for the AGENTIC scout — the ONE place tools are enabled (Stage 2).

    A SIBLING of llm_cmd(), never a replacement: llm_cmd() stays --no-tools byte-for-byte so
    propose/codegen never run agentically. Differences: tools ON but ALLOWLISTED to the read-only
    crucible-research MCP (SCOUT_TOOLS), an --mcp-config pointing at that server, and a hard
    --max-turns cap so an agentic turn cannot loop unboundedly (defence-in-depth — the scout has no
    backtest to run, unlike the codegen path the --no-tools comment warns about). Routed to
    SCOUT_MODEL via the 'scout' MODEL_POLICY tier; absent it, falls back to the $0 forge MODEL.

    ⚠ FLAG SPELLINGS ASSUMED from the pi flag surface summon succeeded (--mcp-config / --tools /
    --max-turns). VERIFY against `summon --help` on the forge box before enabling SCOUT_AGENTIC — the
    agentic path is opt-in and default-OFF precisely so an unverified flag can never break a night."""
    return ["summon", "-p", "--model", SCOUT_MODEL, *_thinking_args(),
            "--mcp-config", SCOUT_MCP, "--tools", ",".join(SCOUT_TOOLS),
            "--max-turns", str(SCOUT_MAX_TURNS),
            "--no-context-files", "--system-prompt", SYS, "--mode", "json"]


def planner_cmd() -> list[str]:
    """Tool-less summon invocation for the night-planner (Stage 4). Identical discipline to llm_cmd()
    (--no-tools PURE generation) but routed to PLANNER_MODEL via the 'planner' MODEL_POLICY tier, so
    Fable-5 orchestration cost stays O(nights). Absent the tier => falls back to the $0 forge MODEL."""
    return ["summon", "-p", "--model", PLANNER_MODEL, *_thinking_args(), "--no-tools",
            "--no-context-files", "--system-prompt", SYS, "--mode", "json"]
