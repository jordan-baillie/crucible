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


# ── Agentic scout (Fable-5) ─────────────────────────────────────────────────────────────────────
# The ONE agentic forge path: a scout turn where the model drives the crucible-research MCP itself
# (x/web/paper search + scrape/extract) instead of the forge pre-fetching sources. OFF by default —
# no forge code calls scout_cmd() yet (the nightly scout stays tool-less via llm_cmd); this is the
# config surface the agentic scout switches on once summon ships the three flags below (it now does).
#
# summon's real flag names (verified against `summon --help`): --mcp-config / --tools / --max-turns.
def _default_scout_mcp() -> str:
    """Default --mcp-config target: the repo's crucible-research MCP launcher (stdio). summon accepts
    either this bare executable OR a {"mcpServers": {...}} JSON file, so FORGE_SCOUT_MCP can point at
    whatever a given box prefers. Graceful: never raises at import (mirrors _policy_model)."""
    try:
        import crucible_paths
        return str(crucible_paths.ROOT / "mcp" / "run.sh")
    except Exception:
        return "mcp/run.sh"


# summon --mcp-config target (the crucible-research MCP). FORGE_SCOUT_MCP overrides per box.
SCOUT_MCP = os.environ.get("FORGE_SCOUT_MCP") or _default_scout_mcp()
# summon --tools allowlist. HARD SAFETY BOUNDARY: read-only research tools ONLY — never bash/write/
# edit/exec. summon registers the MCP tools then activates only these names (the rest, incl. the X
# helpers x_user_tweets/x_user_info/x_balance and all built-ins, stay inert).
SCOUT_TOOLS = ["x_search", "web_search", "research_search", "scrape_url", "extract_url"]
# summon --max-turns: hard cap so an agentic tool loop can never run unbounded (the key backstop).
SCOUT_MAX_TURNS = int(os.environ.get("FORGE_SCOUT_MAX_TURNS", "12"))


def scout_cmd() -> list[str]:
    """The summon invocation for the AGENTIC scout turn (Fable-5 drives the crucible-research MCP).

    Differs from llm_cmd() in exactly the agentic dimension — it ADDS the three summon agentic flags
    and, critically, OMITS --no-tools (that flag is what makes llm_cmd() a pure non-agentic call):
      --mcp-config <path>   load the crucible-research MCP (stdio) so the model can call its tools
      --tools <allowlist>   restrict the turn to read-only research tools (no bash/write/edit/exec)
      --max-turns <n>       hard cap so a tool loop can't run unbounded
    Model routing (MODEL, incl. claude-fable-5 via OAuth), system prompt, --no-context-files, and the
    --mode json stream contract are IDENTICAL to llm_cmd(), so agent.llm.assistant_text/stream_error
    parse the agentic stream unchanged."""
    return ["summon", "-p", "--model", MODEL, *_thinking_args(),
            "--mcp-config", SCOUT_MCP,
            "--tools", ",".join(SCOUT_TOOLS),
            "--max-turns", str(SCOUT_MAX_TURNS),
            "--no-context-files", "--system-prompt", SYS, "--mode", "json"]
