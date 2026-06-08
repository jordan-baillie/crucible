"""Shared forge-agent config — single source of truth for the LLM invocation all smiths use."""
import os

# The model every smith uses for propose / codegen / scout.
# Override per-run with FORGE_MODEL (e.g. fall back to sonnet if Max usage is tight).
MODEL = os.environ.get("FORGE_MODEL", "claude-opus-4-8")
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def pi_cmd() -> list[str]:
    """The pi invocation for ALL forge LLM calls.

    --no-tools is critical: these are PURE generation calls (given context -> return JSON/code).
    Without it, pi runs AGENTICALLY and the codegen step ran the entire backtest itself in a bash
    tool loop until the 15-min timeout -> crash, plus ~2x compute and heavy Max-quota burn (the
    issuance-factor run died exactly this way). --no-context-files skips AGENTS.md discovery."""
    return ["pi", "-p", "--model", MODEL, "--no-tools", "--no-context-files",
            "--system-prompt", SYS, "--mode", "json"]
