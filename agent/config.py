"""Shared forge-agent config — single source of truth for the LLM invocation all smiths use."""
import json
import os


def _policy_model(tier: str = "frontier", failsafe: str = "claude-opus-4-8") -> str:
    """Read the central model policy (/root/.pi/model-policy.json). Failsafe = $0-Max model."""
    try:
        import crucible_paths
        with open(crucible_paths.MODEL_POLICY) as fh:
            return json.load(fh)["tiers"][tier]
    except Exception:
        return failsafe


# The model every smith uses for propose / codegen / scout.
# Resolution order: FORGE_MODEL env (per-run override) > central policy > $0-Max failsafe.
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


def pi_cmd() -> list[str]:
    """The pi invocation for ALL forge LLM calls.

    --no-tools is critical: these are PURE generation calls (given context -> return JSON/code).
    Without it, pi runs AGENTICALLY and the codegen step ran the entire backtest itself in a bash
    tool loop until the 15-min timeout -> crash, plus ~2x compute and heavy Max-quota burn (the
    issuance-factor run died exactly this way). --no-context-files skips AGENTS.md discovery."""
    return ["pi", "-p", "--model", MODEL, *_thinking_args(), "--no-tools", "--no-context-files",
            "--system-prompt", SYS, "--mode", "json"]
