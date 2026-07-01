"""Fable-5 agentic-scout config contract (agent/config.scout_cmd + SCOUT_* constants).

Pins the summon invocation for the ONE agentic forge path so it can't silently drift from summon's
real flags (--mcp-config / --tools / --max-turns) or lose its safety boundary (read-only tools only,
a hard turn cap). scout_cmd() is OFF by default in the forge; this test guards the surface that turns
it on. Pure config — no network, no summon subprocess."""
import importlib

import agent.config as C


def _flag_value(cmd, flag):
    """Return the single arg following `flag` in a CLI list (or None if the flag is absent)."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_scout_cmd_is_agentic_and_carries_the_three_flags():
    cmd = C.scout_cmd()
    # summon print + json stream, same as the tool-less path.
    assert cmd[:2] == ["summon", "-p"]
    assert _flag_value(cmd, "--mode") == "json"
    # The three agentic flags, with summon's real names.
    assert "--mcp-config" in cmd
    assert _flag_value(cmd, "--mcp-config") == C.SCOUT_MCP
    assert _flag_value(cmd, "--tools") == ",".join(C.SCOUT_TOOLS)
    assert _flag_value(cmd, "--max-turns") == str(C.SCOUT_MAX_TURNS)


def test_scout_cmd_is_not_the_tool_less_generation_call():
    cmd = C.scout_cmd()
    # --no-tools is what makes llm_cmd() non-agentic; the scout MUST NOT carry it.
    assert "--no-tools" not in cmd
    # It is genuinely different from the pure-generation invocation.
    assert cmd != C.llm_cmd()


def test_scout_cmd_shares_model_and_system_prompt_with_llm_cmd():
    scout, llm = C.scout_cmd(), C.llm_cmd()
    # Same model routing (incl. claude-fable-5 via OAuth) and same system prompt / context policy,
    # so the JSON stream parses identically to the proven tool-less path.
    assert _flag_value(scout, "--model") == _flag_value(llm, "--model") == C.MODEL
    assert _flag_value(scout, "--system-prompt") == _flag_value(llm, "--system-prompt") == C.SYS
    assert "--no-context-files" in scout


def test_scout_tools_are_read_only_research_tools_only():
    # HARD safety boundary: the agentic scout must never receive a mutating/exec tool.
    forbidden = {"bash", "write", "edit", "exec", "shell", "run", "python", "read", "apply_patch"}
    assert forbidden.isdisjoint(C.SCOUT_TOOLS)
    # The five research tools the crucible-research MCP exposes for discovery.
    assert C.SCOUT_TOOLS == ["x_search", "web_search", "research_search", "scrape_url", "extract_url"]


def test_scout_max_turns_is_a_positive_bound():
    assert isinstance(C.SCOUT_MAX_TURNS, int)
    assert C.SCOUT_MAX_TURNS > 0


def test_scout_cmd_thinking_flag_follows_forge_thinking(monkeypatch):
    # scout inherits the same --thinking plumbing as every other forge call.
    monkeypatch.setenv("FORGE_THINKING", "high")
    importlib.reload(C)
    try:
        cmd = C.scout_cmd()
        assert _flag_value(cmd, "--thinking") == "high"
    finally:
        monkeypatch.delenv("FORGE_THINKING", raising=False)
        importlib.reload(C)


def test_forge_scout_mcp_env_overrides_the_mcp_config_path(monkeypatch):
    monkeypatch.setenv("FORGE_SCOUT_MCP", "/custom/mcp-config.json")
    importlib.reload(C)
    try:
        assert C.SCOUT_MCP == "/custom/mcp-config.json"
        assert _flag_value(C.scout_cmd(), "--mcp-config") == "/custom/mcp-config.json"
    finally:
        monkeypatch.delenv("FORGE_SCOUT_MCP", raising=False)
        importlib.reload(C)


def test_forge_scout_max_turns_env_override(monkeypatch):
    monkeypatch.setenv("FORGE_SCOUT_MAX_TURNS", "6")
    importlib.reload(C)
    try:
        assert C.SCOUT_MAX_TURNS == 6
        assert _flag_value(C.scout_cmd(), "--max-turns") == "6"
    finally:
        monkeypatch.delenv("FORGE_SCOUT_MAX_TURNS", raising=False)
        importlib.reload(C)
