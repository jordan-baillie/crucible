"""Fable-5 orchestration model seam (agent/config.py). Pure — no LLM, no network.

Locks the cost-safety boundary from tasks/FABLE5_ORCHESTRATION_PLAN.md:
  - llm_cmd() (propose/codegen) stays --no-tools and never gains tools/MCP — codegen never agentic.
  - scout_cmd() is the ONE tools-on sibling: MCP wired, tool-allowlisted, hard --max-turns cap.
  - planner_cmd() is tool-less like llm_cmd() but its own model.
  - all three route via the MODEL_POLICY tiers seam, and FAIL SAFE to the $0 forge MODEL when a tier
    is absent — so the whole feature ships DARK at $0 until a policy key opts Fable-5 in.
"""
from agent import config


def test_llm_cmd_stays_pure_generation():
    cmd = config.llm_cmd()
    assert "--no-tools" in cmd                     # codegen/propose never run agentically
    assert "--mcp-config" not in cmd and "--tools" not in cmd
    assert "--max-turns" not in cmd
    assert config.MODEL in cmd
    assert config.pi_cmd() == config.llm_cmd()     # back-compat alias must never diverge


def test_scout_cmd_is_the_one_tools_on_sibling():
    cmd = config.scout_cmd()
    assert "--no-tools" not in cmd                 # the scout is agentic BY DESIGN
    assert "--mcp-config" in cmd and config.SCOUT_MCP in cmd
    assert "--tools" in cmd
    # tool surface is ALLOWLISTED to the read-only research MCP (no write/exec tool)
    tools = cmd[cmd.index("--tools") + 1]
    assert set(tools.split(",")) == set(config.SCOUT_TOOLS)
    assert "bash" not in tools and "write" not in tools and "edit" not in tools
    # a hard turn cap is present (defence-in-depth against an unbounded agentic loop)
    assert "--max-turns" in cmd
    assert int(cmd[cmd.index("--max-turns") + 1]) == config.SCOUT_MAX_TURNS
    assert config.SCOUT_MODEL in cmd


def test_planner_cmd_is_toolless_with_its_own_model():
    cmd = config.planner_cmd()
    assert "--no-tools" in cmd                     # planner is PURE generation (advisory hint only)
    assert "--tools" not in cmd and "--mcp-config" not in cmd
    assert config.PLANNER_MODEL in cmd


def test_policy_model_fails_safe_to_the_dollar_zero_model():
    # an absent tier (or absent policy file) must resolve to the given failsafe — the property that
    # keeps codegen on $0 and lets the whole feature ship dark until a 'scout'/'planner' key is added.
    assert config._policy_model("definitely_not_a_tier", failsafe="ZZZ-FAILSAFE") == "ZZZ-FAILSAFE"


def test_orchestration_tiers_default_to_the_forge_model_when_unconfigured():
    # With no FORGE_SCOUT_MODEL/FORGE_PLANNER_MODEL env and no MODEL_POLICY file present, the scout and
    # planner tiers fall back to the same $0 model the forge already uses — Fable-5 is strictly opt-in.
    import os
    from crucible_paths import MODEL_POLICY
    if not os.environ.get("FORGE_SCOUT_MODEL") and not MODEL_POLICY.exists():
        assert config.SCOUT_MODEL == config.MODEL
    if not os.environ.get("FORGE_PLANNER_MODEL") and not MODEL_POLICY.exists():
        assert config.PLANNER_MODEL == config.MODEL
