"""Shared forge-agent config — single source of truth for the LLM all smiths use."""
import os

# The model every smith uses for propose / codegen / scout.
# Override per-run with the FORGE_MODEL env var (e.g. fall back to sonnet if Max usage is tight).
MODEL = os.environ.get("FORGE_MODEL", "claude-opus-4-8")
