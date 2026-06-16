#!/usr/bin/env bash
# Launch the crucible research MCP (stdio). Used by an MCP client's server config.
exec "$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/server.py"
