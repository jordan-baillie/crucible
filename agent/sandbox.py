"""Sandbox for agent-generated signal code (D2: AST denylist + rlimits — pragmatic, dependency-free).

The harness (trusted) owns ALL I/O — registry/wiki/telegram writes and the rails. The agent's
generated module must be pure compute over a data panel; data is fetched by sdk.adapters (which
the agent CALLS, not reimplements). So we reject any generated code that does process spawning,
networking (raw), filesystem writes, dynamic eval, deserialization, or broker/capital calls —
none of which a legitimate signal/data-spec ever needs. Calibrated against existing generated
strategies (they use only: numpy, pandas, yfinance, sys, warnings, math, sdk.*, tsmom).
"""
from __future__ import annotations

import ast

# Importing any of these from generated code is a violation (data libs don't need them).
_BANNED_MODULES = {
    "os", "subprocess", "socket", "shutil", "requests", "urllib", "http", "httplib",
    "ftplib", "smtplib", "telnetlib", "ctypes", "cffi", "pickle", "marshal",
    "multiprocessing", "asyncio", "importlib", "pty", "pathlib", "tempfile",
    # broker / live-capital — never allowed in a research signal
    "ccxt", "ib_insync", "ibapi", "alpaca_trade_api", "paramiko", "fabric",
}
_BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals", "vars"}
# Dangerous attribute/method names (catches os.system even via alias, Path.write_text, etc.)
_BANNED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmtree", "rmdir", "mkdir", "makedirs",
    "write_text", "write_bytes", "spawn", "spawnv", "fork", "kill", "chmod", "chown",
    "putenv", "setuid", "execv", "execve", "connect", "sendall", "urlopen",
}


def scan_code(code: str) -> str | None:
    """Return a short violation string if the code is unsafe, else None.

    SyntaxError -> None (handled by the run/retry loop, not the sandbox)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in _BANNED_MODULES:
                    return f"import {n.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _BANNED_MODULES:
                return f"from {node.module} import ..."
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None)
            if name in _BANNED_CALLS:
                # allow open(...) only if it's NOT a write mode — simplest: ban all open()
                return f"{name}() call"
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f".{node.attr}() use"
        elif isinstance(node, ast.Name):
            if node.id in ("__builtins__",):
                return "__builtins__ access"
    return None


def apply_rlimits() -> None:
    """preexec_fn for the run subprocess: cap CPU time and file size (fork/runaway guards).

    Deliberately NO RLIMIT_AS (it breaks numpy/BLAS over-reservation); the wall-clock timeout
    + RLIMIT_CPU bound runaway compute, the import denylist bounds fork bombs."""
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (1500, 1700))            # ~25 min CPU
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)   # 256 MB max file write
    except Exception:
        pass
