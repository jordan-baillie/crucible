"""agent/llm.py — THE LLM plumbing. One subprocess wrapper, one stream parser, one
JSON extractor (previously triplicated across propose/codegen/scout with divergent
timeouts and a dead copy of SYS in four files)."""
from __future__ import annotations

import json
import subprocess

from agent.config import pi_cmd


def call(prompt: str, timeout: int = 900) -> str:
    """One pi CLI call -> assistant text. Salvages partial output on timeout
    (long generations are still usually complete when the stream is cut).
    900s default: Fable-5 codegen measured 154-367s over 18 production runs
    (2026-06-10/11) and Anthropic's Fable-5 guide says individual turns run
    longer by default — the old 420s left ~50s headroom on the slowest run and
    the salvage path was silently converting near-timeouts into truncated code
    (-> false consistency failures -> wasted fix() calls)."""
    try:
        r = subprocess.run(pi_cmd(), input=prompt, capture_output=True, text=True,
                           timeout=timeout)
        return assistant_text(r.stdout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        text = assistant_text(out)
        # Salvage is a degraded path, not success — say so in the unit log so a
        # truncation seen downstream is attributable to the timeout, not the model.
        print(f"[llm] TIMEOUT at {timeout}s — salvaged {len(text)} chars of partial output", flush=True)
        return text


def assistant_text(stream: str) -> str:
    """Return the FULL assistant message from a pi JSON stream. pi streams cumulative
    snapshots, so return the longest single assistant-text candidate (the final
    complete message), not a concatenation."""
    parts = []
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        d = ev.get("delta") or {}
        if d.get("text"):
            parts.append(d["text"])
        msg = ev.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"])
        for k in ("text", "content"):
            v = ev.get(k)
            if isinstance(v, str) and v:
                parts.append(v)
    return max(parts, key=len) if parts else ""


def extract_json(text: str, open_ch: str = "{", close_ch: str = "}"):
    """First-{...last-} JSON extraction (the dance previously copy-pasted 3x).
    Returns the parsed object, or None on failure (callers decide the fallback)."""
    try:
        s, e = text.find(open_ch), text.rfind(close_ch)
        if s < 0 or e <= s:
            return None
        return json.loads(text[s:e + 1])
    except Exception:
        return None
