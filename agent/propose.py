"""Generation step: read the shared wiki, propose ONE new untested hypothesis (Claude Max OAuth, $0).
Grounded in accumulated knowledge so it never re-tests closed sets or duplicates experiments."""
import json, subprocess
from pathlib import Path
from agent.config import pi_cmd

WIKI = Path("/root/research-wiki")
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def _read(p):
    f = WIKI / p
    return f.read_text() if f.exists() else ""


def propose() -> dict:
    context = (
        "=== OVERVIEW ===\n" + _read("overview.md") +
        "\n\n=== PATTERNS & ANTI-PATTERNS (obey these) ===\n" + _read("patterns/META-LESSONS.md") +
        "\n\n=== CLOSED DECISIONS (never re-open) ===\n" + _read("decisions/CLOSED.md") +
        "\n\n=== EXISTING EXPERIMENTS (do not duplicate) ===\n" + _read("index.md") +
        "\n\n=== DATA WE OWN / CAN USE (build ONLY on these; anything else is DATA-GATED -> Gate-0 FAIL) ===\n" + _read("DATA_CATALOG.md") +
        "\n\n=== WEB-SCOUTED CANDIDATES (fresh external ideas — prefer a strong one of these) ===\n" + _read("candidates.md")
    )
    prompt = f"""{context}

You are a quant research agent. Propose EXACTLY ONE new, untested strategy hypothesis to test next.
HARD CONSTRAINTS (from the wiki above):
- Must be a RISK PREMIUM or a COMBINATION of complementary premia — NOT a standalone prediction edge in a liquid market.
- Must NOT duplicate any existing experiment and must NOT violate any anti-pattern or closed decision.
- Must be data-FEASIBLE on the OWNED/FREE data in the DATA CATALOG above (for US equities PREFER survivorship-clean Sharadar SEP/SF1 via sep_panel/us_universe/sf1, NOT yfinance). If it would need a DATA-GATED source, set prior=low and state exactly what's missing in gate0_data_check.
- Prefer: combinations of validated legs, complementary premia (opposite tails), or less-efficient corners.
- DIVERSIFY — do NOT propose yet another variant of a premium that already appears 2+ times in the experiments/queue above. If one theme (e.g. PEAD/SUE) is already well-represented, pick a DIFFERENT premium or market entirely.
Return ONLY a JSON object:
{{"title": "...", "premium": "...", "market": "...", "data_source": "...", "free_or_owned": "...",
"signal_approach": "one-paragraph frozen construction", "why_not_duplicate": "...", "prior": "low|medium|high",
"pairs_with": "...", "gate0_data_check": "what to verify before building"}}"""
    r = subprocess.run(pi_cmd(), input=prompt, capture_output=True, text=True, timeout=300)
    text = _assistant_text(r.stdout)
    try:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])
    except Exception:
        return {"raw": text[:1500] or r.stdout[:800], "error": "parse_failed", "stderr": r.stderr[:300]}


def _assistant_text(stream: str) -> str:
    """Return the FULL assistant message. pi streams cumulative snapshots, so return the
    longest single assistant-text candidate (the final complete message), not a concat."""
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
            if ev.get("type") in ("agent_message", "assistant_message") and isinstance(ev.get(k), str):
                parts.append(ev[k])
    return max(parts, key=len) if parts else ""


if __name__ == "__main__":
    p = propose()
    print(json.dumps(p, indent=2)[:2000])
