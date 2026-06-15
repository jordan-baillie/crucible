"""Firecrawl research-search adapter for the scout — the ACADEMIC-PAPER frontier layer.

agent/brave.py gives broad web + news snippets; this ADDS Firecrawl's `/search` with
`categories:["research"]`, which surfaces actual arXiv / SSRN / ResearchGate papers (with descriptive
snippets carrying real backtest stats) that general web search misses. It is ADDITIVE and graceful:
the scout runs Brave AND this, both feed the SAME Claude-Max distillation; if the key/endpoint is
unavailable this degrades to a no-op string and never breaks the nightly scout.

Cost: ~2 credits per query (search-only, no page scrape). ~8 credits / nightly scout run (4 queries).
Key resolution: env FIRECRAWL_API_KEY -> ~/.pi/agent/settings.json firecrawl.apiKey -> /root/.pi/...
The scout's own wiki-grounded Claude-Max distillation remains the LLM layer (Firecrawl is web I/O only).
"""
import json
import os
import urllib.request
from pathlib import Path

BASE = "https://api.firecrawl.dev/v2"


def _key() -> str:
    k = os.environ.get("FIRECRAWL_API_KEY")
    if k:
        return k
    for p in (Path.home() / ".pi/agent/settings.json", Path("/root/.pi/agent/settings.json")):
        try:
            return (json.loads(p.read_text()).get("firecrawl") or {}).get("apiKey", "") or ""
        except Exception:
            continue
    return ""


def _post(path: str, body: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        f"{BASE}/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def research_search(query: str, limit: int = 6, tbs: str = "qdr:y"):
    """Firecrawl /search restricted to the RESEARCH (academic-paper) category, recency-filtered (tbs).
    Returns ([{title,url,description,category}], error|None). ~2 credits/call (search-only)."""
    if not _key():
        return [], "no firecrawl key"
    try:
        d = _post("search", {"query": query, "categories": ["research"], "sources": ["web"],
                             "tbs": tbs, "limit": limit})
        data = d.get("data") or {}
        items = data.get("web") or data.get("research") or (data if isinstance(data, list) else [])
        out = [{"title": r.get("title"), "url": r.get("url"),
                "description": r.get("description"), "category": r.get("category")}
               for r in items if isinstance(r, dict)]
        return out, None
    except Exception as e:
        return [], str(e)[:200]


def rich_research_text(query: str, limit: int = 6) -> str:
    """Formatted academic-paper results for the scout's distillation (mirrors brave.rich_search_text)."""
    items, err = research_search(query, limit=limit)
    if err and not items:
        return f"(firecrawl research search unavailable: {err})"
    if not items:
        return "(no research papers found)"
    return "\n".join(f"- [PAPER] {r['title']} ({r['url']})\n  {r.get('description', '')}"
                     for r in items)[:5000]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "cross-sectional cryptocurrency momentum factor backtest"
    print(rich_research_text(q))
