"""Brave Search adapter for the scout — uses the UPGRADED API features (set BRAVE_API_KEY to a paid key):
- web/search with `extra_snippets` (up to 5 excerpts/result = ~5x content), `freshness` (recent research),
  `count` (up to 20), optional `goggles_id` (custom re-ranking toward academic/practitioner sources).
- summarizer / LLM-Context endpoint (pre-extracted, LLM-ready grounding content) when entitled.
Graceful: if a feature isn't on the key's plan, Brave ignores it and we fall back to basic results.
"""
import json, os, urllib.parse, urllib.request

BASE = "https://api.search.brave.com/res/v1"
KEY = os.environ.get("BRAVE_API_KEY", "")
# Optional custom Goggle to bias toward research sources (host a goggle + set its raw URL here).
QUANT_GOGGLE = os.environ.get("BRAVE_QUANT_GOGGLE", "")  # e.g. raw github URL of a .goggle file


def _get(path, params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    req = urllib.request.Request(url, headers={"X-Subscription-Token": KEY, "Accept": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=40).read())
    except Exception as e:
        return {"_error": str(e)[:200]}


def web_search(query, count=15, freshness="py", extra_snippets=True):
    """Rich web results: each = {title, url, description, extra_snippets:[...]}. freshness py=last year."""
    d = _get("web/search", {"q": query, "count": count, "freshness": freshness,
                            "text_decorations": 0, "extra_snippets": 1 if extra_snippets else 0,
                            "goggles_id": QUANT_GOGGLE or None, "result_filter": "web"})
    out = []
    for r in (d.get("web", {}) or {}).get("results", []):
        out.append({"title": r.get("title"), "url": r.get("url"),
                    "description": r.get("description"),
                    "extra_snippets": r.get("extra_snippets", [])})
    return out, d.get("_error")


def llm_context(query, count=10):
    """Pre-extracted, LLM-ready grounding content (Brave 'LLM Context' / summarizer). Premium-tier.
    Returns a big text blob if entitled, else None (caller falls back to web_search)."""
    # Summarizer flow: web/search with summary=1 yields a summarizer key, then fetch the summary.
    d = _get("web/search", {"q": query, "count": count, "summary": 1, "extra_snippets": 1})
    sk = (d.get("summarizer", {}) or {}).get("key")
    if sk:
        s = _get("summarizer/search", {"key": sk, "entity_info": 1})
        txt = " ".join(b.get("data", "") if isinstance(b, dict) else str(b)
                       for b in (s.get("summary", []) or []))
        if txt.strip():
            return txt
    return None


def rich_search_text(query, count=12):
    """One call the scout uses: best-available content for a query (LLM-context if entitled, else
    web results + extra_snippets). Returns a compact text block for distillation."""
    ctx = llm_context(query)
    if ctx:
        return f"[LLM-CONTEXT] {ctx[:3500]}"
    results, err = web_search(query, count=count)
    if err and not results:
        return f"(search error: {err})"
    parts = []
    for r in results[:count]:
        snips = " | ".join(r.get("extra_snippets", [])[:5])
        parts.append(f"- {r['title']} ({r['url']})\n  {r.get('description','')}\n  {snips}")
    return "\n".join(parts)[:4500]
