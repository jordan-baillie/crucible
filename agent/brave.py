"""Brave Search adapter for the scout. Key (BSA8...) is on the Brave "Search" plan ($5/1K, 50 rps).
That plan's "LLM context = results optimized for models & agents" IS the rich web-search response itself:
clean URLs/text + up to 5 `extra_snippets`/result + news/videos. We exploit it fully:
  web/search: extra_snippets (~5x content), freshness=py (recent), count=20, optional goggles, + news/search.
The scout's OWN Claude-Max distillation (wiki-grounded + disciplined) is the LLM layer — better than a
generic summarizer. The generative "Answers" product (chat/completions/summarizer) is a SEPARATE
subscription (verified: OPTION_NOT_IN_PLAN on this key) and is NOT needed — do not pay for it.
Optional: set BRAVE_AI_KEY to a Data-for-AI key to also use Brave's summarizer (redundant with our own).
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


def news_search(query, count=5):
    """Recent news/developments (new strategy launches, vendor announcements) — with extra_snippets."""
    d = _get("news/search", {"q": query, "count": count, "extra_snippets": 1, "freshness": "py"})
    rs = d.get("results") or (d.get("news", {}) or {}).get("results") or []
    return [{"title": r.get("title"), "url": r.get("url"),
             "description": r.get("description"), "extra_snippets": r.get("extra_snippets", [])} for r in rs]


def llm_context(query, count=10):
    """Brave Summarizer / 'Data for AI' product (separate subscription, NOT on the Search plan).
    Returns None unless a Data-for-AI key is configured via BRAVE_AI_KEY. The scout's own
    Claude-Max distillation (wiki-grounded) covers this need, so this is optional."""
    if not os.environ.get("BRAVE_AI_KEY"):
        return None
    global KEY
    saved, KEY = KEY, os.environ["BRAVE_AI_KEY"]
    try:
        d = _get("web/search", {"q": query, "count": count, "summary": 1})
        sk = (d.get("summarizer", {}) or {}).get("key")
        if sk:
            s = _get("summarizer/search", {"key": sk, "entity_info": 1})
            txt = " ".join(b.get("data", "") if isinstance(b, dict) else str(b)
                           for b in (s.get("summary", []) or []))
            return txt.strip() or None
    finally:
        KEY = saved
    return None


def rich_search_text(query, count=18):
    """The scout's research call — fully exploits the upgraded SEARCH plan: count=20, extra_snippets
    (~5x content), freshness=py (recent), + recent news. The scout LLM distills it, grounded
    in our wiki/discipline. (Optional Brave summarizer auto-used only if BRAVE_AI_KEY is set.)"""
    ctx = llm_context(query)
    if ctx:
        return f"[BRAVE-AI-SUMMARY] {ctx[:3500]}"
    web, err = web_search(query, count=count, freshness="py")
    news = news_search(query, count=4)
    if err and not web:
        return f"(search error: {err})"
    parts = []
    for r in (web[:count] + news):
        snips = " | ".join((r.get("extra_snippets") or [])[:5])
        parts.append(f"- {r['title']} ({r['url']})\n  {r.get('description','')}\n  {snips}")
    return "\n".join(parts)[:6000]
