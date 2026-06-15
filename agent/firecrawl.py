"""Firecrawl layer for the scout: (1) research-category SEARCH (academic-paper frontier) and
(2) DEEP-EXTRACT (full methodology/data of top open-access papers, beyond the abstract).

agent/brave.py gives broad web+news snippets; this adds Firecrawl `/search categories:["research"]`
(arXiv/SSRN/ResearchGate papers) plus a `/scrape` JSON-mode deep dive on the top OPEN-ACCESS (arXiv)
papers to pull structured {mechanism, methodology, data_used, sample_period, key_result, caveats}.
Both feed the SAME wiki-grounded Claude-Max distillation. Additive + graceful: any failure degrades
to a no-op and never breaks the nightly scout. Cost: ~2 credits/research-search + ~5 credits/deep dive
(capped at 2/run) -> ~18 credits/run.

NOTE: deep extraction uses `/scrape` JSON-mode (synchronous, ~5 credits, the docs-recommended tool for
KNOWN open arXiv URLs). FIRE-1 `/agent` IS available on this plan (account healthy: ~5000 credits/mo)
but is far more expensive and unnecessary here: it does a pre-flight cost estimate and refuses
('Refusal: Agent reached max credits', 0 used) whenever maxCredits < that estimate, and it estimates
academic-research tasks at >200 credits each (>=40x /scrape) — too costly for routine scout use, and its
autonomous URL-discovery is redundant with our Brave+Firecrawl search layer (which already yields the
URLs). Reserve /agent only for a rare autonomous no-URL task worth ~200+ credits. Key: env
FIRECRAWL_API_KEY -> ~/.pi/agent/settings.json firecrawl.apiKey.
"""
import json
import os
import urllib.request
from pathlib import Path

BASE = "https://api.firecrawl.dev/v2"

# Frozen deep-extract schema (the structured detail the smith's distillation gets beyond the abstract).
EXTRACT_SCHEMA = {"type": "object", "properties": {
    "mechanism": {"type": "string"}, "methodology": {"type": "string"},
    "data_used": {"type": "string"}, "sample_period": {"type": "string"},
    "key_quantitative_result": {"type": "string"}, "caveats_or_costs": {"type": "string"}}}

# Domains with reliably OPEN full text (deep-extract is worth 5 credits). Gated domains
# (researchgate/sciencedirect/wiley/springer) expose only the abstract — no better than the search
# snippet — so we DON'T scrape them (would waste credits).
_OPEN_DOMAINS = ("arxiv.org",)


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


def _post(path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BASE}/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def is_extractable(url: str) -> bool:
    """True only for reliably OPEN full-text domains (deep-extract worth the credits)."""
    u = (url or "").lower()
    return any(d in u for d in _OPEN_DOMAINS)


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
        return [{"title": r.get("title"), "url": r.get("url"),
                 "description": r.get("description"), "category": r.get("category")}
                for r in items if isinstance(r, dict)], None
    except Exception as e:
        return [], str(e)[:200]


def format_research(items) -> str:
    if not items:
        return "(no research papers found)"
    return "\n".join(f"- [PAPER] {r['title']} ({r['url']})\n  {r.get('description', '')}"
                     for r in items)[:5000]


def rich_research_text(query: str, limit: int = 6) -> str:
    items, err = research_search(query, limit=limit)
    if err and not items:
        return f"(firecrawl research search unavailable: {err})"
    return format_research(items)


def deep_extract(url: str):
    """Full structured methodology/data from a known OPEN-ACCESS paper URL via /scrape JSON-mode.
    Synchronous, ~5 credits. Returns dict (EXTRACT_SCHEMA) or None. No-op on non-open / missing key."""
    if not _key() or not is_extractable(url):
        return None
    try:
        d = _post("scrape", {"url": url, "onlyMainContent": True, "formats": [{
            "type": "json",
            "prompt": ("Extract this finance/quant research paper's mechanism, methodology, data "
                       "sources, sample period, key quantitative result, and any caveats or costs."),
            "schema": EXTRACT_SCHEMA}]})
        return (d.get("data") or {}).get("json") or None
    except Exception:
        return None


def format_deep(item, data) -> str:
    title = (item or {}).get("title", "paper")
    url = (item or {}).get("url", "")
    fields = "; ".join(f"{k}: {str(v)[:380]}" for k, v in (data or {}).items() if v)
    return f"- [DEEP DIVE] {title} ({url})\n  {fields}"[:2000]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "cross-sectional cryptocurrency momentum factor backtest"
    items, _ = research_search(q)
    print(format_research(items))
    for p in items:
        if is_extractable(p.get("url", "")):
            print("\n", format_deep(p, deep_extract(p["url"])))
            break
