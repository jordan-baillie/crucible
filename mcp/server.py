"""crucible research MCP — the ideas-generation agent's external-research tools, unified.

Two ALIGNED connector families behind one MCP (stdio):
  • x_*         — FinTwit via twitterapi.io   (agent/x_twitter.py)
  • web/research/scrape/extract — Firecrawl    (agent/firecrawl.py, upgraded)

X is an IDEA source, never a price feed: mine REASONING/MECHANISMS for testable premia; the gate stack
on owned data is the sole validator (no point-in-time/survivorship problem). All tools are GRACEFUL —
a missing key or an out-of-credit (HTTP 402) backend returns an actionable message, never an exception.

Run (stdio):  crucible/mcp/.venv/bin/python crucible/mcp/server.py
Register with any MCP client (Claude Code / pi) pointing at that command.
"""
import sys
from pathlib import Path

# import the crucible connectors (stdlib-only modules) — server runs from the mcp venv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP            # noqa: E402
from agent import x_twitter as X                  # noqa: E402
from agent import firecrawl as F                  # noqa: E402

mcp = FastMCP("crucible-research")


# ── X (FinTwit) — twitterapi.io ───────────────────────────────────────────────────────────────────
@mcp.tool()
def x_search(query: str, query_type: str = "Latest", limit: int = 20) -> str:
    """Search X/Twitter (cashtags, keywords, operators e.g. '$BTC funding -is:retweet').
    query_type: 'Latest' (recent) or 'Top' (most engaged). Mine the REASONING for testable premia, not
    the directional call. Returns recent matching tweets (author, time, engagement, text)."""
    tweets, err = X.x_search(query, query_type=query_type, limit=limit)
    if err:
        return f"x_search failed: {err}" + (" — twitterapi.io account needs a top-up." if "402" in err else "")
    return X.format_tweets(tweets)


@mcp.tool()
def x_user_tweets(handle: str, limit: int = 20, include_replies: bool = False) -> str:
    """Recent tweets from one X account (its timeline). Use to read a commentator's REASONING over time."""
    tweets, err = X.x_user_tweets(handle, limit=limit, include_replies=include_replies)
    if err:
        return f"x_user_tweets failed: {err}" + (" — twitterapi.io account needs a top-up." if "402" in err else "")
    return X.format_tweets(tweets)


@mcp.tool()
def x_user_info(handle: str) -> str:
    """Profile for one X account (name, bio, follower count, verified)."""
    info, err = X.x_user_info(handle)
    if err:
        return f"x_user_info failed: {err}" + (" — twitterapi.io account needs a top-up." if "402" in err else "")
    return "\n".join(f"{k}: {v}" for k, v in info.items())


# ── Firecrawl — web research (general search + papers + scrape + structured extract) ────────────────
@mcp.tool()
def web_search(query: str, limit: int = 8) -> str:
    """General web search (news, blogs, sites) via Firecrawl. For broad context — practitioner write-ups,
    data-source discovery, market commentary. Returns title/url/snippet per result."""
    items, err = F.web_search(query, limit=limit)
    if err and not items:
        return f"web_search failed: {err}"
    return "\n".join(f"- {r['title']} ({r['url']})\n  {r.get('description') or ''}" for r in items)[:6000]


@mcp.tool()
def research_search(query: str, limit: int = 6) -> str:
    """Academic-paper search (arXiv/SSRN/ResearchGate) via Firecrawl's research category. For grounding a
    hypothesis in published factor/anomaly literature. Returns paper title/url/abstract-snippet."""
    return F.rich_research_text(query, limit=limit)


@mcp.tool()
def scrape_url(url: str, max_chars: int = 12000) -> str:
    """Fetch a single web page as clean markdown (main content) via Firecrawl. Use to READ a specific
    article/paper/page found via search."""
    md, err = F.scrape_url(url, max_chars=max_chars)
    return md if md else f"scrape_url failed: {err}"


@mcp.tool()
def extract_url(url: str, prompt: str) -> str:
    """Structured LLM extraction from a web page via Firecrawl — pass a `prompt` describing the fields to
    pull (e.g. 'extract this paper's mechanism, data, sample period, key result, caveats'). Returns JSON."""
    data, err = F.extract_url(url, prompt)
    if err and not data:
        return f"extract_url failed: {err}"
    import json
    return json.dumps(data, indent=2)[:8000]


if __name__ == "__main__":
    mcp.run()
