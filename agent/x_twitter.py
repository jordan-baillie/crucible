"""X (Twitter) connector for the ideas-generation agent — the SIBLING of agent/firecrawl.py.

Where firecrawl.py grounds generation in academic papers, this grounds it in FinTwit: query X for
keyword/cashtag SEARCH, a commentator's recent TIMELINE, or a PROFILE — via the cheapest scraper
(twitterapi.io, $0.15/1K tweets, pay-per-use, no minimums). Same conventions as firecrawl.py: thin REST,
key-from-settings, GRACEFUL (any failure returns empty + reason, never raises). Exposed to the agent as
MCP tools (mcp/server.py) alongside the firecrawl tools.

X is an IDEA source, never a price feed: the agent mines REASONING/MECHANISMS for testable premia, and the
gate stack on owned data is the sole validator — so there is no point-in-time/survivorship problem.

Backend is swappable behind _get(): twitterapi.io today (cheapest + already keyed); an Apify pay-per-result
Actor could be slotted in later. Key: env TWITTERAPI_IO_KEY -> ~/.pi/agent/settings.json twitterapi.apiKey.
"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.twitterapi.io"


def _key() -> str:
    k = os.environ.get("TWITTERAPI_IO_KEY")
    if k:
        return k
    for p in (Path.home() / ".pi/agent/settings.json", Path("/root/.pi/agent/settings.json")):
        try:
            return (json.loads(p.read_text()).get("twitterapi") or {}).get("apiKey", "") or ""
        except Exception:
            continue
    return ""


def _get(path: str, params: dict, timeout: int = 30) -> dict:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={"x-api-key": _key()})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _tweet(t: dict) -> dict:
    """Normalize a twitterapi.io tweet object (defensive about field names)."""
    a = t.get("author") or {}
    return {
        "id": t.get("id") or t.get("tweet_id"),
        "url": t.get("url") or t.get("twitterUrl"),
        "author": a.get("userName") or a.get("screen_name") or t.get("userName"),
        "created_at": t.get("createdAt") or t.get("created_at"),
        "text": t.get("text") or t.get("full_text") or "",
        "likes": t.get("likeCount", t.get("favorite_count")),
        "retweets": t.get("retweetCount", t.get("retweet_count")),
        "replies": t.get("replyCount", t.get("reply_count")),
        "views": t.get("viewCount", t.get("view_count")),
    }


def _tweets_from(d: dict) -> list:
    """Pull the tweet list out of the various shapes twitterapi.io returns."""
    for k in ("tweets", "data"):
        v = d.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and isinstance(v.get("tweets"), list):
            return v["tweets"]
    return d.get("results") if isinstance(d.get("results"), list) else []


def x_search(query: str, query_type: str = "Latest", limit: int = 20):
    """Advanced X search (cashtags/keywords/operators). query_type 'Latest'|'Top'. ~$0.15/1K tweets.
    Returns ([normalized tweet dicts], error|None). GRACEFUL."""
    if not _key():
        return [], "no twitterapi.io key"
    try:
        d = _get("twitter/tweet/advanced_search", {"query": query, "queryType": query_type})
        return [_tweet(t) for t in _tweets_from(d)[:limit] if isinstance(t, dict)], None
    except Exception as e:
        return [], str(e)[:200]


def x_user_tweets(handle: str, limit: int = 20, include_replies: bool = False):
    """A user's recent tweets (timeline). Returns ([tweet dicts], error|None). GRACEFUL."""
    if not _key():
        return [], "no twitterapi.io key"
    try:
        d = _get("twitter/user/last_tweets",
                 {"userName": handle.lstrip("@"), "includeReplies": str(include_replies).lower()})
        return [_tweet(t) for t in _tweets_from(d)[:limit] if isinstance(t, dict)], None
    except Exception as e:
        return [], str(e)[:200]


def x_user_info(handle: str):
    """A user's profile (followers/bio/etc). Returns (dict, error|None). GRACEFUL."""
    if not _key():
        return {}, "no twitterapi.io key"
    try:
        d = _get("twitter/user/info", {"userName": handle.lstrip("@")})
        u = d.get("data") or d
        return {
            "handle": u.get("userName") or handle.lstrip("@"),
            "name": u.get("name"),
            "bio": u.get("description") or u.get("bio"),
            "followers": u.get("followers") or u.get("followers_count"),
            "following": u.get("following") or u.get("friends_count"),
            "verified": u.get("isBlueVerified") or u.get("verified"),
        }, None
    except Exception as e:
        return {}, str(e)[:200]


def format_tweets(tweets) -> str:
    if not tweets:
        return "(no tweets)"
    out = []
    for t in tweets:
        eng = f"\u2665{t.get('likes')} \u21ba{t.get('retweets')}" if t.get("likes") is not None else ""
        out.append(f"- @{t.get('author')} ({t.get('created_at')}) {eng}\n  {(t.get('text') or '').strip()[:300]}")
    return "\n".join(out)[:6000]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "$BTC funding rate"
    tw, err = x_search(q, limit=5)
    print(f"search '{q}': {len(tw)} tweets (err={err})")
    print(format_tweets(tw))
