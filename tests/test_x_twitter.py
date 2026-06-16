"""X (twitterapi.io) connector — sibling of the Firecrawl layer. Graceful + correct parsing.
Mocked (offline, deterministic): no live API calls, no spend."""
import pytest

from agent import x_twitter as X


@pytest.fixture()
def keyed(monkeypatch):
    monkeypatch.setattr(X, "_key", lambda: "test-key")
    return monkeypatch


def test_no_key_is_graceful(monkeypatch):
    monkeypatch.setattr(X, "_key", lambda: "")
    assert X.x_search("$BTC") == ([], "no twitterapi.io key")
    assert X.x_user_tweets("foo") == ([], "no twitterapi.io key")
    assert X.x_user_info("foo") == ({}, "no twitterapi.io key")


def test_backend_error_is_graceful(keyed):
    def boom(*a, **k):
        raise RuntimeError("HTTP Error 402: Payment Required")
    keyed.setattr(X, "_get", boom)
    tweets, err = X.x_search("$BTC")
    assert tweets == [] and "402" in err          # graceful, surfaces the reason (MCP maps -> 'top-up')


def test_tweet_normalization_defensive():
    t = X._tweet({"id": "1", "url": "u", "text": "hi", "createdAt": "2026-06-16",
                  "likeCount": 5, "retweetCount": 2, "author": {"userName": "bob"}})
    assert t == {"id": "1", "url": "u", "author": "bob", "created_at": "2026-06-16",
                 "text": "hi", "likes": 5, "retweets": 2, "replies": None, "views": None}


@pytest.mark.parametrize("payload", [
    {"tweets": [{"text": "a"}, {"text": "b"}]},
    {"data": {"tweets": [{"text": "a"}, {"text": "b"}]}},
    {"data": [{"text": "a"}, {"text": "b"}]},
    {"results": [{"text": "a"}, {"text": "b"}]},
])
def test_tweets_from_handles_all_shapes(payload):
    assert len(X._tweets_from(payload)) == 2


def test_x_search_parses_and_limits(keyed):
    keyed.setattr(X, "_get", lambda *a, **k: {"tweets": [{"text": f"t{i}", "author": {"userName": "x"}}
                                                          for i in range(50)]})
    tweets, err = X.x_search("q", limit=5)
    assert err is None and len(tweets) == 5 and tweets[0]["author"] == "x"


def test_x_user_info_parses(keyed):
    keyed.setattr(X, "_get", lambda *a, **k: {"data": {"userName": "bob", "name": "Bob",
                                                       "description": "trader", "followers": 1000}})
    info, err = X.x_user_info("bob")
    assert err is None and info["handle"] == "bob" and info["followers"] == 1000 and info["bio"] == "trader"


def test_format_tweets_renders():
    out = X.format_tweets([{"author": "bob", "created_at": "2026-06-16", "likes": 3, "retweets": 1,
                            "text": "negative funding = crowded short"}])
    assert "@bob" in out and "crowded short" in out
