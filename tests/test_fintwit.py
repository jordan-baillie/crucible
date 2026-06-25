"""scout._fintwit — the FinTwit (twitterapi.io) layer wired into the autonomous scout as a THIRD
grounded source beside Brave (web) and Firecrawl (papers). Mocked (offline, deterministic): no live
API calls, no spend. Locks in the graceful + opt-out + integration contract so a flaky/paid source
can never break the nightly scout."""
import agent.scout as S


def test_fintwit_formats_top_tweets(monkeypatch):
    captured = {}

    def fake_search(query, query_type="Latest", limit=20):
        captured["query_type"] = query_type
        return ([{"author": "quant", "created_at": "2026-06-24", "text": "VRP edge", "likes": 9}], None)

    monkeypatch.setattr("agent.x_twitter.x_search", fake_search)
    monkeypatch.delenv("SCOUT_FINTWIT", raising=False)
    out = S._fintwit("volatility risk premium", n=5)
    assert "@quant" in out and "VRP edge" in out
    assert captured["query_type"] == "Top"      # bias to the most-engaged tweets (signal/noise)


def test_fintwit_graceful_on_backend_error(monkeypatch):
    # x_search itself is graceful (returns ([], err)); _fintwit must surface that, never raise.
    monkeypatch.setattr("agent.x_twitter.x_search", lambda *a, **k: ([], "HTTP 402: Credits is not enough"))
    monkeypatch.delenv("SCOUT_FINTWIT", raising=False)
    out = S._fintwit("x")
    assert "fintwit search failed" in out and "402" in out


def test_fintwit_graceful_on_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("agent.x_twitter.x_search", boom)
    monkeypatch.delenv("SCOUT_FINTWIT", raising=False)
    out = S._fintwit("x")
    assert "fintwit search failed" in out and "network down" in out


def test_fintwit_opt_out_skips_api(monkeypatch):
    def must_not_call(*a, **k):
        raise AssertionError("x_search must not be called when SCOUT_FINTWIT=0")
    monkeypatch.setattr("agent.x_twitter.x_search", must_not_call)
    monkeypatch.setenv("SCOUT_FINTWIT", "0")
    assert "disabled" in S._fintwit("x")
