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


def test_fintwit_sanitizes_query_before_search(monkeypatch):
    # the X leg must strip -filter:replies (verified live to ZERO results) before hitting the API
    sent = {}

    def fake_search(query, query_type="Latest", limit=20):
        sent["query"] = query
        return ([], None)

    monkeypatch.setattr("agent.x_twitter.x_search", fake_search)
    monkeypatch.delenv("SCOUT_FINTWIT", raising=False)
    S._fintwit("($VIX OR $VXX) short vol min_faves:5 lang:en -filter:replies")
    assert "-filter:replies" not in sent["query"]
    assert "min_faves:5" in sent["query"] and "$VIX" in sent["query"]    # good operators preserved


def test_sanitize_x_query():
    assert S._sanitize_x_query("$VIX short min_faves:5 -filter:replies") == "$VIX short min_faves:5"
    assert S._sanitize_x_query("$VIX min_faves:10 lang:en") == "$VIX min_faves:10 lang:en"
    assert S._sanitize_x_query("  $VIX   -filter:replies   short  ") == "$VIX short"
    assert S._sanitize_x_query(None) == ""


def test_normalize_queries_shapes():
    N = S._normalize_queries
    # new {web,x} object: parallel web/x, distinct
    assert N([{"web": "equity VRP backtest", "x": "$VIX min_faves:5"}], 4) == (
        ["equity VRP backtest"], ["$VIX min_faves:5"])
    # legacy plain string -> reused for both legs (back-compat)
    assert N(["credit spread momentum"], 4) == (["credit spread momentum"], ["credit spread momentum"])
    # web-only object -> x falls back to web
    assert N([{"web": "treasury auction"}], 4) == (["treasury auction"], ["treasury auction"])
    # x-only object -> still usable, web reuses x
    assert N([{"x": "$HYG credit"}], 4) == (["$HYG credit"], ["$HYG credit"])
    # garbage -> empty (caller raises LLMError, never a false result)
    assert N(["", {}, 123, None], 4) == ([], [])
    # capped to n_queries
    web, x = N([{"web": f"q{i}", "x": f"x{i}"} for i in range(6)], 2)
    assert web == ["q0", "q1"] and x == ["x0", "x1"]
