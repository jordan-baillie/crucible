# crucible research MCP

The ideas-generation agent's **external-research tools**, unified behind one MCP (stdio). Two aligned
connector families:

| family | backend | tools |
|---|---|---|
| **X / FinTwit** | twitterapi.io ($0.15/1K, pay-per-use) — `agent/x_twitter.py` | `x_search`, `x_user_tweets`, `x_user_info` |
| **Web research** | Firecrawl (upgraded) — `agent/firecrawl.py` | `web_search`, `research_search` (papers), `scrape_url`, `extract_url` |

**X is an IDEA source, never a price feed.** The agent mines REASONING/MECHANISMS for testable premia;
the gate stack on owned data is the sole validator — so there is no point-in-time / survivorship problem.
All tools are **graceful**: a missing key or out-of-credit backend returns an actionable message, not an error.

## Register with an MCP client (Claude Code / pi)
```json
{
  "mcpServers": {
    "crucible-research": {
      "command": "/root/crucible/mcp/run.sh"
    }
  }
}
```
(or `"command": "/root/crucible/mcp/.venv/bin/python", "args": ["/root/crucible/mcp/server.py"]`)

## Keys
- **twitterapi.io** — `~/.pi/agent/settings.json` `twitterapi.apiKey` (or env `TWITTERAPI_IO_KEY`).
  ⚠️ The account must hold a prepaid balance; an empty balance returns **HTTP 402** and the X tools report
  "twitterapi.io account needs a top-up." (Top up at twitterapi.io.)
- **Firecrawl** — `~/.pi/agent/settings.json` `firecrawl.apiKey` (already funded).

## Swappable X backend
`agent/x_twitter._get()` is the single swap point. To switch from twitterapi.io to an Apify pay-per-result
Actor (e.g. `xquik/x-tweet-scraper`, also $0.15/1K with a ~$5/mo free credit), reimplement `_get()` against
the Apify run-sync endpoint — the tool surface and MCP are unchanged.

## Setup (already done)
- venv: `python3 -m venv mcp/.venv && mcp/.venv/bin/pip install mcp` (gitignored).
- The connector modules (`agent/x_twitter.py`, `agent/firecrawl.py`) are stdlib-only and importable by the
  forge too; the MCP just exposes them to agentic clients.

## Note on the nightly forge
The nightly forge scout runs **non-agentically** (`--no-tools`, by design). This MCP is for **agentic**
ideas-generation (interactive research sessions, or a future agentic "X-gather" scout step). Wiring it into
the nightly forge is a deliberate follow-on, not done here.
