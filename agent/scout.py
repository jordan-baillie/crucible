"""Scout: the 'ingest external sources' step. Searches the web for NEW edges/strategies based on
wiki gaps, distills findings into the wiki, and queues testable candidates for the propose step.
This makes generation OPEN (discovers what others run) instead of only recombining what we know.
Pattern: karpathy LLM-Wiki 'ingest' + a research scout. LLM via the pi CLI; web via Brave (broad
web+news snippets) AND Firecrawl `categories:[research]` (arXiv/SSRN/ResearchGate academic papers).
Both feed the SAME wiki-grounded Claude-Max distillation; Firecrawl is additive + graceful (never
breaks the scout if its key/endpoint is down) AND twitterapi.io FinTwit search (practitioner chatter; SCOUT_FINTWIT=0 to
opt out). Each source is graceful: a failure degrades that source, never breaks the scout.
~8 Firecrawl credits/run + ~12 tweets/query FinTwit (sub-cent)."""
import json
import os
from datetime import date
from pathlib import Path
from agent.llm import call as _llm_call, extract_json, LLMError

from crucible_paths import WIKI  # central config


def _read(p):
    f = WIKI / p
    return f.read_text(encoding="utf-8") if f.exists() else ""


def _read_tail(p, max_chars: int):
    """Bounded read (newest-last) so the scout prompt does not grow without limit with project
    history. A 31KB+ index.md bloats the distill prompt toward the call timeout; a timed-out call
    salvages TRUNCATED JSON -> unparseable -> (historically) a SILENT 0-candidate night. Capping the
    tested-list removes that trigger. Mirrors propose._read_tail (same problem, same fix)."""
    t = _read(p)
    if len(t) <= max_chars:
        return t
    cut = t[-max_chars:]
    nl = cut.find("\n")
    return f"[... older entries omitted ({len(t) - max_chars} chars) ...]\n" + (cut[nl + 1:] if nl >= 0 else cut)


def _pi(prompt):
    return _llm_call(prompt, timeout=900)  # Fable-5 turns run long; match agent.llm default (2026-06-12)


def _brave(query, n=5):
    try:
        from agent.brave import rich_search_text
        return rich_search_text(query)
    except Exception as e:
        return f"(search failed: {e})"


def _research(query):
    """(text, items) from the Firecrawl research-paper layer. Graceful: never breaks the scout."""
    try:
        from agent.firecrawl import research_search, format_research
        items, _ = research_search(query)
        return format_research(items), items
    except Exception as e:
        return f"(research search failed: {e})", []


def _sanitize_x_query(q):
    """Strip X-search operators that return 0 results on the twitterapi.io backend (verified live
    2026-06-25: a trailing `-filter:replies` ZEROES the result set; every other operator — cashtags,
    min_faves:, lang:, OR-groups — works). Belt-and-braces so an LLM that ignores the prompt guidance
    still gets a working FinTwit query; the gate stack validates ideas regardless of their source."""
    import re
    q = re.sub(r"\s*-filter:\S+", "", q or "")     # negated filters zero the set on this backend
    return re.sub(r"\s+", " ", q).strip()


def _fintwit(query, n=12):
    """(text) from the FinTwit layer (twitterapi.io x_search) — practitioner chatter as a THIRD
    grounded source alongside Brave (broad web) and Firecrawl (papers). 'Top' = the most-engaged
    tweets (higher signal/noise than 'Latest'). IDEA source ONLY: the agent mines reasoning/
    mechanisms for testable premia; the gate stack on owned data is the sole validator (so there is
    no point-in-time/survivorship concern). Graceful: never breaks the scout. Opt out: SCOUT_FINTWIT=0
    (a paid call ~$0.15/1K tweets; ~12 tweets/query here -> sub-cent/run, but kept toggleable)."""
    if os.environ.get("SCOUT_FINTWIT", "1").strip().lower() in ("0", "false", "no", "off"):
        return "(fintwit disabled via SCOUT_FINTWIT)"
    try:
        from agent.x_twitter import x_search, format_tweets
        tw, err = x_search(_sanitize_x_query(query), query_type="Top", limit=n)
        return format_tweets(tw) if not err else f"(fintwit search failed: {err})"
    except Exception as e:
        return f"(fintwit search failed: {e})"


def _deep_dive(papers_all, n=2):
    """Deep-extract full methodology/data of up to n OPEN-ACCESS (arXiv) papers via /scrape JSON-mode."""
    try:
        from agent.firecrawl import deep_extract, is_extractable, format_deep
    except Exception:
        return ""
    out, seen = [], set()
    for p in papers_all:
        if len(out) >= n:
            break
        u = (p or {}).get("url", "")
        if u and u not in seen and is_extractable(u):
            seen.add(u)
            try:
                data = deep_extract(u)
                if data:
                    out.append(format_deep(p, data))
            except Exception:
                pass
    return ("\n\n### DEEP DIVES (full methodology/data of top open-access papers)\n" +
            "\n\n".join(out)) if out else ""


def _json(text):
    for op, cl in (("{", "}"), ("[", "]")):
        obj = extract_json(text, op, cl)
        if obj is not None:
            return obj
    return None


def _normalize_queries(raw, n_queries):
    """Normalize the query-gen output into parallel (web_queries, x_queries).

    The query-gen now emits one object per query: {"web": broad web/paper search string,
    "x": the SAME premium as an X advanced-search string (cashtags + operators)}. This accepts
    that shape AND legacy plain strings (back-compat) AND an x-only object — the X leg falls back
    to the web string whenever no X-native variant is given, so Brave/Firecrawl/FinTwit each get a
    usable query. Returns ([], []) when nothing is usable (the caller raises LLMError — a parse/
    timeout failure must never masquerade as a real result).
    """
    web, x = [], []
    for item in (raw or []):
        if isinstance(item, str) and item.strip():
            web.append(item.strip()); x.append(item.strip())
        elif isinstance(item, dict):
            w = str(item.get("web") or item.get("query") or "").strip()
            xv = str(item.get("x") or item.get("twitter") or "").strip()
            if w:
                web.append(w); x.append(xv or w)       # X-native variant, else reuse web
            elif xv:                                      # x-only item is still a usable query
                web.append(xv); x.append(xv)
    return web[:n_queries], x[:n_queries]


def _diversity_brief() -> str:
    """Generator-only steer: the already-spent / already-live family-space (agent.joint_state.brief).
    Graceful — returns '' when there is nothing to steer against (fresh machine / deploy disabled)."""
    try:
        from agent.joint_state import brief
        return brief()
    except Exception:
        return ""


def _agentic_enabled() -> bool:
    return os.environ.get("SCOUT_AGENTIC", "0").strip().lower() in ("1", "true", "yes", "on")


def _scout_agentic(ctx: str, n_queries: int) -> dict:
    """Stage 3: ONE agentic Fable-5 turn that DRIVES the crucible-research MCP itself (search -> READ
    the source -> cross-check vs the wiki's tested/closed families) and returns the SAME distill JSON
    as the tool-less path. This is the mcp/README 'future agentic X-gather scout'. Fail-loud preserved:
    an unparseable/empty result (incl. a Fable-5 refusal -> empty completion) RAISES rather than logging
    a false 0-candidate night (the 2026-06-22 lesson)."""
    from agent.config import scout_cmd
    prompt = (f"{ctx}\n\nYou are a quant research scout with LIVE research tools: web_search (broad web), "
              f"research_search (arXiv/SSRN/ResearchGate papers), scrape_url/extract_url (READ a specific "
              f"page/paper), x_search (FinTwit practitioner chatter — IDEA source only). "
              f"Find up to {n_queries} NEW, specific, backtestable edges real practitioners or recent "
              f"research actually use. DIVERSIFY HARD across DISTINCT premia AND markets; prefer edges "
              f"buildable on the OWNED data above; AVOID anything already tested/closed/deployed. "
              f"USE the tools to CHASE and VERIFY each mechanism — read the actual source and confirm it "
              f"is genuinely distinct from the wiki's tested/closed families — BEFORE proposing it. "
              f"When done, return ONLY the distill JSON (no prose):\n"
              f'{{"summary": "2-3 sentences on what is new/relevant", '
              f'"candidates": [{{"title":"...","premium":"...","market":"...","why_promising":"...",'
              f'"data_feasible":"free/owned?","not_already_tested":"...","source":"..."}}], '
              f'"premia_updates": ["short factual updates with source"], '
              f'"contradictions": ["any finding that contradicts a wiki claim"]}}')
    text = _llm_call(prompt, timeout=1800, cmd=scout_cmd())  # agentic turns run long; --max-turns is the bound
    findings = _json(text)
    if not isinstance(findings, dict) or "candidates" not in findings:
        raise LLMError(f"agentic scout returned unparseable output (len {len(text)}); "
                       f"refusing to log a false 0-candidate night")
    return findings


def scout(n_queries=4):
    ctx = ("=== OVERVIEW ===\n" + _read("overview.md") +
           "\n\n=== ANTI-PATTERNS (avoid) ===\n" + _read("patterns/META-LESSONS.md")[:2500] +
           "\n\n=== DATA WE OWN (prefer ideas buildable on these) ===\n" + _read("DATA_CATALOG.md") +
           "\n\n=== ALREADY TESTED / SURFACED (recent; older omitted) ===\n" + _read_tail("index.md", 12_000)
           + _diversity_brief())
    # Stage 3: agentic path (SCOUT_AGENTIC=1) — Fable-5 drives the MCP itself; same candidates.md
    # contract out, so the whole downstream (dedup/closed-family/theme-cap/gate stack) is unchanged.
    if _agentic_enabled():
        from agent.config import SCOUT_MODEL
        findings = _scout_agentic(ctx, n_queries)
        _ingest([f"agentic-scout via {SCOUT_MODEL}"], findings)
        return findings
    # 1. generate targeted search queries aimed at wiki gaps / untested promising directions
    q_raw = _pi(f"{ctx}\n\nYou are a quant research scout. Based on the wiki's UNTESTED promising "
                f"directions and gaps, produce {n_queries} web-search queries that would surface NEW, "
                f"specific, backtestable strategies/edges that real practitioners or recent research "
                f"actually use (risk premia, structural edges, combinations) — NOT generic. "
                f"DIVERSIFY HARD: each query a DISTINCT premium AND market — span e.g. one equity "
                f"factor/event, one rates/credit, one volatility, one cross-asset/commodity/FX. Do NOT "
                f"cluster (the wiki is ALREADY heavy on crypto funding-carry and PEAD — deliberately look "
                f"ELSEWHERE). Prefer edges buildable on the OWNED data above. Avoid anything tested/closed. "
                f"Return ONLY a JSON array of objects, one per query: "
                f'[{{"web": "broad web/paper search string", '
                f'"x": "the SAME premium as an X/Twitter advanced-search string. Use a CASHTAG OR-group '
                f"($VIX OR $VXX), AT MOST 1-2 key terms, and the operators min_faves:5 (signal filter) + "
                f"lang:en. Keep it LOOSE — a long AND-chain of keywords returns nothing. Do NOT use "
                f'-filter:replies (it zeroes results). No URLs."}}]. "x" targets FinTwit; "web" targets Brave + papers.')
    raw = _json(q_raw)
    queries, x_queries = _normalize_queries(raw, n_queries)
    if not queries:
        # A truncated/timed-out/garbled query-gen call used to silently fall back to two HARDCODED
        # queries (one of them crypto) — masking a dead LLM call AND re-injecting a de-pinned focus.
        # Fail LOUD instead: the director catches scout() non-fatally, so the night is attributable
        # (logged as 'scout failed') rather than written as a real result built on canned queries.
        raise LLMError(f"scout query-gen returned no parseable queries (output {len(q_raw)} chars)")
    # 2. search — Brave (broad web+news) + Firecrawl research papers + FinTwit practitioner chatter;
    #    deep-dive top open-access papers. Each source is graceful: a failure degrades, never breaks.
    blocks, papers_all = [], []
    for q, xq in zip(queries, x_queries):
        ptxt, pitems = _research(q)
        papers_all += pitems
        blocks.append(f"### QUERY: {q}\n{_brave(q)}\n\n[ACADEMIC PAPERS — research frontier]\n{ptxt}"
                      f"\n\n[FINTWIT — practitioner chatter for `{xq}` (IDEA source only; validate on owned data)]"
                      f"\n{_fintwit(xq)}")
    results = "\n\n".join(blocks) + _deep_dive(papers_all)
    # 3. distill into structured findings + testable candidates (with sources), flag contradictions
    d_raw = _pi(f"{ctx}\n\n=== WEB SEARCH RESULTS ===\n{results}\n\n"
                f"Distill these into NEW knowledge for the wiki. Return ONLY JSON:\n"
                f'{{"summary": "2-3 sentences on what is new/relevant", '
                f'"candidates": [{{"title":"...","premium":"...","market":"...","why_promising":"...",'
                f'"data_feasible":"free/owned?","not_already_tested":"...","source":"..."}}], '
                f'"premia_updates": ["short factual updates to add to premia/market pages, with source"], '
                f'"contradictions": ["any finding that contradicts a wiki claim"]}}\n'
                f'Surface DIVERSE candidates across DIFFERENT premia/markets (NOT multiple variants of one '
                f'theme); PREFER ideas buildable on the OWNED data above (mark data_feasible accordingly).')
    findings = _json(d_raw)
    if not isinstance(findings, dict) or "candidates" not in findings:
        # PARSE/TIMEOUT FAILURE: truncated JSON won't parse -> _json returns None. The historical
        # code coerced that to {"candidates": []} and _ingest logged '0 candidates' — indistinguishable
        # from a genuine empty result. The 2026-06-22 STORM of identical 0-candidate logs was exactly
        # this (the pi->summon auth migration breaking every call). NEVER record a failed distill as a
        # real null: raise so it is attributable and writes NO misleading 0-candidate ingest. A GENUINE
        # empty result (valid JSON with candidates: []) still falls through and ingests honestly.
        raise LLMError(f"scout distill returned unparseable output (len {len(d_raw)}); "
                       f"refusing to log a false 0-candidate night")
    _ingest(queries, findings)
    return findings


def _ingest(queries, findings):
    """Write the distilled web findings into the wiki (sources page + candidates queue + log)."""
    today = date.today()
    src = WIKI / "sources" / f"{today}-scout.md"
    src.parent.mkdir(exist_ok=True)
    cand_lines = "\n".join(
        f"- **{c.get('title')}** ({c.get('premium')}, {c.get('market')}) — {c.get('why_promising')} "
        f"[data: {c.get('data_feasible')}] src: {c.get('source')}" for c in findings.get("candidates", []))
    src.write_text(f"""---
type: scout-ingest
date: {today}
queries: {queries}
---
# Web Scout — {today}

## Summary
{findings.get('summary','')}

## New testable candidates (queued for generation)
{cand_lines or '(none surfaced)'}

## Premia/market updates
{chr(10).join('- ' + u for u in findings.get('premia_updates', [])) or '(none)'}

## Contradictions with wiki claims (review)
{chr(10).join('- ⚠️ ' + c for c in findings.get('contradictions', [])) or '(none)'}
""")
    # candidates queue that propose reads
    cq = WIKI / "candidates.md"
    with open(cq, "a", encoding="utf-8") as f:
        f.write(f"\n## [{today}] web-scout candidates\n{cand_lines}\n")
    with open(WIKI / "log.md", "a", encoding="utf-8") as f:
        f.write(f"\n## [{today}] ingest | web-scout: {len(findings.get('candidates',[]))} candidates, "
                f"{len(findings.get('contradictions',[]))} contradictions -> sources/{today}-scout.md")
    print(f"[scout] ingested {len(findings.get('candidates',[]))} candidates, "
          f"{len(findings.get('contradictions',[]))} contradictions -> wiki")


if __name__ == "__main__":
    f = scout()
    print(json.dumps(f, indent=2)[:1800])
