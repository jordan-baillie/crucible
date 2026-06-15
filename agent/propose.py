"""Generation step: read the shared wiki, propose ONE new untested hypothesis (LLM via the pi CLI).
Grounded in accumulated knowledge so it never re-tests closed sets or duplicates experiments."""
import json  # noqa: F401 (used by prompt formatting)
import os
from pathlib import Path
from agent.llm import call as _llm_call, assistant_text as _assistant_text, extract_json  # noqa: F401 (re-export for legacy importers)

from crucible_paths import WIKI  # central config
SYS = "You are Claude Code, Anthropic's official CLI for Claude."


def _read(p):
    f = WIKI / p
    return f.read_text(encoding="utf-8") if f.exists() else ""


def _focus() -> str:
    """Operator-directed SEARCH FOCUS (env CRUCIBLE_FOCUS), injected into every arm. Reversible:
    unset/empty = the general retail-deployable bias; 'crypto' = hunt crypto-deployable space only."""
    if os.environ.get("CRUCIBLE_FOCUS", "").strip().lower() != "crypto":
        return ""
    return (
        "\n\n=== SEARCH FOCUS: CRYPTO (operator-directed 2026-06-15 — overrides the retail-equity bias) ===\n"
        "Propose CRYPTO-deployable strategies ONLY this run:\n"
        "- Venue: spot + PERPETUAL futures on Binance/Bybit; the LIQUID MAJORS (BTC/ETH/SOL/BNB/XRP and other deep perps).\n"
        "- Crypto deployability: perps short FREELY (NO stock-borrow constraint — a crypto long/short IS deployable),\n"
        "  ~20bps round-trip taker cost, <=2x leverage, no options. Set market='crypto', retail_tradable_5k='yes'.\n"
        "- Data reachable (adapters only — NEVER raw I/O): funding_rates() (perp funding, majors, 2019+);\n"
        "  binance_klines(CRYPTO_MAJORS, market='perp'|'spot') = daily OHLCV + volume + trades + TAKER_BUY_QUOTE\n"
        "  (deep-history flow/positioning proxy) for 12 liquid majors -> basis (perp vs spot), momentum/reversal,\n"
        "  realized vol, liquidity tiers, taker-flow crowding; yf_panel for extra spot. ⚠ Binance OI + long/short\n"
        "  ratio are LAST-30-DAYS only (NOT backtestable -> DATA-GATED); use taker_buy_quote for deep-history flow.\n"
        "- DORMANT — do NOT just re-propose delta-neutral FUNDING CARRY: funding has compressed to ~0/negative in 2025-26\n"
        "  (see markets/crypto.md); it earns nothing today. Naive crypto MOMENTUM has FAILED before.\n"
        "- LOOK BEYOND CARRY for something that could pay in the CURRENT regime: cross-sectional crypto factors\n"
        "  (momentum/reversal/low-vol across coins), basis- or taker-flow/crowding-CONDITIONAL timing, vol/term-structure,\n"
        "  illiquidity in alts, regime gates. PREFER conditional/combination constructions over naive single signals.\n"
    )


def _read_tail(p, max_chars: int):
    """E5: growing files (index, candidates) are read NEWEST-LAST and capped — prompt size must
    not rise monotonically with project history (cost, latency, 420s-timeout risk; the director
    makes up to 12 propose calls per top-up). Curated files (overview, lessons, closed, catalog)
    stay full: they are pruned by the weekly lint, not by truncation — dropping an anti-pattern
    or a closed decision from the prompt would un-learn it."""
    t = _read(p)
    if len(t) <= max_chars:
        return t
    cut = t[-max_chars:]
    nl = cut.find("\n")  # start at a whole line
    return f"[... older entries omitted ({len(t) - max_chars} chars) ...]\n" + (cut[nl + 1:] if nl >= 0 else cut)


def propose() -> dict:
    context = (
        "=== OVERVIEW ===\n" + _read("overview.md") +
        "\n\n=== PATTERNS & ANTI-PATTERNS (obey these) ===\n" + _read("patterns/META-LESSONS.md") +
        "\n\n=== CLOSED DECISIONS (never re-open) ===\n" + _read("decisions/CLOSED.md") +
        "\n\n=== EXISTING EXPERIMENTS (do not duplicate; newest last, older omitted) ===\n" + _read_tail("index.md", 12_000) +
        "\n\n=== DATA WE OWN / CAN USE (build ONLY on these; anything else is DATA-GATED -> Gate-0 FAIL) ===\n" + _read("DATA_CATALOG.md") +
        "\n\n=== WEB-SCOUTED CANDIDATES (fresh external ideas — prefer a strong one of these) ===\n" + _read_tail("candidates.md", 8_000)
        + _focus()
    )
    prompt = f"""{context}

You are a quant research agent. Propose EXACTLY ONE new, untested strategy hypothesis to test next.
HARD CONSTRAINTS (from the wiki above):
- Must be a RISK PREMIUM or a COMBINATION of complementary premia — NOT a standalone prediction edge in a liquid market.
- Must NOT duplicate any existing experiment and must NOT violate any anti-pattern or closed decision.
- Must be data-FEASIBLE on the OWNED/FREE data in the DATA CATALOG above (for US equities PREFER survivorship-clean Sharadar SEP/SF1 via sep_panel/us_universe/sf1, NOT yfinance). If it would need a DATA-GATED source, set prior=low and state exactly what's missing in gate0_data_check.
- Prefer: combinations of validated legs, complementary premia (opposite tails), or less-efficient corners.
- DIVERSIFY — do NOT propose yet another variant of a premium that already appears 2+ times in the experiments/queue above. If one theme (e.g. PEAD/SUE) is already well-represented, pick a DIFFERENT premium or market entirely.
- CROWDING/DECAY — PREFER novel inefficiencies in less-arbitraged corners over heavily-published factors. Famous factors (betting-against-beta/low-vol, value, momentum, size) are CROWDED -> decayed + regime-fragile (cf. BAB: passed every single-universe gate but FAILED cross-market and is heavily published). If you propose a known premium, it must be a genuinely under-exploited IMPLEMENTATION or corner.
- DEPLOYABILITY (board 2026-06-09 — avoid STRANDED ALPHA) — a pass is only valuable if it can be TRADED at retail scale (~$5K, IB/Alpaca, no special borrow). STRONGLY PREFER: long-only or long-tilt equity books in liquid-enough names, futures-implementable premia (IB micro-futures: equity index, rates, FX, metals, energy), ETF-implementable cross-asset books, crypto perps (small size). AVOID proposing strategies whose construction REQUIRES: shorting illiquid/hard-to-borrow names (micro-cap short legs are NOT borrowable at retail), gross leverage >2x, intraday execution, or options/OTC structures we cannot route. A long/short CROSS-SECTIONAL design is acceptable ONLY if the short leg is liquid large/mid-caps or an index hedge (e.g. short SPY/sector ETF vs the long book). If the cleanest test of the premium is long-short but a deployable long-only/index-hedged variant exists, PROPOSE THE DEPLOYABLE VARIANT.
Return ONLY a JSON object:
{{"title": "...", "premium": "...", "market": "...", "data_source": "...", "free_or_owned": "...",
"signal_approach": "one-paragraph frozen construction", "why_not_duplicate": "...", "prior": "low|medium|high",
"pairs_with": "...", "gate0_data_check": "what to verify before building",
"crowding_risk": "low|medium|high — how heavily-published/arbitraged is this edge? (favour low)",
"retail_tradable_5k": "yes|no — can THIS construction be executed at ~$5K via IB/Alpaca (instruments routable, short leg borrowable or index-hedged, no >2x gross leverage)? If no, this proposal will be DOWN-RANKED — prefer redesigning it to a deployable variant",
"scope": "broad|local — broad if a UNIVERSAL mechanism (theory says it appears across markets; a pass must later GENERALISE) or local if defensibly universe-specific (then forward-validation confirms it)",
"generalization_plan": "if broad: the untouched universes to confirm the mechanism in (e.g. other cap-tiers/sectors/asset-classes); if local: the economic reason it lives ONLY in this universe + the forward-validation plan"}}"""
    # 420s: Fable-5 measured 285s on a real propose — 300s left only ~15s headroom (2026-06-10 smoke test)
    text = _llm_call(prompt)
    obj = extract_json(text)
    return obj if obj is not None else {"raw": text[:1500], "error": "parse_failed"}


def _ctx() -> str:
    return ("=== ANTI-PATTERNS (obey) ===\n" + _read("patterns/META-LESSONS.md")[:2000] +
            "\n\n=== DATA WE OWN ===\n" + _read("DATA_CATALOG.md")[:1500] + _focus())


# The full proposal JSON contract — IDENTICAL for every arm (explore/refine/orthogonal/crossover);
# the director's gates (dedup, closed-family, theme cap, deployability) apply to all arms identically.
_CONTRACT_FIELDS = ("title, premium, market, data_source, free_or_owned, signal_approach, "
                    "why_not_duplicate, prior, pairs_with, gate0_data_check, crowding_risk, "
                    "retail_tradable_5k, scope, generalization_plan")

_DEPLOYABILITY = ("DEPLOYABILITY: the construction must be retail-tradable at ~$5K (IB/Alpaca; no illiquid "
                  "short legs — index-hedge instead; no >2x gross leverage; no intraday execution).")


def _parent_block(elite: dict, idx: int | None = None) -> str:
    """Per-parent summary for refine/orthogonal/crossover prompts (QuantaAlpha parent_template,
    enriched with our gate-stack verdict summary — they only had an IC number)."""
    s = elite.get("summary") or {}
    head = f"### PARENT {idx}" if idx else "ELITE STRATEGY"
    return (f"{head} (fitness/DSR {elite.get('fitness')})\n"
            f"Verdict summary: holdout_sharpe={s.get('holdout_sharpe')} search_sharpe={s.get('search_sharpe')} "
            f"n_trades={s.get('n_trades')} scope={s.get('scope')} tier={s.get('tier')}\n"
            f"Proposal:\n{json.dumps(elite.get('proposal'), indent=2)}")


def mutate(elite: dict) -> dict:
    """REFINE an elite: ONE targeted change to make it more robust / generalise better — keep what
    worked, fix what's fragile. NOT a fresh idea. Same JSON proposal format. (Evolutionary exploit step.)"""
    ctx = _ctx()
    prompt = f"""{ctx}

ELITE STRATEGY to EVOLVE (fitness/DSR {elite.get('fitness')}):
{json.dumps(elite.get('proposal'), indent=2)}

Propose ONE targeted MUTATION to make it MORE ROBUST or GENERALISE better — e.g. a different/broader universe,
a complementary leg, a cleaner/lower-turnover construction, a regime filter, a cost-hardening. Keep what worked;
fix the fragile part. {_DEPLOYABILITY} Mutating TOWARD deployability (e.g. long-only/index-hedged
variant) is itself a high-value mutation. This is an EVOLUTION of THIS strategy, NOT a new idea, and must still
obey the anti-patterns and use owned/free data. Return ONLY the SAME JSON proposal object ({_CONTRACT_FIELDS})
with the mutation applied (title should note it's a variant)."""
    obj = extract_json(_llm_call(prompt))
    return obj if obj is not None else {"error": "mutate_parse_failed"}


def orthogonal(elite: dict) -> dict:
    """ORTHOGONAL mutation (QuantaAlpha port, spec tasks/prompt-ports-quantaalpha.md §1): a new hypothesis
    near-INDEPENDENT of the parent — different mechanism — while exploiting the parent's hard-won knowledge
    of what data/universes/constructions actually survive our gates."""
    prompt = f"""{_ctx()}

{_parent_block(elite)}

Propose ONE NEW hypothesis that is ORTHOGONAL to this parent. "Orthogonal" means ALL of:
1. A completely DIFFERENT market hypothesis / economic mechanism (not a variant, not a refinement)
2. Different data dimensions or feature types driving the signal
3. Different investment logic or market perspective
4. Expected signal correlation to the parent ~0 (different return drivers)
You are judged on DIFFERENTIATION, not resemblance. DO exploit what the parent's success teaches about
which data, universes and construction styles survive our gates (e.g. its universe is tradable and clean) —
but the MECHANISM must be new. Must not duplicate existing experiments, must obey the anti-patterns, must
use owned/free data. {_DEPLOYABILITY}
Return ONLY a JSON proposal object ({_CONTRACT_FIELDS}) PLUS one extra field:
"orthogonality_reason": "why this is near-independent of the parent on the data / logic / horizon / market-state axes"."""
    obj = extract_json(_llm_call(prompt))
    return obj if obj is not None else {"error": "orthogonal_parse_failed"}


def crossover(elite_a: dict, elite_b: dict) -> dict:
    """CROSSOVER (QuantaAlpha port, spec §2): hybridize two elites from DIFFERENT families (enforced
    upstream by elite.sample_pair, not by prompt). The hybrid must have a reason to beat BOTH parents."""
    prompt = f"""{_ctx()}

{_parent_block(elite_a, 1)}

{_parent_block(elite_b, 2)}

Propose ONE HYBRID strategy that FUSES the validated mechanisms of these two parents (e.g. parent 1's
signal construction with parent 2's timing/universe edge). Requirements:
- Identify each parent's STRENGTHS and WEAKNESSES; the hybrid must AVOID weaknesses COMMON to both.
- The hybrid needs a specific reason to beat BOTH parents (synergy), not just average them.
- It must be ONE coherent tradable construction, not two books stapled together.
- Must not duplicate existing experiments, must obey the anti-patterns, owned/free data only. {_DEPLOYABILITY}
Return ONLY a JSON proposal object ({_CONTRACT_FIELDS}) PLUS two extra fields:
"fusion_logic": "how the parents' mechanisms combine",
"expected_benefit_over_parents": "why the hybrid should beat BOTH parents"."""
    obj = extract_json(_llm_call(prompt))
    return obj if obj is not None else {"error": "crossover_parse_failed"}


