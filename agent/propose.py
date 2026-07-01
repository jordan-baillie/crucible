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


_CRYPTO_FOCUS = (
        "\n\n=== SEARCH FOCUS: CRYPTO (operator-directed 2026-06-15 — overrides the retail-equity bias) ===\n"
        "Propose CRYPTO-deployable strategies ONLY this run:\n"
        "- Venue: spot + PERPETUAL futures on Binance/Bybit. UNIVERSE: use binance_universe(75) for the BROAD\n"
        "  liquid cross-section (top ~75 USDT perps by volume, liquidity-screened) on any CROSS-SECTIONAL factor;\n"
        "  the 12 CRYPTO_MAJORS are only for SINGLE-ASSET timing (BTC/ETH vol, basis). A real cross-section is\n"
        "  REQUIRED — the breadth/Fundamental-Law gate FLAGS a high IR on a tiny correlated cross-section as overfit.\n"
        "- Crypto deployability: perps short FREELY (NO stock-borrow constraint — a crypto long/short IS deployable),\n"
        "  ~20bps round-trip taker cost, <=2x leverage, no options. Set market='crypto', retail_tradable_5k='yes'.\n"
        "- Data reachable (adapters only — NEVER raw I/O): funding_rates() (perp funding, majors, 2019+);\n"
        "  binance_klines(binance_universe(75), market='perp'|'spot') = daily OHLCV + volume + trades + TAKER_BUY_QUOTE\n"
        "  (deep-history flow/positioning proxy) over the BROAD universe -> basis (perp vs spot), momentum/reversal,\n"
        "  CRYPTO VOL/VRP: deribit_dvol('BTC'|'ETH') = DVOL implied-vol index (daily ~2021+) -> VRP = DVOL minus realized.\n"
        "  realized vol, liquidity tiers, taker-flow crowding; yf_panel for extra spot. ⚠ Binance OI + long/short\n"
        "  ratio are LAST-30-DAYS only (NOT backtestable -> DATA-GATED); use taker_buy_quote for deep-history flow.\n"
        "  ON-CHAIN FUNDAMENTALS: coinmetrics_metrics(CM_COMMUNITY_MAJORS, metrics=(...)) = free daily 2010+ —\n"
        "  CapMVRVCur (MVRV over/under-valuation), AdrActCnt (active addresses/adoption), TxCnt/TxTfrCnt (usage),\n"
        "  HashRate, SplyCur, IssTotUSD (supply/issuance), CapMrktCurUSD. (CC BY-NC personal-use; realized-cap/NVT PAID.)\n"
        "  CROSS-EXCHANGE: bybit_funding((...)) pairs with funding_rates (Binance) -> funding DISPERSION\n"
        "  (bybit-binance = localized crowding/venue dislocation). STABLECOIN FLOWS: coinmetrics_metrics\n"
        "  ((\"usdt\",\"usdc\"),(\"SplyCur\",)) -> supply growth=inflow / contraction=outflow (crypto macro-liquidity).\n"
        "- DORMANT — do NOT just re-propose delta-neutral FUNDING CARRY: funding has compressed to ~0/negative in 2025-26\n"
        "  (see markets/crypto.md); it earns nothing today. Naive crypto MOMENTUM has FAILED before.\n"
        "- LOOK BEYOND CARRY for something that could pay in the CURRENT regime: cross-sectional crypto factors\n"
        "  (momentum/reversal/low-vol/illiquidity ACROSS the broad binance_universe(75) — a real cross-section beats\n"
        "  the 12-major one), basis- or taker-flow/crowding-CONDITIONAL timing, vol/term-structure (deribit_dvol),\n"
        "  regime gates. PREFER conditional/combination constructions over naive single signals.\n"
)


_COMMODITIES_FOCUS = (
    "\n\n=== SEARCH FOCUS: COMMODITY FUTURES (operator-directed; updated 2026-06-16 with the sparse-signal lesson) ===\n"
    "Propose COMMODITY-FUTURES strategies ONLY this run over the owned 17-root Databento complex. The conditioning\n"
    "TRIO is wired (price + positioning + fundamentals) BUT the first exercise (13 runs, 0 passes) proved that\n"
    "storage-theory built DIRECTLY on the fundamentals reports trades far too sparsely to clear the screen (0-7\n"
    "trades — EIA is weekly, USDA quarterly, and extreme-state conditioning cuts it further). THE LESSON, OBEY IT:\n"
    "the PRIMARY signal MUST be a DAILY, higher-frequency BASE; slow fundamentals are a CONDITIONING OVERLAY,\n"
    "NEVER the trade trigger.\n"
    "- UNIVERSE: the 17 roots — ENERGY {CL crude, NG natgas, HO heating-oil, RB gasoline}, METALS {GC gold,\n"
    "  SI silver, HG copper, PL platinum, PA palladium}, GRAINS/OILSEEDS {ZC corn, ZS soybeans, ZW wheat,\n"
    "  ZL soyoil, ZM soymeal}, LIVESTOCK {LE live-cattle, HE lean-hogs, GF feeder}. Use a real CROSS-SECTION\n"
    "  (rank across roots) or a within-complex relative-value pair. ⚠ PA rank-2 coverage thin (~91%) — drop it\n"
    "  from cross-sections or require both legs present.\n"
    "- PRIMARY BASE — pick a DAILY signal that trades often enough for the screen (THE open frontier):\n"
    "  • CURVE CARRY / ROLL-YIELD: fut_curve(root, n_contracts=2) slope close_2/close_1 — rank the cross-section\n"
    "    daily, long backwardation / short contango. • BASIS-MOMENTUM (Boons-Prado JF 2019): momentum of the\n"
    "    front-minus-back return spread (the headline untested premium). ⚠ fut_curve is NOT roll-adjusted:\n"
    "    compute returns WITHIN a contract month; NEVER diff close_1 across a roll. These rebalance daily/weekly\n"
    "    -> hundreds of trades, enough for statistical power.\n"
    "- SECONDARY BASE (weekly, still tradeable): HEDGING-PRESSURE (Basu-Miffre) from cot_positioning(roots,\n"
    "  start_year=2010) -> {root}_comm_net/_noncomm_net/_oi; comm_net/oi = commercials (hedgers) net, predicts\n"
    "  the premium speculators earn. ⚠ join on the Friday RELEASE date (Tue data + 3d), NEVER the Tuesday.\n"
    "- OVERLAY ONLY (slow — TILT/GATE the base, do NOT trigger on it): eia_series(series_id) ENERGY inventories\n"
    "  (e.g. 'PET.WCESTUS1.W'=US crude stocks weekly; natgas storage) + usda_nass(commodity,\n"
    "  statisticcat_desc='STOCKS') GRAIN stocks. Storage theory as a CONFIRM on the daily base (e.g. take\n"
    "  curve-carry only when inventory confirms the curve state; size up when storage is extreme) — a\n"
    "  low-frequency tilt, not the entry. ⚠ PIT: condition on the report RELEASE date (EIA ~Wed; USDA release\n"
    "  calendar — NOT the survey reference period like 'FIRST OF MAR'); usda 'Value' is a comma-string; NASS\n"
    "  query caps at 50k rows.\n"
    "- DEPLOYABILITY: commodity futures are NATIVELY retail-tradable at ~$5K via IB MICRO contracts (MCL crude,\n"
    "  MGC gold, micro grains; standard ZC/ZS/ZW in small size) — longable AND shortable with NO borrow\n"
    "  constraint (unlike equity shorts), scalable to size. Set market='futures', retail_tradable_5k='yes'.\n"
    "- KNOWN (wiki markets/futures.md — do NOT re-propose): cross-asset TREND is VALIDATED as a crisis hedge\n"
    "  (boreas-tsmom); DM FX/bond CARRY is DEAD (boreas-carry-fxbond); fundamentals-AS-TRIGGER storage-theory is\n"
    "  too sparse (proven 2026-06-16, 0/13). Build a DAILY base premium (curve carry / basis-momentum) with\n"
    "  fundamentals as the slow overlay — that combination is the genuinely open frontier.\n"
)


# Operator-directed search focuses (env CRUCIBLE_FOCUS). Aliases map to one block; unknown/empty -> no
# steer (general retail-deployable bias). Generator-only: injected into every arm's prompt, the rails /
# gate stack are untouched (bad ideas still burn regardless of where they came from). Add a focus here.
_FOCUSES = {
    "crypto": _CRYPTO_FOCUS,
    "commodities": _COMMODITIES_FOCUS,
    "commodity": _COMMODITIES_FOCUS,
    "futures": _COMMODITIES_FOCUS,
}


def _focus() -> str:
    """Operator-directed SEARCH FOCUS (env CRUCIBLE_FOCUS), injected into every arm. Reversible:
    unset/empty/unknown = the general retail-deployable bias; otherwise dispatch to a focus block.
    Supported: 'crypto' (crypto-deployable), 'commodities'/'commodity'/'futures' (commodity-futures trio)."""
    return _FOCUSES.get(os.environ.get("CRUCIBLE_FOCUS", "").strip().lower(), "")


def _diversity_brief() -> str:
    """Generator-only steer toward UNSPENT family-space (agent.joint_state.brief) — the already
    closed / live / heavily-exploited families to avoid. '' when there is nothing to steer against.
    Same category as _focus(): injected into the prompt, the rails / gate stack are untouched."""
    try:
        from agent.joint_state import brief
        return brief()
    except Exception:
        return ""


def _read_tail(p, max_chars: int):
    """E5: growing files (index, candidates) are read NEWEST-LAST and capped — prompt size must
    not rise monotonically with project history (cost, latency, call-timeout risk; the director
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
        + _focus() + _diversity_brief()
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
"seeded_by": "the EXACT title of the WEB-SCOUTED CANDIDATE above that this hypothesis is built on, or 'self' if it is novel / a recombination of wiki knowledge NOT taken from the scouted list. Be honest — this is measured, not graded.",
"crowding_risk": "low|medium|high — how heavily-published/arbitraged is this edge? (favour low)",
"retail_tradable_5k": "yes|no — can THIS construction be executed at ~$5K via IB/Alpaca (instruments routable, short leg borrowable or index-hedged, no >2x gross leverage)? If no, this proposal will be DOWN-RANKED — prefer redesigning it to a deployable variant",
"scope": "broad|local — broad if a UNIVERSAL mechanism (theory says it appears across markets; a pass must later GENERALISE) or local if defensibly universe-specific (then forward-validation confirms it)",
"generalization_plan": "if broad: the untouched universes to confirm the mechanism in (e.g. other cap-tiers/sectors/asset-classes); if local: the economic reason it lives ONLY in this universe + the forward-validation plan"}}"""
    # Inherits the 900s default from agent.llm.call. Fable-5 measured ~285s on a real propose
    # (2026-06-10 smoke test); the 900s ceiling leaves ample headroom.
    text = _llm_call(prompt)
    obj = extract_json(text)
    return obj if obj is not None else {"raw": text[:1500], "error": "parse_failed"}


def _ctx() -> str:
    return ("=== ANTI-PATTERNS (obey) ===\n" + _read("patterns/META-LESSONS.md")[:2000] +
            "\n\n=== DATA WE OWN ===\n" + _read("DATA_CATALOG.md")[:1500] + _focus() + _diversity_brief())


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
    blocker = s.get("blocker")
    return (f"{head} (fitness/DSR {elite.get('fitness')})\n"
            f"Verdict summary: holdout_sharpe={s.get('holdout_sharpe')} search_sharpe={s.get('search_sharpe')} "
            f"pbo={s.get('pbo')} n_trades={s.get('n_trades')} scope={s.get('scope')} tier={s.get('tier')}\n"
            + (f"BLOCKER TO FIX (what stopped this parent passing): {blocker}\n" if blocker else "")
            + f"Proposal:\n{json.dumps(elite.get('proposal'), indent=2)}")


def mutate(elite: dict) -> dict:
    """REFINE an elite: ONE targeted change to make it more robust / generalise better — keep what
    worked, fix what's fragile. NOT a fresh idea. Same JSON proposal format. (Evolutionary exploit step.)"""
    ctx = _ctx()
    s = elite.get("summary") or {}
    blocker = s.get("blocker")
    # Lead with the ACTUAL wall so the mutation attacks it instead of wandering (audit 2026-06-25:
    # 48% of near-misses die on PBO and the loop never knew). When a blocker is known the mutation
    # MUST target it; otherwise fall back to general robustness.
    directive = (f"This parent did NOT pass. THE WALL TO BREAK (your mutation MUST target this, not a "
                 f"cosmetic tweak): {blocker}\n\n" if blocker else "")
    prompt = f"""{ctx}

ELITE STRATEGY to EVOLVE (fitness/DSR {elite.get('fitness')}, holdout_sharpe={s.get('holdout_sharpe')}, pbo={s.get('pbo')}):
{json.dumps(elite.get('proposal'), indent=2)}

{directive}Propose ONE targeted MUTATION to make it MORE ROBUST or GENERALISE better — if a WALL is named above,
the mutation must DIRECTLY reduce it; otherwise: a different/broader universe, a complementary leg, a cleaner/
lower-turnover construction, a regime filter, a cost-hardening. Keep what worked; fix the fragile part.
{_DEPLOYABILITY} Mutating TOWARD deployability (e.g. long-only/index-hedged variant) is itself a high-value
mutation. This is an EVOLUTION of THIS strategy, NOT a new idea, and must still obey the anti-patterns and use
owned/free data. Return ONLY the SAME JSON proposal object ({_CONTRACT_FIELDS}) with the mutation applied
(title should note it's a variant)."""
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


