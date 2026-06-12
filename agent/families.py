"""Coarse family buckets — robust to LLM rewording — so the queue dedup + elite sampling group sibling
strategies (e.g. value×mom variants) under ONE family instead of treating each reworded proposal as novel.
Used by director._theme (queue theme-cap) and elite.sample (exploit diversity).

Bucket on the TITLE primarily (concise, leads with the real family) — the premium prose often lists OTHER
premia incidentally ("the missing AQR style; Value/Momentum/Carry already covered") which hijacks matching.
Order: unambiguous single families FIRST, then the value+momentum COMBINATION, then the generic singles."""
import re

_LOW_RISK = ("betting-against", "bab", "low-beta", "low beta", "low-vol", "low vol", "low volatility", "leverage-aversion")
_PAIRS = [
    ("illiquidity", ("amihud", "illiquidity", "illiquid", "liquidity premium", "liquidity risk")),
    ("carry", ("carry", "funding rate", "roll yield")),
    ("pead", ("pead", "post-earnings", "earnings-surprise", "earnings surprise", "sue", "drift")),
    ("skew", ("skew", "lottery", "idiosyncratic vol", "max return")),
    ("quality", ("quality", "profitability", "gross profit", "accrual")),
    ("reversal", ("reversal", "mean-reversion", "mean reversion", "short-term reversal")),
    ("trend", ("trend", "tsmom", "time-series momentum", "time series momentum", "managed futures")),
    ("momentum", ("momentum", "12-1")),
    ("value", ("value", "book-to", "earnings yield", "b/m")),
    ("size", ("size premium", "small-cap premium", "smb")),
    ("seasonal", ("seasonal", "calendar", "turn-of-month", "month-of-year")),
    ("credit", ("credit", "default spread", "duration")),
]


def family_bucket(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in _LOW_RISK):          # unambiguous single family first
        return "low_risk"
    has_val = any(k in t for k in ("value", "book-to", "book to", "earnings yield", "b/m", "btm"))
    has_mom = "momentum" in t or "12-1" in t or "12_1" in t
    if has_val and has_mom:                       # the value×mom COMBINATION (before 'trend' grabs 'trend-following')
        return "value_momentum"
    for name, kws in _PAIRS:
        if any(k in t for k in kws):
            return name
    return re.sub(r"[^a-z0-9]", "", t)[:24] or "misc"


# ---------------------------------------------------------------- dashboard lanes
# Swimlane classifier for the research map (#35 inversion, 2026-06-13): previously a
# hand-port inside atlas wiki_map.py that diverged RICHER than family_bucket (vrp/event/
# sports/multi lanes existed only there). Consolidated HERE — crucible owns family
# semantics — and shipped to the dashboard via the forge_state.json artifact.
_LANE_LOW_RISK = _LOW_RISK + ("defensive",)
_LANE_PAIRS = [
    ("illiquidity", ("amihud", "illiquidity", "illiquid", "liquidity premium",
                     "liquidity risk", "liquidity_premium", "liquidity_provision", "liquidity")),
    ("carry", ("carry", "funding rate", "funding-carry", "roll yield", "basis")),
    ("event", ("pead", "post-earnings", "post_earnings", "earnings-surprise", "earnings surprise",
               "sue", "drift", "event", "announcement", "fomc", "auction", "dividend",
               "index_flow", "recon", "deletion", "spin-off", "spinoff")),
    ("skew", ("skew", "lottery", "idiosyncratic vol", "max return", "higher_moment")),
    ("quality", ("quality", "profitability", "gross profit", "accrual")),
    ("reversal", ("reversal", "mean-reversion", "mean reversion", "short-term reversal")),
    ("vrp", ("vrp", "volatility_risk", "volatility risk", "short-vol", "variance", "vvix")),
    ("trend", ("trend", "tsmom", "time-series momentum", "time series momentum", "managed futures")),
    ("value_momentum", ("value_momentum", "value+momentum", "valmom", "val_mom", "value mom")),
    ("momentum", ("momentum", "12-1")),
    ("value", ("value", "book-to", "earnings yield", "b/m")),
    ("size", ("size premium", "small-cap premium", "smb")),
    ("seasonal", ("seasonal", "calendar", "turn-of-month", "month-of-year")),
    ("credit", ("credit", "default spread", "duration")),
    ("positioning", ("hedging pressure", "hedging-pressure", "cot", "positioning", "short interest")),
    ("share_issuance", ("issuance", "repurchas", "buyback")),
    ("sports", ("pitcher", "sports", "mlb", "nrl")),
]

LANE_LABELS = {
    "illiquidity": "Illiquidity", "carry": "Carry", "event": "Event / Announcement",
    "skew": "Skew / Lottery", "quality": "Quality / Accruals", "reversal": "Reversal",
    "vrp": "Volatility Risk Premium", "trend": "Trend / TSMOM",
    "value_momentum": "Value × Momentum", "momentum": "Momentum", "value": "Value",
    "size": "Size", "seasonal": "Seasonality", "credit": "Credit",
    "positioning": "Positioning / Flow", "share_issuance": "Share Issuance",
    "sports": "Sports", "low_risk": "Low-Risk / BAB", "multi": "Multi-Premium Books",
    "other": "Other",
}


def lane_of(family: str, title: str = "") -> str:
    t = f"{family} {title}".lower()
    if any(k in t for k in _LANE_LOW_RISK):
        return "low_risk"
    has_val = any(k in t for k in ("value", "book-to", "book to", "earnings yield", "b/m", "btm"))
    has_mom = "momentum" in t or "12-1" in t or "12_1" in t
    if has_val and has_mom:
        return "value_momentum"
    if "combo" in t or "two-premium" in t or "two premium" in t or ("book" in t and "trend" in t):
        # multi-premium blended books (carry×trend etc.) get their own lane — they relate
        # to several premia and would otherwise pollute single lanes
        if sum(1 for _, kws in _LANE_PAIRS if any(k in t for k in kws)) >= 2:
            return "multi"
    for name, kws in _LANE_PAIRS:
        if any(k in t for k in kws):
            return name
    return "other"
