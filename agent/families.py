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
