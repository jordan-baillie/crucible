"""agent/fintwit.py — the credibility-vetted FinTwit idea-grounding layer for the scout.

Reads the one-direction artifact `fintwit_digest.json` PRODUCED by the credibility-engine
(scripts/export_fintwit_digest.py) — top accounts that have been RESOLVED against price data and
clear the credibility floor, with their recent REASONING posts. The scout feeds these to the SAME
Claude-Max distillation, which extracts STRUCTURAL PREMIA (never the directional call).

Design: design-fintwit-scout-grounding.md. Mirrors agent/firecrawl.py — additive + GRACEFUL: any
failure (artifact absent / stale / unparseable) returns empty so the scout runs exactly as today.
No cross-repo imports (file-artifact contract, #34). X is an IDEA source here, never a price feed —
the gate stack on owned data is the sole validator, so there is no PIT/survivorship problem.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Same-box default; override with CRUCIBLE_FINTWIT_DIGEST. (credibility-engine writes here.)
_DEFAULT = "/root/credibility-engine/outputs/fintwit_digest.json"
MAX_AGE_DAYS = 21          # staleness guard: an old digest is ignored (the engine may be paused)
POSTS_PER_ACCT = 5         # cap per account in the distillation context
MAX_ACCOUNTS = 15


def _path() -> str:
    return os.environ.get("CRUCIBLE_FINTWIT_DIGEST", _DEFAULT)


def read_digest(max_age_days: int = MAX_AGE_DAYS):
    """(accounts, err). GRACEFUL: returns ([], reason) on any problem — never raises."""
    p = Path(_path())
    if not p.exists():
        return [], "no fintwit digest"
    try:
        d = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return [], f"unparseable digest: {str(e)[:120]}"
    if d.get("schema_version") != 1:
        return [], f"unsupported schema_version {d.get('schema_version')}"
    gen = d.get("generated_at")
    if gen:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(gen)).days
            if age > max_age_days:
                return [], f"stale digest ({age}d > {max_age_days}d)"
        except Exception:  # noqa: BLE001
            pass  # unparseable timestamp -> don't block on it
    return (d.get("accounts") or [])[:MAX_ACCOUNTS], ""


def format_fintwit(accounts) -> str:
    """Format vetted accounts + their reasoning posts for the distillation context."""
    if not accounts:
        return "(no vetted FinTwit accounts)"
    out = []
    for a in accounts:
        c = a.get("credibility") or {}
        head = (f"@{a.get('handle')} [credibility: recency-weighted hit-rate "
                f"{c.get('recency_weighted_hit_rate')}, n={c.get('resolvable_claims')} resolved calls; "
                f"assets: {a.get('primary_assets') or '?'}]")
        posts = a.get("recent_posts") or []
        plines = "\n".join(f"  • {(p.get('text') or '').strip()[:280].replace(chr(10), ' ')}"
                           for p in posts[:POSTS_PER_ACCT])
        out.append(f"{head}\n{plines}")
    return "\n\n".join(out)


def fintwit_text() -> str:
    """Convenience for the scout: formatted vetted-FinTwit block, or a one-line reason if unavailable."""
    accounts, err = read_digest()
    if not accounts:
        return f"(vetted FinTwit unavailable: {err})"
    return format_fintwit(accounts)
