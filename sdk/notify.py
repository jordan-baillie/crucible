"""Telegram alerts. One sender (_send) + message builders sharing one verdict renderer
(previously creds+urlencode+urlopen+error-handling written three times, and the alert
formatted the verdict independently of the wiki page — drift risk).

O2: alerts now carry the STAGE-2 evidence (MCPT p-value, generalization breadth) — the
numbers that distinguish a PASS from a candidate — not just stage-1 stats."""
import json
import os  # noqa: F401 (kept: legacy importers patch notify.os in tests)
import urllib.parse
import urllib.request


def _creds():
    from crucible_paths import SECRETS
    try:
        s = json.load(open(SECRETS))
    except (OSError, ValueError):
        return None, None
    return s.get("telegram_bot_token"), s.get("telegram_chat_id")


def _send(text: str, label: str = "message") -> bool:
    tok, chat = _creds()
    if not tok or not chat:
        print(f"[notify] telegram creds missing; skipping {label}")
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                       "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage",
                               data=data, timeout=20)
        print(f"[notify] telegram {label} sent")
        return True
    except Exception as e:
        print(f"[notify] telegram send failed: {e}")
        return False


def _fmt(v, pct: bool = False) -> str:
    """None-safe number formatting (the old f-string raised TypeError on a None maxdd)."""
    if v is None:
        return "?"
    return f"{v:.1%}" if pct else str(v)


def render_verdict_lines(verdict: dict) -> list:
    """The shared stage-1 + stage-2 evidence block (single source — used by alerts;
    keep in sync with the wiki page via this one function, not parallel f-strings)."""
    lines = [
        f"tier: {verdict.get('tier')} (FDR bar {verdict.get('promote_bar')}, "
        f"n_families {verdict.get('n_families')})",
        f"DSR {_fmt(verdict.get('dsr'))} | CPCV {_fmt(verdict.get('median_cpcv'))} | "
        f"PBO {_fmt(verdict.get('pbo'))}",
        f"holdout Sharpe {_fmt(verdict.get('holdout_sharpe'))} | "
        f"holdout {'PASS' if verdict.get('holdout_pass') else 'FAIL'}",
    ]
    m = verdict.get("mcpt") or {}
    if m:
        p = m.get("p_value", m.get("p_value_lb"))
        lines.append(f"MCPT p={_fmt(p)} ({m.get('n_ran', '?')} perms"
                     f"{', benchmark-relative' if m.get('benchmark_relative') else ''}) "
                     f"-> {'PASS' if verdict.get('mcpt_pass') else 'FAIL'}")
    g = verdict.get("generalization")
    if g:
        pos = sum(1 for x in g.values() if x is not None and x > 0)
        ran = sum(1 for x in g.values() if x is not None)
        lines.append(f"breadth: {pos}/{ran} untouched universes positive OOS "
                     f"({', '.join(f'{k} {v}' for k, v in g.items())})")
    return lines


def _soft_line(verdict: dict) -> str:
    """One PASS-alert line for pre-registered soft expectations — a falsified mechanism
    claim must reach the human WITH the green alert, not wait for a wiki read."""
    soft = verdict.get("soft_expectations")
    if not soft:
        return ""
    bad = [r["name"] for r in soft if r.get("pass") is not True]
    if not bad:
        return f"soft expectations: all {len(soft)} ✓\n"
    return f"⚠️ SOFT EXPECTATIONS FALSIFIED/ERROR: {', '.join(bad)} (story wrong, gates hold)\n"


def telegram_pass(spec, verdict: dict) -> bool:
    """Fires ONLY on a full-gate PASS (rare by design)."""
    body = "\n".join(render_verdict_lines(verdict))
    msg = (f"🟢 STRATEGY PASSED ALL GATES\n\n"
           f"<b>{spec.title}</b>\n"
           f"id: {spec.id} | markets: {', '.join(spec.markets)}\n\n"
           f"{body}\n"
           f"deployment ✓ (peak {_fmt(verdict.get('deploy_peak'))}, "
           f"{_fmt(verdict.get('deploy_sectors'))} sectors)\n"
           f"full Sharpe {_fmt(verdict.get('full_sharpe'))} | "
           f"maxDD {_fmt(verdict.get('full_maxdd'), pct=True)} | "
           f"{_fmt(verdict.get('n_trades'))} trades\n"
           f"{_soft_line(verdict)}\n"
           f"⚠️ Human review required before ANY capital. See wiki/experiments/{spec.id}.md")
    return _send(msg, label="🟢 PASS alert")


def telegram_candidate(spec, verdict: dict) -> bool:
    """Fires on a STAGE-1 pass — a CANDIDATE, not a confirmed edge."""
    needs = verdict.get("needs_confirmation", "fluke-confirmation")
    body = "\n".join(render_verdict_lines(verdict))
    msg = (f"🟡 STAGE-1 CANDIDATE (NOT confirmed)\n\n"
           f"<b>{spec.title}</b>\n"
           f"id: {spec.id} | scope: {verdict.get('scope', '?')} | "
           f"markets: {', '.join(spec.markets)}\n\n"
           f"{body}\n\n"
           f"⏳ REQUIRES <b>{needs}</b> before it's a real edge — a single-universe pass "
           f"can be a non-generalising overfit outlier (cf. BAB). NO capital until "
           f"confirmed + human review. See wiki/experiments/{spec.id}.md")
    # Severity routing (2026-06-12): a candidate needs nothing from the human TODAY
    # (forward-validation runs on its own) -> morning report, not a phone buzz.
    notice(msg, source="candidate")
    return True


def telegram_msg(text: str) -> bool:
    """Generic message (digest/heartbeat)."""
    return _send(text, label="message")


# ── Severity routing (2026-06-12, operator directive: "only alert critical issues") ─────
# CRITICAL -> immediate Telegram. Everything else -> notices.jsonl, folded into the ONE
# daily morning report. Critical =
#   - full-gate PASS (requires human review; rare by design)
#   - gate-canary BREACH (gate stack rotted; promotions unsafe)
#   - loop unit death (loop-alert@)
#   - money-path sentinel failures (holdout-ledger integrity, equity band, forward-paper
#     dead/failed-steps)
#   - live execution blocked/error/diverging (atlas daily)
# Everything else (digests, candidates, yellow canary warnings, data-freshness drift,
# forward-track nudges) is morning-report material, not a phone buzz.

def telegram_critical(text: str) -> bool:
    """Immediate Telegram — reserve for events needing same-day human attention."""
    return _send(text, label="CRITICAL alert")


def notice(text: str, source: str = "?") -> None:
    """Non-critical notice -> logs/notices.jsonl; the morning report drains the file
    into its 📥 section. Never sends Telegram."""
    import datetime
    from crucible_paths import ROOT
    p = ROOT / "logs" / "notices.jsonl"
    p.parent.mkdir(exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            "source": source, "text": text}) + "\n")
    print(f"[notify] notice queued for morning report ({source})")


def drain_notices() -> list:
    """Read + clear the notices file (morning report only). Drained rows are appended
    to notices-archive.jsonl so nothing is ever lost to truncation."""
    from crucible_paths import ROOT
    p = ROOT / "logs" / "notices.jsonl"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    rows = []
    for l in txt.splitlines():
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    with open(p.parent / "notices-archive.jsonl", "a", encoding="utf-8") as f:
        f.write(txt if txt.endswith("\n") or not txt else txt + "\n")
    p.unlink()
    return rows
