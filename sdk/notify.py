"""Telegram alert — fires ONLY when a strategy passes ALL gates (rare by design)."""
import json, os, urllib.parse, urllib.request


def _creds():
    s = json.load(open(os.path.expanduser("~/.atlas-secrets.json")))
    return s.get("telegram_bot_token"), s.get("telegram_chat_id")


def telegram_pass(spec, verdict: dict):
    tok, chat = _creds()
    if not tok or not chat:
        print("[notify] telegram creds missing; skipping alert"); return False
    msg = (f"🟢 STRATEGY PASSED ALL GATES\n\n"
           f"<b>{spec.title}</b>\n"
           f"id: {spec.id} | markets: {', '.join(spec.markets)}\n\n"
           f"tier: {verdict['tier']} (FDR bar {verdict['promote_bar']}, n_families {verdict['n_families']})\n"
           f"DSR {verdict['dsr']} | CPCV {verdict['median_cpcv']} | PBO {verdict['pbo']}\n"
           f"holdout Sharpe {verdict['holdout_sharpe']} | holdout_gate PASS\n"
           f"deployment ✓ (peak {verdict['deploy_peak']}, {verdict['deploy_sectors']} sectors)\n"
           f"full Sharpe {verdict['full_sharpe']} | maxDD {verdict['full_maxdd']:.1%} | {verdict['n_trades']} trades\n\n"
           f"⚠️ Human review required before ANY capital. See wiki/experiments/{spec.id}.md")
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg, "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)
        print("[notify] 🟢 Telegram alert sent (passed all gates)"); return True
    except Exception as e:
        print(f"[notify] telegram send failed: {e}"); return False


def telegram_candidate(spec, verdict: dict):
    """Fires when a strategy clears the STAGE-1 single-universe gates — a CANDIDATE, not a confirmed
    edge. Confirmation (generalization or forward-validation) is required before any capital."""
    tok, chat = _creds()
    if not tok or not chat:
        print("[notify] telegram creds missing; skipping candidate alert"); return False
    needs = verdict.get("needs_confirmation", "fluke-confirmation")
    msg = (f"🟡 STAGE-1 CANDIDATE (NOT confirmed)\n\n"
           f"<b>{spec.title}</b>\n"
           f"id: {spec.id} | scope: {verdict.get('scope','?')} | markets: {', '.join(spec.markets)}\n\n"
           f"Cleared all single-universe gates: tier {verdict['tier']} (bar {verdict['promote_bar']}), "
           f"DSR {verdict['dsr']} | CPCV {verdict['median_cpcv']} | PBO {verdict['pbo']} | holdout {verdict['holdout_sharpe']}\n\n"
           f"⏳ REQUIRES <b>{needs}</b> before it's a real edge — a single-universe pass can be a "
           f"non-generalising overfit outlier (cf. BAB). NO capital until confirmed + human review. "
           f"See wiki/experiments/{spec.id}.md")
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg, "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)
        print("[notify] 🟡 Telegram CANDIDATE alert sent (stage-1 pass; needs confirmation)"); return True
    except Exception as e:
        print(f"[notify] telegram send failed: {e}"); return False


def telegram_msg(text: str):
    """Generic message (digest/heartbeat)."""
    tok, chat = _creds()
    if not tok or not chat:
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)
        return True
    except Exception:
        return False
