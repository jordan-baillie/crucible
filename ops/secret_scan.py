#!/usr/bin/env python3
"""Secret scan for the research wiki before its nightly off-box push
(LOOPS_FRAMEWORK_PLAN 1.4a: the off-box repo must never carry tokens).

Scans every file that would be pushed (tracked + untracked, .gitignore respected)
for credential patterns AND for the literal live secret values read from the
local secrets stores (never printed — only their presence is reported).

Exit 0 = clean (push may proceed). Exit 1 = SECRETS FOUND (push must be blocked).
Designed to FAIL CLOSED: a crash also blocks the push (exit !=0).

Usage: python3 ops/secret_scan.py [repo_dir]   (default /root/research-wiki)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Generic credential shapes (provider-documented formats)
PATTERNS = [
    ("telegram bot token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}\b")),
    ("alpaca key id", re.compile(r"\b(?:PK|AK)[A-Z0-9]{16,20}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai/anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("slack token", re.compile(r"\bxox[bpars]-[A-Za-z0-9-]{10,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("url-embedded basic auth", re.compile(r"https?://[^/\s:]+:[^@\s/]{8,}@")),
]

# Local stores whose VALUES must never appear off-box (values read, never printed)
SECRET_STORES = ["/root/.atlas-secrets.json", "/root/.crucible-secrets.json"]

SKIP_SUFFIXES = {".parquet", ".png", ".jpg", ".gz", ".zip", ".pyc", ".pdf"}
MAX_FILE_BYTES = 5_000_000


def _live_values() -> list[str]:
    vals: list[str] = []
    for store in SECRET_STORES:
        p = Path(store)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for v in data.values():
            if isinstance(v, str) and len(v) >= 8:  # short values would false-positive
                vals.append(v)
    return vals


def scan(repo: Path) -> int:
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                       cwd=repo, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"[secret-scan] git ls-files failed in {repo}: {r.stderr.strip()}")
        return 1  # fail closed
    files = [repo / f for f in r.stdout.splitlines() if f.strip()]
    live = _live_values()
    hits: list[str] = []

    for f in files:
        if f.suffix.lower() in SKIP_SUFFIXES or not f.is_file():
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        rel = f.relative_to(repo)
        for name, pat in PATTERNS:
            m = pat.search(text)
            if m:
                redacted = m.group(0)[:6] + "…" + m.group(0)[-3:]
                hits.append(f"{rel}: {name} ({redacted})")
        for v in live:
            if v in text:
                hits.append(f"{rel}: LIVE secret value from local store (redacted)")
                break

    if hits:
        print(f"[secret-scan] ❌ {len(hits)} hit(s) in {repo} — PUSH MUST BE BLOCKED")
        for h in hits:
            print(f"[secret-scan]   {h}")
        return 1
    print(f"[secret-scan] clean: {len(files)} files in {repo}")
    return 0


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/research-wiki")
    sys.exit(scan(repo))
