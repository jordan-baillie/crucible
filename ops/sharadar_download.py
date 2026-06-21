#!/usr/bin/env python3
"""Bulk-download Sharadar tables (SEP/TICKERS/ACTIONS) via Nasdaq Data Link export API.

Download-and-own: grab the full survivorship-free dataset locally so the subscription can be
cancelled. Web API only (no package). Key from ~/.atlas-secrets.json NASDAQ_DATA_LINK_API_KEY.
"""
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

# Download hardening (2026-06-21): the bulk file download must never hang forever.
# A stalled connection on the previous unbounded urlretrieve() left the weekly refresh
# wedged for >1.5d (SEP.zip went 8d stale; sentinel S1 fired). Two bounds make that class
# unrepresentable: a per-read socket timeout AND an overall no-progress (stall) deadline.
DOWNLOAD_READ_TIMEOUT_S = 120      # per-read socket timeout (a frozen connection aborts here)
DOWNLOAD_DEADLINE_S = 2400         # overall wall-clock cap (SEP.zip ~1GB; trickle-forever guard)
DOWNLOAD_CHUNK = 1 << 20           # 1 MiB

# Re-homed from atlas/scripts 2026-06-13 (#36): the DOWNLOADER is research-domain
# (crucible adapters are the primary consumer). Data stays at the shared path both
# repos read (atlas dashboard reads TICKERS; crucible sdk reads SEP/SF1).
OUT = Path("/root/atlas/data/sharadar")


def _key():
    k = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if k:
        return k
    p = os.path.expanduser("~/.atlas-secrets.json")
    return json.load(open(p)).get("NASDAQ_DATA_LINK_API_KEY")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def export_table(table: str, key: str, max_wait_s: int = 1800) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    op = urllib.request.build_opener(_NoRedirect)
    url = f"https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json?qopts.export=true&api_key={key}"
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            r = op.open(url, timeout=90)
            d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            d = json.loads(e.read())
        f = d["datatable_bulk_download"]["file"]
        status = f.get("status", "fresh")
        link = f.get("link")
        if status == "fresh" and link:
            dest = OUT / f"{table}.zip"
            print(f"  {table}: status=fresh, downloading...", flush=True)
            _download(link, dest, table)
            mb = dest.stat().st_size / 1e6
            print(f"  {table}: {mb:.1f} MB -> {dest}", flush=True)
            return str(dest)
        print(f"  {table}: status={status}, waiting 10s...", flush=True)
        time.sleep(10)
    raise RuntimeError(f"{table} export timed out")


def _download(link: str, dest: Path, table: str) -> None:
    """Bounded, atomic file download. Streams to <dest>.part with a per-read socket timeout
    and an overall stall watchdog, then atomically renames. A partial/stalled download is
    aborted (and never left in place as a fake-complete file) — the fix for the 2026-06-21
    wedge where urlretrieve() hung with no timeout and a 64KB partial sat as the 'result'."""
    part = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(link, timeout=DOWNLOAD_READ_TIMEOUT_S) as r, open(part, "wb") as fh:
            while True:
                try:
                    chunk = r.read(DOWNLOAD_CHUNK)
                except socket.timeout as e:
                    raise RuntimeError(f"{table} download read timed out after {DOWNLOAD_READ_TIMEOUT_S}s") from e
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if time.time() - t0 > DOWNLOAD_DEADLINE_S:
                    raise RuntimeError(f"{table} download exceeded {DOWNLOAD_DEADLINE_S}s deadline "
                                       f"(got {got / 1e6:.1f} MB) — aborting trickle")
        if got == 0:
            raise RuntimeError(f"{table} download produced 0 bytes")
        part.replace(dest)
    finally:
        if part.exists():
            part.unlink()  # never leave a partial masquerading as complete


def main():
    key = _key()
    if not key:
        print("ERROR: no NASDAQ_DATA_LINK_API_KEY")
        return 1
    tables = sys.argv[1:] or ["TICKERS", "ACTIONS", "SEP"]
    for t in tables:
        export_table(t, key)
    print("[sharadar_download] DONE:", tables, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
