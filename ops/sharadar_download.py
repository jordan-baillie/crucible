#!/usr/bin/env python3
"""Bulk-download Sharadar tables (SEP/TICKERS/ACTIONS) via Nasdaq Data Link export API.

Download-and-own: grab the full survivorship-free dataset locally so the subscription can be
cancelled. Web API only (no package). Key from ~/.atlas-secrets.json NASDAQ_DATA_LINK_API_KEY.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

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
            urllib.request.urlretrieve(link, dest)
            mb = dest.stat().st_size / 1e6
            print(f"  {table}: {mb:.1f} MB -> {dest}", flush=True)
            return str(dest)
        print(f"  {table}: status={status}, waiting 10s...", flush=True)
        time.sleep(10)
    raise RuntimeError(f"{table} export timed out")


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
