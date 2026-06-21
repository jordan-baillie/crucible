#!/usr/bin/env python3
"""G6/G7 consolidation shadow-soak (prereg-g6g7-consolidation 2026-06-13).

Nightly: compute G6/G7 via BOTH implementations (crucible forward/evidence.py post-fold
and atlas atlas/execution/gates.py) over the same data/live/<book> files. Telegram on any
pass/fail divergence; silent log row otherwise. Cutover (deleting atlas's recomputation)
only after >=7 clean days. Self-removing: this script and its timer die at cutover.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ATLAS = Path("/root/atlas")
sys.path.insert(0, str(ATLAS))

from forward import evidence  # noqa: E402
from atlas.execution import gates as agates  # noqa: E402

LIVE = Path("/root/atlas/data/live")
SOAK_LOG = Path(__file__).parent / "soak_g6g7.jsonl"


def _books() -> list[str]:
    reg = json.loads((ATLAS / "config" / "live_strategies.json").read_text(encoding="utf-8"))
    return [s["name"] for s in reg]


def compare(book: str) -> dict:
    d = LIVE / book
    fills = evidence._jsonl(d / "fills.jsonl")
    runs = evidence._jsonl(d / "runs.jsonl")

    # returns=[] -> pure 60d window (no epoch floor): this is a PARITY check against atlas's
    # slippage_gate/broker_error_gate, which are not epoch-aware, so both sides must window
    # identically. Production evaluate() passes real returns and DOES epoch-window.
    c6 = evidence._g6(book, [], fills)
    c7 = evidence._g7(runs, [])
    a6 = agates.slippage_gate(fills)
    a7 = agates.broker_error_gate(runs)

    row = {"ts": datetime.now().isoformat(timespec="seconds"), "book": book,
           "crucible": {"g6_pass": c6.get("pass"), "g6_median": c6.get("value"),
                        "g7_pass": c7.get("pass"), "g7_rate": c7.get("value")},
           "atlas": {"g6_pass": a6.get("pass"), "g6_median": a6.get("median_bps"),
                     "g7_pass": a7.get("pass"),
                     "g7_rate": (a7.get("error_rate_pct") / 100.0
                                 if a7.get("error_rate_pct") is not None else None)}}
    row["diverged"] = (row["crucible"]["g6_pass"] != row["atlas"]["g6_pass"]
                       or row["crucible"]["g7_pass"] != row["atlas"]["g7_pass"])
    return row


def main() -> int:
    diverged = []
    for book in _books():
        try:
            row = compare(book)
        except Exception as e:
            row = {"ts": datetime.now().isoformat(timespec="seconds"), "book": book,
                   "error": str(e)[:200], "diverged": True}
        with open(SOAK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if row.get("diverged"):
            diverged.append(row)
    if diverged:
        from sdk.notify import telegram_msg
        telegram_msg("⚠️ G6/G7 SOAK DIVERGENCE\n"
                     + "\n".join(json.dumps(r, default=str)[:300] for r in diverged)
                     + "\nConsolidation cutover BLOCKED until explained.")
        return 1
    print(f"[soak] {len(_books())} book(s) compared — no divergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
