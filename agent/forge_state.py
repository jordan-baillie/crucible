#!/usr/bin/env python3
"""forge_state.json producer (#35 inversion, 2026-06-13 review finding #4).

The atlas dashboard previously parsed crucible internals directly (run_log.jsonl,
wiki registry/candidates, LOOP_DISABLED) ~820 lines deep, including a hand-ported
lane classifier that had diverged. Inversion: crucible OWNS the semantics and emits
one versioned snapshot artifact; atlas renders it.

Artifact: <wiki>/.dashboard/forge_state.json (schema_version 1, atomic write).
Sections: summary, fdr, cycles[:50], candidates[:14], lanes (experiment/queue/premia
stem -> lane via agent.families.lane_of — the ONE classifier), loop_disabled, log_tail.
Host-level facts (systemd timer state) stay on the atlas side — they're about the
host, not the research.

Run: end of the nightly forge service + after triage. Cheap (~100 rows + wiki scan).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crucible_paths import WIKI  # noqa: E402
from agent.families import lane_of, LANE_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN_LOG = ROOT / "agent" / "run_log.jsonl"
LOOP_DISABLED = ROOT / "LOOP_DISABLED"
CYCLE_LOG = Path("/tmp/crucible_forge.log")
OUT = WIKI / ".dashboard" / "forge_state.json"
SCHEMA_VERSION = 1


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _clip(v, n=400):
    return (str(v)[:n] if v is not None else None)


def _parse_cycles(rows: list[dict]) -> list[dict]:
    cycles = []
    for o in rows:
        v = o.get("verdict") or {}
        p = o.get("proposal") or {}
        ran = bool(o.get("ran"))
        passed = bool(o.get("passed_all") or v.get("PASSED_ALL_GATES"))
        tier = v.get("tier") or ("PASS" if passed else None)
        if passed:
            status = "pass"
        elif ran and tier == "PROMOTE":
            status = "near_miss"   # cleared the FDR bar, failed a later gate
        elif ran:
            status = "fail"
        else:
            status = "error"
        ss, hs = _num(v.get("search_sharpe")), _num(v.get("holdout_sharpe"))
        degradation = round((hs / ss - 1) * 100, 1) if ss not in (None, 0) and hs is not None else None
        title = o.get("title") or p.get("title") or "(untitled)"
        cycles.append({
            "ts": o.get("ts"), "id": o.get("id"), "title": title,
            "status": status, "ran": ran, "tier": tier, "passed_all": passed,
            "family": v.get("family"), "lane": lane_of(v.get("family") or "", title),
            "agent": o.get("agent"), "arm": o.get("arm"),
            "premium": _clip(p.get("premium"), 240), "market": _clip(p.get("market"), 200),
            "hypothesis": {
                "signal_approach": _clip(p.get("signal_approach"), 600),
                "why_not_duplicate": _clip(p.get("why_not_duplicate"), 500),
                "pairs_with": _clip(p.get("pairs_with"), 300),
                "prior": _clip(p.get("prior"), 60),
            },
            "data": {
                "free_or_owned": _clip(p.get("free_or_owned"), 200),
                "data_source": _clip(p.get("data_source"), 300),
                "gate0_data_check": _clip(p.get("gate0_data_check"), 500),
            },
            "metrics": {
                "search_sharpe": ss, "holdout_sharpe": hs, "degradation_pct": degradation,
                "holdout_pass": v.get("holdout_pass"),
                "holdout_reasons": v.get("holdout_reasons") or [],
                "full_sharpe": _num(v.get("full_sharpe")), "full_maxdd": _num(v.get("full_maxdd")),
                "n_trades": _num(v.get("n_trades")), "dsr": _num(v.get("dsr")),
                "median_cpcv": _num(v.get("median_cpcv")), "pbo": _num(v.get("pbo")),
                "deployment_passed": v.get("deployment_passed"),
                "promote_bar": _num(v.get("promote_bar")), "n_families": _num(v.get("n_families")),
            },
        })
    cycles.reverse()
    return cycles


def _parse_registry() -> dict:
    rows: list[dict] = []
    for p in sorted(glob.glob(str(WIKI / ".registry" / "*.jsonl"))):
        rows.extend(_read_jsonl(Path(p)))
    families, history, seen = [], [], set()
    bar = 0.90
    for r in rows:
        pd = _num(r.get("promote_dsr"))
        if pd is not None:
            history.append(round(pd, 4))
            bar = max(bar, pd)
        fam = r.get("family")
        if fam and fam not in seen:
            seen.add(fam)
            families.append({"family": fam, "tier": r.get("tier"),
                             "dsr": _num(r.get("dsr")), "passed_all": bool(r.get("passed_all"))})
    return {"bar": round(bar, 4), "n_families": len(seen), "families": families, "history": history}


_CAND_RE = re.compile(r"^- \*\*(.+?)\*\*\s*\((.*?)\)\s*[—-]\s*(.*)$")


def _parse_candidates() -> list[dict]:
    out, seen = [], set()
    try:
        lines = (WIKI / "candidates.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in reversed(lines):
        m = _CAND_RE.match(ln.strip())
        if not m:
            continue
        title, tags, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        key = re.sub(r"[^a-z]", "", title.lower())[:28]
        if key in seen:
            continue
        seen.add(key)
        dm = re.search(r"\[data:\s*(.*?)\]", rest)
        data_note = (dm.group(1) if dm else "").strip()
        out.append({"title": title, "tags": tags,
                    "summary": rest.split("[data:")[0].split(" src:")[0].strip()[:320],
                    "data_note": data_note[:140],
                    "free": bool(re.search(r"\b(free|owned)\b", data_note, re.I))})
    return out


def _work_queue() -> dict:
    out = {"queued": 0, "claimed": 0, "done": 0}
    for r in _read_jsonl(WIKI / ".queue" / "queue.jsonl"):
        st = r.get("status")
        if st in out:
            out[st] += 1
    return out


def _lanes() -> dict:
    """stem/id -> lane for everything the research map renders. ONE classifier."""
    lanes: dict[str, str] = {}
    for f in glob.glob(str(WIKI / "experiments" / "*.md")):
        p = Path(f)
        try:
            head = p.read_text(encoding="utf-8")[:2000]
        except OSError:
            continue
        fam = re.search(r"^family:\s*(.+)$", head, re.M)
        title = re.search(r"^#\s+(.+)$", head, re.M)
        lanes[p.stem] = lane_of(fam.group(1).strip() if fam else "",
                                title.group(1).strip() if title else p.stem)
    for r in _read_jsonl(WIKI / ".queue" / "queue.jsonl"):
        if r.get("id"):
            prop = r.get("proposal") or {}
            lanes[r["id"]] = lane_of(prop.get("premium") or "", r.get("title") or "")
    for f in glob.glob(str(WIKI / "premia" / "*.md")):
        p = Path(f)
        lanes[p.stem] = lane_of("", p.stem.replace("-", " ").replace("_", " "))
    return lanes


def build() -> dict:
    cycles = _parse_cycles(_read_jsonl(RUN_LOG))
    fdr = _parse_registry()
    candidates = _parse_candidates()
    work_q = _work_queue()

    n_exp = len(glob.glob(str(WIKI / "experiments" / "*.md")))
    n_src = len(glob.glob(str(WIKI / "sources" / "*.md")))
    n_premia = len(glob.glob(str(WIKI / "premia" / "*.md")))
    n_pat = len(glob.glob(str(WIKI / "patterns" / "*.md")))

    n_cycles = len(cycles)
    n_ran = sum(1 for c in cycles if c["ran"])
    n_pass = sum(1 for c in cycles if c["passed_all"])
    n_near = sum(1 for c in cycles if c["status"] == "near_miss")
    n_err = sum(1 for c in cycles if c["status"] == "error")
    best_h = max((c["metrics"]["holdout_sharpe"] for c in cycles
                  if c["metrics"]["holdout_sharpe"] is not None), default=None)

    log_tail: list[str] = []
    try:
        if CYCLE_LOG.exists():
            log_tail = CYCLE_LOG.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        pass

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "loop_disabled": LOOP_DISABLED.exists(),
        "summary": {
            "cycles": n_cycles, "ran": n_ran, "passes": n_pass, "near_misses": n_near,
            "fails": n_cycles - n_pass - n_near - n_err, "errors": n_err,
            "experiments": n_exp, "sources": n_src, "candidates": len(candidates),
            "families": fdr["n_families"], "wiki_pages": n_exp + n_src + n_premia + n_pat,
            "fdr_bar": fdr["bar"], "best_holdout_sharpe": best_h,
            "work_queue": work_q["queued"] + work_q["claimed"],
            "work_queue_detail": work_q, "scout_ideas": len(candidates),
            "free_candidates": sum(1 for c in candidates if c["free"]),
        },
        "fdr": fdr,
        "cycles": cycles[:50],
        "candidates": candidates[:14],
        "lanes": _lanes(),
        "lane_labels": LANE_LABELS,
        "log_tail": log_tail,
    }


def main() -> int:
    state = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    tmp.replace(OUT)
    print(f"[forge_state] wrote {OUT} — {state['summary']['cycles']} cycles, "
          f"{len(state['lanes'])} lane assignments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
