"""forward/breadth_calibration.py — calibration for the breadth/Fundamental-Law overfit gate (#49).

Per FROZEN pre-reg prereg-breadth-overfit-gate.md §4. READ-ONLY: re-run signal(), build the full gate
context (price_matrix + search-window trade ledger), call the real _gc_breadth_overfit Check, record the
implied_IC / effective_breadth / rho / N distribution. Confirms whether IMPLIED_IC_MAX=0.20 sits in a
clean gap above the realistic IC band and whether any genuinely-broad book is wrongly near the cliff.
NO registry/holdout interaction. Writes forward/breadth_calibration.jsonl (+ _summary.json).
"""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "forward" / "breadth_calibration.jsonl"
SUMMARY = ROOT / "forward" / "breadth_calibration_summary.json"
MEM_CAP_GB = 10
SIGNAL_TIMEOUT_S = 240


def calibrate_one(module_path: str) -> dict:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB * 1024**3, MEM_CAP_GB * 1024**3))
    except Exception:
        pass
    name = Path(module_path).stem
    row = {"module": name}
    try:
        import crucible_paths  # noqa
        from sdk import cost_model as cm
        from sdk.gates import GateContext
        from sdk.harness import _gc_breadth_overfit, _price_matrix

        spec_mod = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        SPEC = getattr(mod, "SPEC", None)
        if SPEC is None:
            return {**row, "skip": "no SPEC"}
        row.update({"id": SPEC.id, "family": SPEC.family, "markets": list(SPEC.markets or []),
                    "is_crypto": bool(cm.is_crypto(getattr(SPEC, "markets", None)))})

        panel = SPEC.load_data()
        ret, trades = SPEC.signal(panel, **dict(SPEC.default_params or {}))
        ret = pd.Series(ret).dropna(); ret.index = pd.to_datetime(ret.index)
        cut = pd.Timestamp(SPEC.holdout_start)
        search = ret[ret.index < cut]
        search_trades = [t for t in (trades or []) if str(t.get("entry_date", "")) < SPEC.holdout_start]

        ctx = GateContext(spec=SPEC, panel=panel, price_matrix=_price_matrix(panel), search=search,
                          search_trades=search_trades, holdout_pass=True, deploy_candidate=True)
        res = _gc_breadth_overfit(ctx)
        row["evaluated"] = res.evaluated
        row["metrics"] = res.metrics
        row["would_demote"] = bool(res.passed is False)
        return row
    except MemoryError:
        return {**row, "error": "MemoryError (hit per-worker cap)"}
    except Exception as e:
        return {**row, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-400:]}


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    ev = [r for r in rows if r.get("evaluated")]
    err = [r for r in rows if r.get("error")]
    ne = [r for r in rows if "evaluated" in r and not r["evaluated"]]

    def dist(vals):
        a = np.array([v for v in vals if v is not None], float)
        if len(a) == 0:
            return {}
        q = np.quantile(a, [0, .25, .5, .75, .9, .95, .99, 1.0])
        return {"n": len(a), "min": round(q[0], 4), "p25": round(q[1], 4), "median": round(q[2], 4),
                "p75": round(q[3], 4), "p90": round(q[4], 4), "p95": round(q[5], 4),
                "p99": round(q[6], 4), "max": round(q[7], 4)}

    iic = [r["metrics"].get("implied_ic") for r in ev]
    br = [r["metrics"].get("effective_breadth") for r in ev]
    nn = [r["metrics"].get("n_names") for r in ev]
    rho = [r["metrics"].get("avg_pairwise_corr") for r in ev]
    would = [r for r in ev if r.get("would_demote")]
    top = sorted([{"id": r["id"], "family": r.get("family"), "is_crypto": r.get("is_crypto"),
                   "implied_ic": r["metrics"].get("implied_ic"), "IR": r["metrics"].get("information_ratio"),
                   "eff_breadth": r["metrics"].get("effective_breadth"), "n_names": r["metrics"].get("n_names"),
                   "rho": r["metrics"].get("avg_pairwise_corr"), "would_demote": r.get("would_demote")}
                  for r in ev if r["metrics"].get("implied_ic") is not None],
                 key=lambda d: -d["implied_ic"])[:20]
    summ = {"n_modules": len(rows), "evaluated": len(ev), "not_evaluated": len(ne), "errors": len(err),
            "implied_ic_dist": dist(iic), "effective_breadth_dist": dist(br),
            "n_names_dist": dist(nn), "avg_pairwise_corr_dist": dist(rho),
            "IMPLIED_IC_MAX": 0.20, "would_demote_count": len(would),
            "would_demote_ids": [r["id"] for r in would], "top_implied_ic": top}
    SUMMARY.write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def main(workers: int = 6):
    modules = sorted(str(p) for p in (ROOT / "strategies").glob("auto_*.py"))
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            try:
                done.add(json.loads(l)["module"])
            except Exception:
                pass
    todo = [m for m in modules if Path(m).stem not in done]
    print(f"[breadth_cal] {len(modules)} modules, {len(done)} done, {len(todo)} to do, workers={workers}", flush=True)
    with OUT.open("a") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(calibrate_one, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result(timeout=SIGNAL_TIMEOUT_S * 3)
            except Exception as e:
                row = {"module": Path(futs[fut]).stem, "error": f"future: {type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n"); f.flush()
            m = row.get("metrics", {})
            tag = (f"iic={m.get('implied_ic')} brEff={m.get('effective_breadth')} N={m.get('n_names')} "
                   f"demote={row.get('would_demote')}" if row.get("evaluated")
                   else (row.get("error") or row.get("skip") or m.get("reason", "not_evaluated")))
            print(f"[{i}/{len(todo)}] {row.get('module','?')[:44]:44s} -> {tag}", flush=True)
    print("[breadth_cal] runs done; summarizing...", flush=True)
    summarize()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main(workers=int(os.environ.get("BREADTH_CAL_WORKERS", "6")))
