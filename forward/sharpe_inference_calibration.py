"""forward/sharpe_inference_calibration.py — calibration for the Sharpe-inference gate (#48).

Per FROZEN pre-reg prereg-sharpe-inference-gate.md §3. READ-ONLY: re-run signal(), slice the SEARCH
window, call the real _gc_sharpe_inference Check, record the Lo-deflation / PSR / MinTRL distribution.
Confirms whether LO_DEFLATION_FLOOR=0.70 / LO_SHARPE_FLOOR=0.5 sit in a clean gap and \u2014 critically \u2014
whether any LEGITIMATE autocorrelated book (trend/carry) hugs the cliff (false-positive risk).
NO registry/holdout interaction. Writes forward/sharpe_inference_calibration.jsonl (+ _summary.json).
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

OUT = ROOT / "forward" / "sharpe_inference_calibration.jsonl"
SUMMARY = ROOT / "forward" / "sharpe_inference_calibration_summary.json"
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
        from sdk.harness import _gc_sharpe_inference

        spec_mod = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        SPEC = getattr(mod, "SPEC", None)
        if SPEC is None:
            return {**row, "skip": "no SPEC"}
        row.update({"id": SPEC.id, "family": SPEC.family, "markets": list(SPEC.markets or []),
                    "is_crypto": bool(cm.is_crypto(getattr(SPEC, "markets", None)))})

        panel = SPEC.load_data()
        ret, _ = SPEC.signal(panel, **dict(SPEC.default_params or {}))
        ret = pd.Series(ret).dropna()
        ret.index = pd.to_datetime(ret.index)
        search = ret[ret.index < pd.Timestamp(SPEC.holdout_start)]

        ctx = GateContext(spec=SPEC, panel=None, price_matrix=None, search=search,
                          search_trades=[], holdout_pass=True, deploy_candidate=True)
        res = _gc_sharpe_inference(ctx)
        row["evaluated"] = res.evaluated
        row["metrics"] = res.metrics
        row["would_demote"] = bool(res.passed is False)   # active=False today, but this is what WOULD demote
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
        q = np.quantile(a, [0, .05, .1, .25, .5, .75, .9, .95, 1.0])
        return {"n": len(a), "min": round(q[0], 3), "p5": round(q[1], 3), "p10": round(q[2], 3),
                "p25": round(q[3], 3), "median": round(q[4], 3), "p75": round(q[5], 3),
                "p90": round(q[6], 3), "p95": round(q[7], 3), "max": round(q[8], 3)}

    defl = [r["metrics"].get("lo_deflation_factor") for r in ev]
    lo_adj = [r["metrics"].get("lo_adjusted_sharpe") for r in ev]
    naive = [r["metrics"].get("naive_sharpe") for r in ev]
    psr = [r["metrics"].get("psr_vs_zero") for r in ev]
    would = [r for r in ev if r.get("would_demote")]
    # DANGER ZONE: legit books near the cliff (deflation in [0.60,0.80] -> some autocorrelation,
    # the zone where a real trend/carry book could be wrongly demoted). The false-positive check.
    danger = sorted([{"id": r["id"], "family": r.get("family"), "is_crypto": r.get("is_crypto"),
                      "deflation": r["metrics"].get("lo_deflation_factor"),
                      "lo_adj_sharpe": r["metrics"].get("lo_adjusted_sharpe"),
                      "naive_sharpe": r["metrics"].get("naive_sharpe"),
                      "would_demote": r.get("would_demote")}
                     for r in ev if r["metrics"].get("lo_deflation_factor") is not None
                     and r["metrics"]["lo_deflation_factor"] < 0.85],
                    key=lambda d: d["deflation"])

    summ = {
        "n_modules": len(rows), "evaluated": len(ev), "not_evaluated": len(ne), "errors": len(err),
        "lo_deflation_factor_dist": dist(defl), "lo_adjusted_sharpe_dist": dist(lo_adj),
        "naive_sharpe_dist": dist(naive), "psr_vs_zero_dist": dist(psr),
        "would_demote_count": len(would), "would_demote_ids": [r["id"] for r in would],
        "thresholds": {"LO_DEFLATION_FLOOR": 0.70, "LO_SHARPE_FLOOR": 0.5},
        "most_autocorrelated_books_deflation_lt_0.85": danger,
    }
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
    print(f"[sharpe_cal] {len(modules)} modules, {len(done)} done, {len(todo)} to do, workers={workers}", flush=True)
    with OUT.open("a") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(calibrate_one, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result(timeout=SIGNAL_TIMEOUT_S * 3)
            except Exception as e:
                row = {"module": Path(futs[fut]).stem, "error": f"future: {type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n"); f.flush()
            m = row.get("metrics", {})
            tag = (f"defl={m.get('lo_deflation_factor')} lo_adj={m.get('lo_adjusted_sharpe')} "
                   f"demote={row.get('would_demote')}" if row.get("evaluated")
                   else (row.get("error") or row.get("skip") or "not_evaluated"))
            print(f"[{i}/{len(todo)}] {row.get('module','?')[:46]:46s} -> {tag}", flush=True)
    print("[sharpe_cal] runs done; summarizing...", flush=True)
    summarize()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main(workers=int(os.environ.get("SHARPE_CAL_WORKERS", "6")))
