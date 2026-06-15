"""forward/cost_rescore.py — re-score the forge corpus under the FROZEN cost-aware deployability model.

Per pre-reg research-wiki/methodology/prereg-cost-aware-deployability-gate.md (FROZEN 2026-06-15).
For each forge strategy module: run signal() baseline, then re-priced under the liquidity ladder
(central AND conservative) with borrow-infeasible shorts zeroed; record search/holdout Sharpe, borrow
verdict, cost drag. Writes forward/cost_rescore.jsonl. NO registry/holdout interaction (calls signal()
directly — by construction cannot pollute the write-once ledger).

The re-score is a SCREEN: survivor_screen = borrow_feasible AND re-priced central holdout Sharpe > 0.
Screen survivors get a full-gate confirm in a follow-up (only meaningful if any survive).
Run headless (systemd) — parallel, resumable, per-worker memory cap (large-cap OOM history).
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

OUT = ROOT / "forward" / "cost_rescore.jsonl"
MEM_CAP_GB = 10  # per-worker address-space cap -> an OOM-prone module fails gracefully, never kills the box
SIGNAL_TIMEOUT_S = 240


def _sharpe(r: pd.Series) -> float:
    from sdk.stats import sharpe
    return float(sharpe(pd.Series(r).dropna()))


def _split(ret: pd.Series, holdout_start: str):
    ret = pd.Series(ret).dropna()
    ret.index = pd.to_datetime(ret.index)
    cut = pd.Timestamp(holdout_start)
    return ret[ret.index < cut], ret[ret.index >= cut]


def rescore_one(module_path: str) -> dict:
    """Subprocess worker: re-score ONE module under baseline + central + conservative ladders."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB * 1024**3, MEM_CAP_GB * 1024**3))
    except Exception:
        pass
    name = Path(module_path).stem
    row = {"module": name, "path": module_path}
    try:
        import crucible_paths  # noqa
        from sdk import signal_kit, cost_model as cm

        spec_mod = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        SPEC = getattr(mod, "SPEC", None)
        if SPEC is None:
            return {**row, "skip": "no SPEC"}
        row.update({"id": SPEC.id, "family": SPEC.family, "markets": list(SPEC.markets or []),
                    "holdout_start": SPEC.holdout_start, "has_short_hint": None})

        panel = SPEC.load_data()
        params = dict(SPEC.default_params or {})

        # --- baseline (strategy's own cost assumption) ---
        ret0, trades0 = SPEC.signal(panel, **params)
        s0, h0 = _split(ret0, SPEC.holdout_start)
        row["base_search_sharpe"] = round(_sharpe(s0), 3)
        row["base_holdout_sharpe"] = round(_sharpe(h0), 3)
        row["n_trades"] = len(trades0)

        # --- borrow verdict (ladder-independent) ---
        shortable = cm.shortable_set()
        bv = cm.borrow_verdict(trades0, shortable=shortable)
        row["borrow"] = bv
        row["has_short"] = any(float(t.get("position_value", 0)) < 0 for t in (trades0 or []))

        dv = cm.dollar_volume_map()
        repriced_via_kit = hasattr(mod, "net_of_cost")
        row["repriced_via_kit"] = bool(repriced_via_kit)

        def reprice(ladder):
            rec = {}
            patched = cm.make_net_of_cost(ladder, dv_map=dv, shortable=shortable, record=rec)
            orig_mod = getattr(mod, "net_of_cost", None)
            orig_kit = signal_kit.net_of_cost
            if orig_mod is not None:
                mod.net_of_cost = patched
            signal_kit.net_of_cost = patched
            try:
                r, _ = SPEC.signal(panel, **params)
            finally:
                if orig_mod is not None:
                    mod.net_of_cost = orig_mod
                signal_kit.net_of_cost = orig_kit
            s, h = _split(r, SPEC.holdout_start)
            return round(_sharpe(s), 3), round(_sharpe(h), 3), rec

        cs, ch, crec = reprice(cm.LADDER_CENTRAL)
        vs, vh, vrec = reprice(cm.LADDER_CONSERVATIVE)
        row["central"] = {"search_sharpe": cs, "holdout_sharpe": ch, **crec}
        row["conservative"] = {"search_sharpe": vs, "holdout_sharpe": vh, **vrec}

        # --- screen verdict (pre-reg §4): borrow-feasible AND re-priced central holdout > 0 ---
        survives = bool(bv["borrow_feasible"] and ch > 0 and repriced_via_kit)
        if not bv["borrow_feasible"]:
            cause = "borrow_infeasible"
        elif not repriced_via_kit:
            cause = "custom_cost_not_repriced"  # only borrow assessed; liquidity unknown
        elif ch <= 0:
            cause = "cost_killed"
        else:
            cause = "survives_screen"
        row["survives_screen"] = survives
        row["death_cause"] = cause
        return row
    except MemoryError:
        return {**row, "error": "MemoryError (hit per-worker cap)"}
    except Exception as e:
        return {**row, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}


def main(workers: int = 4):
    modules = sorted(str(p) for p in (ROOT / "strategies").glob("auto_*.py"))
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            try:
                done.add(json.loads(l)["module"])
            except Exception:
                pass
    todo = [m for m in modules if Path(m).stem not in done]
    print(f"[cost_rescore] {len(modules)} modules, {len(done)} done, {len(todo)} to do, workers={workers}", flush=True)

    with OUT.open("a") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rescore_one, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result(timeout=SIGNAL_TIMEOUT_S * 3)
            except Exception as e:
                row = {"module": Path(futs[fut]).stem, "error": f"future: {type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n")
            f.flush()
            tag = row.get("death_cause") or row.get("error") or row.get("skip") or "?"
            print(f"[{i}/{len(todo)}] {row.get('module','?')[:50]:50s} -> {tag}", flush=True)
    print("[cost_rescore] done", flush=True)


if __name__ == "__main__":
    main(workers=int(os.environ.get("RESCORE_WORKERS", "4")))
