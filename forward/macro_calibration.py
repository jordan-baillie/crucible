"""forward/macro_calibration.py — §5 step-2 calibration for the MACRO-NEUTRALIZATION gate.

Per FROZEN pre-reg research-wiki/methodology/prereg-macro-neutralization-gate.md.
READ-ONLY settling experiment: for each forge strategy module, re-run signal(), slice the SEARCH
window (< holdout_start), neutralize those returns against the FROZEN macro factor block, and record
macro_r2 / gross & macro-neutral Sharpe / F-test p-value. Writes forward/macro_calibration.jsonl
(+ _summary.json). Calls signal() directly -> by construction CANNOT touch the write-once registry/holdout.

The summary observes the DISTRIBUTION (and factor-block VIF) so the provisional demotion thresholds
(MACRO_R2_HI=0.50, MACRO_SEL_FLOOR=0.40) can be confirmed/adjusted ONCE from the distribution's
structure. ANTI-HACKING (frozen §6): thresholds are NEVER moved to hit a survivor/would-demote count.

Run headless (systemd), parallel, resumable, per-worker memory cap.
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

OUT = ROOT / "forward" / "macro_calibration.jsonl"
SUMMARY = ROOT / "forward" / "macro_calibration_summary.json"
MEM_CAP_GB = 10
SIGNAL_TIMEOUT_S = 240

# FROZEN starts -> identical cache key across all strategies/workers (prewarmed once in main()).
EQUITY_START = "1998-01-01"
CRYPTO_START = "2017-01-01"
CRYPTO_COLS = ["btc", "eth", "usd", "gold", "vol"]  # pre-reg crypto sub-block (BTC/ETH + USD/gold/vol)


def _demotes(m: dict) -> bool:
    """FROZEN two-pronged demotion rule (preview only; harness does NOT demote yet). Explicit p-check
    avoids the `(p or default)` footgun where a 0.0 p-value (strongest signal) is falsy."""
    return bool(m.get("evaluated") and m["macro_r2"] > 0.50 and m["macro_residual_sharpe"] < 0.40
                and m.get("macro_block_pvalue") is not None and m["macro_block_pvalue"] < 0.05)


def _macro_matrix(is_crypto: bool):
    from sdk.adapters import macro_factor_returns
    if is_crypto:
        return macro_factor_returns(start=CRYPTO_START, include_crypto=True)[CRYPTO_COLS]
    return macro_factor_returns(start=EQUITY_START, include_crypto=False)  # 8 macro factors


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
        from sdk.harness import _macro_decomp, _sharpe

        spec_mod = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        SPEC = getattr(mod, "SPEC", None)
        if SPEC is None:
            return {**row, "skip": "no SPEC"}
        is_crypto = bool(cm.is_crypto(getattr(SPEC, "markets", None)))
        row.update({"id": SPEC.id, "family": SPEC.family, "markets": list(SPEC.markets or []),
                    "holdout_start": SPEC.holdout_start, "is_crypto": is_crypto})

        panel = SPEC.load_data()
        ret, _trades = SPEC.signal(panel, **dict(SPEC.default_params or {}))
        ret = pd.Series(ret).dropna()
        ret.index = pd.to_datetime(ret.index)
        search = ret[ret.index < pd.Timestamp(SPEC.holdout_start)]

        mx = _macro_matrix(is_crypto)
        dec = _macro_decomp(search, mx)
        row["macro"] = dec
        if dec.get("evaluated"):
            g, nresid = dec["gross_sharpe"], dec["macro_residual_sharpe"]
            row["sharpe_drop"] = round(g - nresid, 3)
            # PREVIEW under the FROZEN provisional thresholds (observation only; never tuned to this).
            # NB explicit `is not None` p-check: `(p or 1.0)` is a footgun — a p-value of exactly 0.0
            # (strongest significance) is falsy and would flip to 1.0, spuriously failing the gate.
            row["would_demote"] = _demotes(dec)
        return row
    except MemoryError:
        return {**row, "error": "MemoryError (hit per-worker cap)"}
    except Exception as e:
        return {**row, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-500:]}


def _vif(mx: pd.DataFrame) -> dict:
    """Variance Inflation Factor per factor (VIF_j = 1/(1-R²_j); >5-10 flags redundancy)."""
    df = mx.dropna()
    out = {}
    cols = list(df.columns)
    for j, c in enumerate(cols):
        y = df[c].values
        X = np.column_stack([np.ones(len(df))] + [df[o].values for o in cols if o != c])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
        out[c] = round(1.0 / (1.0 - r2), 2) if r2 < 1 else float("inf")
    return out


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    ev = [r for r in rows if r.get("macro", {}).get("evaluated")]
    ne = [r for r in rows if "macro" in r and not r["macro"].get("evaluated")]
    err = [r for r in rows if r.get("error")]
    skip = [r for r in rows if r.get("skip")]

    def dist(vals):
        a = np.array([v for v in vals if v is not None], float)
        if len(a) == 0:
            return {}
        qs = np.quantile(a, [0, .1, .25, .5, .75, .9, 1.0])
        return {"n": len(a), "min": round(qs[0], 3), "p10": round(qs[1], 3), "p25": round(qs[2], 3),
                "median": round(qs[3], 3), "p75": round(qs[4], 3), "p90": round(qs[5], 3),
                "max": round(qs[6], 3), "mean": round(float(a.mean()), 3)}

    r2 = [r["macro"]["macro_r2"] for r in ev]
    drop = [r.get("sharpe_drop") for r in ev]
    resid = [r["macro"]["macro_residual_sharpe"] for r in ev]
    gross = [r["macro"]["gross_sharpe"] for r in ev]
    would = [r for r in ev if _demotes(r["macro"])]  # recompute from metrics (ignore any buggy stored field)

    # factor-block VIF (equity 8-factor + crypto 5-factor), from the live matrices
    from sdk.adapters import macro_factor_returns
    vif_eq = _vif(macro_factor_returns(start=EQUITY_START))
    vif_cr = _vif(macro_factor_returns(start=CRYPTO_START, include_crypto=True)[CRYPTO_COLS])

    summ = {
        "n_modules": len(rows), "evaluated": len(ev), "not_evaluated": len(ne),
        "errors": len(err), "skipped": len(skip),
        "macro_r2_dist": dist(r2), "sharpe_drop_dist": dist(drop),
        "gross_sharpe_dist": dist(gross), "macro_residual_sharpe_dist": dist(resid),
        "would_demote_count": len(would),
        "would_demote_ids": [r["id"] for r in would],
        "vif_equity_block": vif_eq, "vif_crypto_block": vif_cr,
        "not_evaluated_reasons": {r["id"]: r["macro"]["note"] for r in ne if "id" in r},
        "errors_detail": {r.get("id", r["module"]): r["error"] for r in err},
        # highest-R² strategies (the macro-confounded suspects) for eyeballing
        "top_r2": sorted([{"id": r["id"], "family": r.get("family"), "is_crypto": r.get("is_crypto"),
                           "macro_r2": r["macro"]["macro_r2"], "gross": r["macro"]["gross_sharpe"],
                           "neutral": r["macro"]["macro_residual_sharpe"],
                           "p": r["macro"]["macro_block_pvalue"], "drop": r.get("sharpe_drop")}
                          for r in ev], key=lambda d: -d["macro_r2"])[:20],
    }
    SUMMARY.write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


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
    print(f"[macro_cal] {len(modules)} modules, {len(done)} done, {len(todo)} to do, workers={workers}", flush=True)

    if todo:
        print("[macro_cal] prewarming macro factor cache (equity + crypto)...", flush=True)
        try:
            _macro_matrix(False)
            _macro_matrix(True)
            print("[macro_cal] cache warm.", flush=True)
        except Exception as e:
            print(f"[macro_cal] prewarm warning: {e}", flush=True)

    with OUT.open("a") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(calibrate_one, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result(timeout=SIGNAL_TIMEOUT_S * 3)
            except Exception as e:
                row = {"module": Path(futs[fut]).stem, "error": f"future: {type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n")
            f.flush()
            m = row.get("macro", {})
            tag = (f"r2={m.get('macro_r2')} drop={row.get('sharpe_drop')} demote={row.get('would_demote')}"
                   if m.get("evaluated") else (row.get("error") or row.get("skip")
                   or (m.get("note", "?")[:40] if m else "?")))
            print(f"[{i}/{len(todo)}] {row.get('module','?')[:46]:46s} -> {tag}", flush=True)
    print("[macro_cal] runs done; summarizing...", flush=True)
    summarize()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main(workers=int(os.environ.get("MACRO_CAL_WORKERS", "4")))
