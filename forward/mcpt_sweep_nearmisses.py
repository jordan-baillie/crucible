"""Retroactive MCPT sweep over the 4 un-tested PROMOTE-tier near-misses (all predate the MCPT gate).

Uses sdk.harness._stage2_mcpt — the EXACT function now in the automated gate — so each result is
precisely what the rails would have said. Purpose:
  1) val_mom_trend_smallcap is DEPLOYED (forward-paper). If it fails MCPT it must be pulled —
     a construction artifact must not accumulate a track record toward the real-capital gate.
  2) Calibrate the artifact rate across everything stage-1 ever promoted.

Strategies (all tier PROMOTE at their run time):
  val_mom_trend_smallcap             — DEPLOYED (beta 0.36, sel-alpha 0.74 — clean stage-1) — the stakes
  value_mom_smallmid_sectorneutral   — dead (beta-confound demoted) — calibration only
  value_momentum_complementary_xs_v1 — dead (beta-confound demoted) — calibration only
  value_bm_smallcap_lowturn          — dead (beta-confound demoted, beta 30) — calibration only

Results -> forward/mcpt_sweep_results.json (atomic), one entry per strategy, report-ALL.
"""
import importlib
import json
import os
import sys
import time

sys.path.insert(0, "/root/hephaestus")
import pandas as pd

from sdk.harness import _stage2_mcpt, _sharpe

TARGETS = [
    # (module, deployed?, beta_to_universe from the original verdict — selects the MCPT null:
    #  beta > 0.3 -> benchmark-RELATIVE stat (long-biased books), else absolute)
    ("strategies.auto_value_momentum_complementary_combination_smith2_96154", True, 0.36),   # val_mom_trend_smallcap
    ("strategies.auto_value_momentum_complementary_combination_smith1_99655", False, 0.71),  # value_mom_smallmid_sectorneutral
    ("strategies.auto_value_momentum_complementary_combination_smith3_26200", False, 0.96),  # value_momentum_complementary_xs_v1
    ("strategies.auto_variant_cost_fragility_hardened_low_turn_smith1_28149", False, 30.18), # value_bm_smallcap_lowturn (ran last: panel OOM'd once)
]
OUT = "/root/hephaestus/forward/mcpt_sweep_results.json"


def main():
    results = {}
    for mod_name, deployed, beta in TARGETS:
        t0 = time.time()
        entry = {"module": mod_name, "deployed": deployed, "beta_to_universe": beta}
        try:
            m = importlib.import_module(mod_name)
            entry["id"] = m.SPEC.id
            print(f"[sweep] {m.SPEC.id}: loading panel...", flush=True)
            panel = m.load_data()
            real = _sharpe(pd.Series(m.signal(panel, **m.SPEC.default_params)[0]).dropna())
            entry["real_sharpe"] = round(real, 3)
            print(f"[sweep] {m.SPEC.id}: real Sharpe {real:.3f} ({time.time()-t0:.0f}s) — MCPT 50 perms "
                  f"(beta {beta} -> {'bench-rel' if beta > 0.3 else 'absolute'} null)...", flush=True)
            mcpt_res, mcpt_pass = _stage2_mcpt(m.SPEC, panel, real, n=50, beta_to_universe=beta)
            entry["mcpt"] = mcpt_res
            entry["mcpt_pass"] = mcpt_pass
            del panel
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[sweep] {mod_name} FAILED: {entry['error']}", flush=True)
        entry["elapsed_s"] = round(time.time() - t0, 1)
        results[entry.get("id", mod_name)] = entry
        print(f"[sweep] {entry.get('id', mod_name)}: mcpt_pass={entry.get('mcpt_pass')} "
              f"({entry['elapsed_s']}s)", flush=True)
        # incremental write so partial progress survives
        with open(OUT + ".tmp", "w") as f:
            json.dump(results, f, indent=2)
        os.replace(OUT + ".tmp", OUT)
    n_fail = sum(1 for r in results.values() if r.get("mcpt_pass") is False)
    print(f"[sweep] DONE: {len(results)} tested, {n_fail} MCPT-fail -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
