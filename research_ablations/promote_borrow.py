"""Deterministic full-gate promotion run of the hand-edited amihud v3 + $5M borrow floor.
Calls the harness run_experiment on the FROZEN ablation SPEC. No codegen, no tuning."""
import sys, importlib.util, json
sys.path.insert(0, "/root/crucible")
from sdk import harness

spec = importlib.util.spec_from_file_location(
    "amihud_ablation", "/root/crucible/research_ablations/amihud_v3_borrow_ablation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

# write_wiki=False so this research ablation does not pollute the production wiki registry/loops;
# the verdict is printed + saved to a local json. alert=False so no Telegram.
verdict = harness.run_experiment(mod.SPEC, write_wiki=False, alert=False)
out = "/root/crucible/research_ablations/borrow_promotion_verdict.json"
json.dump(verdict, open(out, "w"), indent=2, default=str)
print("VERDICT WRITTEN:", out)
keys = ["tier", "PASSED_ALL_GATES", "search_sharpe", "holdout_sharpe", "holdout_pass",
        "dsr", "promote_bar", "pbo", "median_cpcv", "mcpt_pass", "deployment_passed",
        "deploy_reasons", "beta_to_universe", "selection_alpha_sharpe", "beta_confound",
        "grid_sharpes"]
print(json.dumps({k: verdict.get(k) for k in keys}, indent=2, default=str))
