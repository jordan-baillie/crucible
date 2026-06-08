import sys; sys.path.insert(0, "/root/hephaestus")
from strategies.example_trend import SPEC
from sdk.harness import run_experiment
v = run_experiment(SPEC, write_wiki=True, alert=True)
print("\n=== HARNESS VERDICT ===")
for k in ("tier","dsr","holdout_sharpe","holdout_pass","deployment_passed","full_sharpe","n_trades","PASSED_ALL_GATES"):
    print(f"  {k}: {v[k]}")
