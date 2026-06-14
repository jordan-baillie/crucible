"""Deterministic ablation: amihud v3 + borrow floor (one variable, hand-edited real module).
Floor=0 MUST reproduce deployed v3 (search 1.889 / holdout 1.462) — the setup self-check."""
import sys, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, "/root/crucible")
from sdk.stats import sharpe          # the SAME sharpe the gate uses
HOLDOUT = "2022-01-01"

spec = importlib.util.spec_from_file_location(
    "amihud_ablation", "/root/crucible/research_ablations/amihud_v3_borrow_ablation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

print("loading panel (once)..."); panel = mod.load_data()
print(f"panel: {panel.shape}, {panel.index.min().date()}..{panel.index.max().date()}\n")

def conc(trades_or_w, daily):
    # rough single-name gross share at the final formation (deployment-sanity proxy)
    return None

print(f"{'floor':>10} | {'search_Sh':>9} | {'holdout_Sh':>10} | {'full_Sh':>8} | {'retain%':>7} | {'avg_shorts':>10}")
print("-"*72)
base_search = None
for floor in [0.0, 3_000_000.0, 5_000_000.0, 8_000_000.0]:
    daily, trades = mod.signal(panel, borrow_floor=floor)
    daily = pd.Series(daily).dropna()
    s = sharpe(daily[daily.index < HOLDOUT])
    h = sharpe(daily[daily.index >= HOLDOUT]) if (daily.index >= HOLDOUT).sum() > 20 else float("nan")
    f = sharpe(daily)
    if floor == 0.0:
        base_search = s
    retain = (s / base_search * 100.0) if base_search else float("nan")
    # avg short names per day (negative-weight tickers in the trade ledger is hard; approximate via signal weights)
    # count distinct short tickers across trades as a coarse breadth proxy
    short_names = len({t.get("ticker") for t in trades if str(t.get("side", "")).lower().startswith("s")}) if trades else 0
    print(f"{floor/1e6:>8.0f}M | {s:>9.3f} | {h:>10.3f} | {f:>8.3f} | {retain:>6.1f}% | {short_names:>10}")

print("\nSELF-CHECK: floor=0 search should be ~1.889, holdout ~1.462 (deployed v3).")
print("If floor=0 does NOT match v3, the copy diverged — fix before trusting floor>0 rows.")
