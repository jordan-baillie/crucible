"""Part C of the regime-burner pre-registration (2026-06-12) — ONE-OFF, REPORT-ONLY retro audit.

Runs the frozen Part-A calm/turbulent split against every deployed strategy module and elite-pool
member that can be re-executed locally. NO state changes, NO retroactive kills: results go to a
wiki page + stdout for human review (pre-registered interpretation: a retro fail is information,
not an automated action — these strategies cleared the stack as it stood).

  python3 scripts/retro_regime_audit.py
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible_paths import WIKI
from sdk.harness import _regime_split, _price_matrix

# deployed/forward strategy modules (resolved from the Atlas live registry's target.json strategy_path)
TARGETS = {
    "val_mom_trend_smallcap (deployed shadow)":
        "strategies.auto_value_momentum_complementary_combination_smith2_96154",
    "amihud_illiq_tranched_v3 (deployed shadow)":
        "strategies.auto_amihud_illiquidity_premium_deployable_sh_smith1_99153",
}


def _audit_module(label: str, mod_name: str) -> dict:
    try:
        m = importlib.import_module(mod_name)
        spec = m.SPEC
        panel = spec.load_data()
        full_ret, _ = spec.signal(panel, **spec.default_params)
        import pandas as pd
        full_ret = pd.Series(full_ret).dropna()
        search = full_ret[full_ret.index < spec.holdout_start]
        res = _regime_split(search, _price_matrix(panel))
        return {"target": label, "module": mod_name, **res}
    except Exception as e:
        return {"target": label, "module": mod_name, "evaluated": False,
                "reason": f"audit error: {type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    results = [_audit_module(lbl, mod) for lbl, mod in TARGETS.items()]
    # elite pool: audit via recorded module files where the run_log links one
    from agent import elite
    for it in elite.top():
        rid = it.get("id") or ""
        mod = f"strategies.{rid.replace('-', '_')}"
        if (ROOT / "strategies" / f"{rid.replace('-', '_')}.py").exists():
            results.append(_audit_module(f"elite: {it['title'][:50]}", mod))
        else:
            results.append({"target": f"elite: {it['title'][:50]}", "module": mod,
                            "evaluated": False, "reason": "module file not on disk (legacy elite)"})

    lines = [f"# Retro regime audit — {date.today()} (Part C, report-only; pre-reg 2026-06-12)",
             "", "NO automated action follows from this page. Human review only.", ""]
    for r in results:
        if r.get("evaluated"):
            verdictline = (f"calm={r['sharpe_calm']} turbulent={r['sharpe_turbulent']} "
                           f"-> {'PASS' if r['pass'] else '**RETRO-FAIL (review)**'}")
        else:
            verdictline = r.get("reason", "not evaluated")
        lines.append(f"- **{r['target']}** (`{r['module']}`): {verdictline}")
        print(f"{r['target']}: {verdictline}")
    out = WIKI / "methodology" / f"retro-regime-audit-{date.today()}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwiki page: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
