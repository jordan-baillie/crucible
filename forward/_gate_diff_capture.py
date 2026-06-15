"""Behaviour-preservation harness for the gate-system refactor (design-gate-system-unification.md).

Runs run_experiment() on deterministic synthetic specs in an isolated tmp wiki/registry and dumps the
gate-decision verdict fields. Run on OLD harness -> baseline, on NEW harness -> compare. Identical =
behaviour preserved (the macro demotion is date-gated to 2026-06-29 so it must be inert today).

    python3 forward/_gate_diff_capture.py > /tmp/gate_X.json
"""
import os, sys, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))


def setup_env():
    d = Path(tempfile.mkdtemp())
    wiki = d / "wiki"
    for s in ["experiments", "patterns", "decisions", ".queue", ".locks", ".registry", ".elite"]:
        (wiki / s).mkdir(parents=True)
    (wiki / "log.md").write_text("# log\n"); (wiki / "index.md").write_text("# index\n")
    ris = d / "ri"; ris.mkdir()
    os.environ["CRUCIBLE_WIKI"] = str(wiki); os.environ["CRUCIBLE_DEPLOY"] = ""
    os.environ["RESEARCH_INTEGRITY_DIR"] = str(ris)
    for m in list(sys.modules):
        if m.startswith(("sdk", "crucible_paths", "research_integrity")):
            sys.modules.pop(m)


def _panel(n=2200, k=8, seed=7, drift=0.0):
    rng = np.random.default_rng(seed); idx = pd.bdate_range(end="2026-06-01", periods=n)
    p = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, (n, k)), axis=0))
    return pd.DataFrame(p, index=idx, columns=[f"A{i}" for i in range(k)])


def _trades(w, p):
    out = []
    for dt, row in w.resample("ME").last().dropna(how="all").iterrows():
        for t, wt in row.dropna().items():
            if abs(wt) > 0.01:
                out.append({"ticker": t, "entry_date": str(dt.date()), "exit_date": str(dt.date()),
                            "position_value": float(wt) * 1e5, "pnl": 0.0, "sector": t})
    return out


def _momo(p, lookback=60, **_):
    rets = p.pct_change(); w = p.pct_change(lookback).clip(-1, 1); w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
    daily = (w * rets).sum(axis=1); boost = pd.Series(0.0006, index=daily.index); boost[daily.index >= "2024-01-01"] = 0.0
    return daily + boost, _trades(w, p)


def _longonly(p, **_):  # ~equal-weight long-only = high beta to EW universe (beta-confound territory if tier-0 passes)
    rets = p.pct_change(); w = pd.DataFrame(1.0 / p.shape[1], index=p.index, columns=p.columns).shift(1)
    daily = (w * rets).sum(axis=1); boost = pd.Series(0.0009, index=daily.index); boost[daily.index >= "2024-01-01"] = 0.0
    return daily + boost, _trades(w, p)


def run():
    setup_env()
    from sdk import harness as H

    def mk(sig, sid, grid=None):
        return H.StrategySpec(id=sid, family=f"t_{sid}", title=sid, markets=["test"], data_desc="syn",
                              pre_registration="FROZEN", load_data=lambda: _panel(), signal=sig,
                              default_params={}, grid=grid or {}, holdout_start="2024-01-01",
                              deploy_max_positions=8, scope="local")
    cases = [("momo", _momo, {"default": {}, "lb120": {"lookback": 120}}),
             ("longonly", _longonly, {"default": {}})]
    out = {}
    for name, sig, grid in cases:
        v = H.run_experiment(mk(sig, f"t-{name}", grid), write_wiki=False, alert=False)
        out[name] = {k: v.get(k) for k in (
            "tier", "stage1_pass", "PASSED_ALL_GATES", "holdout_pass", "holdout_reasons",
            "beta_confound", "beta_to_universe", "selection_alpha_sharpe",
            "macro_r2", "macro_residual_sharpe", "search_sharpe", "holdout_sharpe",
            "dsr", "median_cpcv", "pbo")}
        out[name]["regime_pass"] = (v.get("regime_split") or {}).get("pass")
        out[name]["regime_cov_ok"] = (v.get("regime_coverage") or {}).get("ok")
        out[name]["deployable"] = (v.get("deployability") or {}).get("deployable")
    print(json.dumps(out, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    run()
