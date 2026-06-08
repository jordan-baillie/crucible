"""Hephaestus Strategy SDK — the fixed harness.

The agent writes ONLY a StrategySpec (a signal fn + data loader + pre-registration + grid).
This harness owns everything else and the rails are NON-BYPASSABLE: data split, the
research_integrity gates (CPCV/DSR/PBO + FDR bar + write-once holdout + deployment-sanity),
the FDR registry append, the verdict, the wiki write, and the Telegram-on-pass.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

# rails (per-project FDR/holdout state lives in the research-wiki so it's SHARED across agents)
os.environ.setdefault("RESEARCH_INTEGRITY_DIR", "/root/research-wiki/.registry")
import research_integrity as ri

WIKI = Path("/root/research-wiki")


@dataclass
class StrategySpec:
    """What the agent fills in. Everything else is the fixed harness."""
    id: str                                   # unique, kebab-case -> experiments/<id>.md
    family: str                               # FDR family (one idea); the bar rises per distinct family
    title: str
    markets: list                             # ["futures"], ["crypto"], ...
    data_desc: str                            # human description (free? owned? source)
    pre_registration: str                     # the FROZEN design text (written BEFORE running)
    load_data: Callable[[], pd.DataFrame]     # () -> panel the signal consumes
    signal: Callable[..., tuple]              # (panel, **params) -> (daily_returns: pd.Series, trades: list[dict])
    default_params: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)  # label -> params override (DSR effective-N; honest search burden)
    holdout_start: str = "2022-01-01"         # write-once quarantine
    deploy_max_positions: int = 10
    project: str = "hephaestus"


def _sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def _maxdd(r):
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def run_experiment(spec: StrategySpec, write_wiki=True, alert=True) -> dict:
    """Run one pre-registered hypothesis through ALL rails. Returns the verdict dict."""
    panel = spec.load_data()
    full_ret, trades = spec.signal(panel, **spec.default_params)
    full_ret = pd.Series(full_ret).dropna()
    search = full_ret[full_ret.index < spec.holdout_start]
    holdout = full_ret[full_ret.index >= spec.holdout_start]

    grid = {}
    for label, kw in (spec.grid or {"default": {}}).items():
        r = pd.Series(spec.signal(panel, **{**spec.default_params, **kw})[0]).dropna()
        grid[label] = r[r.index < spec.holdout_start].values

    # --- the gates (non-bypassable) ---
    bundle = ri.assemble_bundle(search.values, trades, grid_returns=grid)
    dep = ri.deployment_sanity(trades, strategy_meta={"max_positions": spec.deploy_max_positions})
    b = bundle if isinstance(bundle, dict) else {}

    s_sh, h_sh = _sharpe(search), _sharpe(holdout)
    deg = (h_sh - s_sh) / abs(s_sh) * 100 if abs(s_sh) > 0.1 else None
    h_pass, h_reasons = ri.holdout_gate(h_sh, deg, dep["passed"])
    # SEARCH-SANITY: a holdout "pass" is meaningless if there was no in-sample edge to confirm.
    # Guards the degenerate case search_sharpe~=0 -> degradation blows up -> spurious holdout pass
    # (this falsely flagged a credit-carry book that the trend over-blend had sunk to ~0 as a near-miss).
    if abs(s_sh) < 0.3:
        h_pass = False
        h_reasons = list(h_reasons) + [f"search Sharpe {s_sh:.2f} < 0.3 — no in-sample edge to confirm"]

    # --- FDR accounting + registry append: ATOMIC across all agents (multi-agent safe).
    #     The heavy CPCV (assemble_bundle) ran above OUTSIDE the lock; here we serialize only
    #     count -> bar -> evaluate -> append (milliseconds). This is what keeps the shared FDR
    #     bar correct when N agents test in parallel -- the non-negotiable Phase-3 invariant. ---
    from sdk.locks import FileLock
    with FileLock("fdr-registry", ttl=120):
        n_fam = ri.distinct_families(extra=spec.family)
        bar = ri.promote_dsr(n_fam)
        tiers = ri.evaluate_tiers(bundle, promote_dsr=bar)
        tier = str(tiers.get("tier"))
        passed_all = tier.upper() == "PROMOTE" and h_pass and dep["passed"]
        try:
            ri.registry.append_run({"strategy": spec.id, "family": spec.family, "tier": tier,
                                    "dsr": b.get("dsr"), "promote_dsr": bar, "n_families": n_fam,
                                    "holdout_touched": True, "passed_all": passed_all})
        except Exception:
            pass

    verdict = {
        "id": spec.id, "family": spec.family, "title": spec.title, "markets": spec.markets,
        "tier": tier, "promote_bar": round(bar, 3), "n_families": n_fam,
        "dsr": b.get("dsr"), "median_cpcv": b.get("median_cpcv_sharpe"), "pbo": b.get("pbo"),
        "deployment_passed": dep["passed"], "deploy_peak": dep.get("peak_concurrent"),
        "deploy_sectors": dep.get("sector_spread"), "deploy_reasons": dep.get("forced_fail_reasons"),
        "search_sharpe": round(s_sh, 3), "holdout_sharpe": round(h_sh, 3),
        "holdout_pass": h_pass, "holdout_reasons": h_reasons,
        "full_sharpe": round(_sharpe(full_ret), 3), "full_maxdd": round(_maxdd(full_ret), 3),
        "n_trades": len(trades), "PASSED_ALL_GATES": passed_all,
    }

    if write_wiki:
        from sdk.wiki import write_experiment
        write_experiment(spec, verdict)
    if alert and passed_all:
        from sdk.notify import telegram_pass
        telegram_pass(spec, verdict)
    return verdict
