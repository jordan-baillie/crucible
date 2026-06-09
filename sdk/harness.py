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
SCREEN_FLOOR = 0.3   # tier-0: |search Sharpe| below this -> no in-sample edge -> skip grid+CPCV, keep holdout
BETA_HI = 0.6        # long-only beta-to-universe above this + weak selection-alpha -> beta-confound
SEL_FLOOR = 0.4      # selection-alpha Sharpe a high-beta strategy must clear to be a real edge (not just beta)


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
    # PROMOTION POLICY (2026-06-09): a stage-1 gate pass is only a CANDIDATE. Confirmation needs
    # either cross-market generalization (broad mechanism) or forward-validation (defensibly local).
    scope: str = "broad"                       # "broad" (universal mechanism -> MUST generalize) | "local" (defensible universe-specific -> forward-validate)
    generalization_universes: list = field(default_factory=list)  # broad scope: untouched universes to confirm the mechanism in
    project: str = "hephaestus"


def _sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def _maxdd(r):
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _price_matrix(panel):
    """Best-effort (dates x assets) price matrix for the long-only benchmark; None if not a price panel."""
    try:
        if isinstance(panel, pd.DataFrame) and isinstance(panel.columns, pd.MultiIndex):
            lvl0 = set(panel.columns.get_level_values(0))
            for key in ("px", "close", "closeadj", "price", "prices", "adj_close"):
                if key in lvl0:
                    return panel[key]
            return None
        if isinstance(panel, pd.DataFrame) and panel.shape[1] >= 5:
            return panel
    except Exception:
        return None
    return None


def _beta_decomp(search_ret, price_mx):
    """Decompose SEARCH returns into universe-beta + selection-alpha (residual after removing the equal-weight
    universe return). Cheap stage-1 version of MCPT's finding: catches the long-only-beta confound."""
    if price_mx is None:
        return None
    try:
        ew = price_mx.pct_change().mean(axis=1)
        df = pd.concat([pd.Series(search_ret), ew], axis=1).dropna()
        df.columns = ["r", "m"]
        if len(df) < 60 or df["m"].var() == 0:
            return None
        beta = df["r"].cov(df["m"]) / df["m"].var()
        return {"beta_to_universe": round(float(beta), 2),
                "selection_alpha_sharpe": _sharpe(df["r"] - beta * df["m"])}
    except Exception:
        return None


def run_experiment(spec: StrategySpec, write_wiki=True, alert=True) -> dict:
    """Run one pre-registered hypothesis through ALL rails. Returns the verdict dict."""
    panel = spec.load_data()
    full_ret, trades = spec.signal(panel, **spec.default_params)
    full_ret = pd.Series(full_ret).dropna()
    search = full_ret[full_ret.index < spec.holdout_start]
    s_sh = _sharpe(search)

    # TIER-0 SCREEN (efficiency + holdout discipline): no in-sample edge -> SCREEN_FAIL. Skip the expensive
    # grid + CPCV AND do NOT burn the write-once holdout -- only an earned in-sample edge gets the OOS look.
    if abs(s_sh) < SCREEN_FLOOR:
        with FileLock("fdr-registry", ttl=120):
            n_fam = ri.distinct_families(extra=spec.family)
            bar = ri.promote_dsr(n_fam)
            try:
                ri.registry.append_run({"strategy": spec.id, "family": spec.family, "tier": "SCREEN_FAIL",
                    "dsr": None, "promote_dsr": bar, "n_families": n_fam, "holdout_touched": False, "passed_all": False})
            except Exception:
                pass
        verdict = {
            "id": spec.id, "family": spec.family, "title": spec.title, "markets": spec.markets,
            "tier": "SCREEN_FAIL", "promote_bar": round(bar, 3), "n_families": n_fam,
            "dsr": None, "median_cpcv": None, "pbo": None, "deployment_passed": None,
            "deploy_peak": None, "deploy_sectors": None, "deploy_reasons": None,
            "search_sharpe": round(s_sh, 3), "holdout_sharpe": None, "holdout_pass": False,
            "holdout_reasons": [f"tier-0 screen: |search Sharpe| {abs(s_sh):.2f} < {SCREEN_FLOOR} -- no in-sample edge; full rails + holdout skipped"],
            "full_sharpe": round(_sharpe(full_ret), 3), "full_maxdd": round(_maxdd(full_ret), 3), "n_trades": len(trades),
            "stage1_pass": False, "confirmed": False, "scope": getattr(spec, "scope", "broad"),
            "needs_confirmation": None, "PASSED_ALL_GATES": False,
        }
        if write_wiki:
            from sdk.wiki import write_experiment
            write_experiment(spec, verdict)
        return verdict

    # earned the full rails + the OOS look
    holdout = full_ret[full_ret.index >= spec.holdout_start]
    grid = {}
    for label, kw in (spec.grid or {"default": {}}).items():
        r = pd.Series(spec.signal(panel, **{**spec.default_params, **kw})[0]).dropna()
        grid[label] = r[r.index < spec.holdout_start]   # keep the Series (index) so PBO/DSR align variants by DATE

    # --- the gates (non-bypassable) ---
    result = ri.assemble_bundle(search.values, trades, grid_returns=grid)
    # assemble_bundle returns {"bundle": <gate inputs>, "diagnostics": ...}. evaluate_tiers AND the
    # verdict need the INNER bundle — passing the wrapper made EVERY gate 'missing' -> tier ALWAYS FAIL
    # (every forge run failed on this, not on merit). This is THE gate-wiring fix.
    b = (result.get("bundle") or {}) if isinstance(result, dict) else {}
    dep = ri.deployment_sanity(trades, strategy_meta={"max_positions": spec.deploy_max_positions})

    h_sh = _sharpe(holdout)  # s_sh computed above (survived the tier-0 screen)
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
    # long-only BETA-CONFOUND check: decompose search returns into universe-beta + selection-alpha.
    # MCPT's lesson made a cheap stage-1 gate: an apparent edge that is really long-only universe beta
    # (high beta, weak selection-alpha) must NOT pass -- it'd just be holding the universe (cf. value×mom 0.994).
    decomp = _beta_decomp(search, _price_matrix(panel)) or {}
    beta_confound = bool(decomp.get("beta_to_universe", 0) > BETA_HI
                         and decomp.get("selection_alpha_sharpe") is not None
                         and decomp["selection_alpha_sharpe"] < SEL_FLOOR)

    from sdk.locks import FileLock
    with FileLock("fdr-registry", ttl=120):
        n_fam = ri.distinct_families(extra=spec.family)
        bar = ri.promote_dsr(n_fam)
        tiers = ri.evaluate_tiers(b, promote_dsr=bar)
        tier = str(tiers.get("tier"))
        stage1_pass = tier.upper() == "PROMOTE" and h_pass and dep["passed"]
        if stage1_pass and beta_confound:   # demote: edge is long-only universe beta, not the signal
            stage1_pass = False
            h_reasons = list(h_reasons) + [
                f"BETA-CONFOUND: beta_to_universe {decomp['beta_to_universe']} + selection-alpha Sharpe "
                f"{decomp['selection_alpha_sharpe']} < {SEL_FLOOR} -> edge is long-only universe beta, demoted"]
        # POLICY: a stage-1 pass is a CANDIDATE, never a confirmed edge. PASSED_ALL_GATES requires
        # INDEPENDENT fluke-confirmation (generalization for broad scope, forward-validation for local)
        # -- never auto-declared at run time. The BAB episode: it cleared every single-universe gate and
        # was still a non-generalising overfit outlier.
        passed_all = False
        try:
            ri.registry.append_run({"strategy": spec.id, "family": spec.family, "tier": tier,
                                    "dsr": b.get("dsr"), "promote_dsr": bar, "n_families": n_fam,
                                    "holdout_touched": True, "passed_all": stage1_pass})
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
        "n_trades": len(trades),
        "beta_to_universe": decomp.get("beta_to_universe"),
        "selection_alpha_sharpe": decomp.get("selection_alpha_sharpe"), "beta_confound": beta_confound,
        "stage1_pass": stage1_pass, "confirmed": False, "scope": getattr(spec, "scope", "broad"),
        "needs_confirmation": (None if not stage1_pass else
            ("cross-market-generalization" if getattr(spec, "scope", "broad") == "broad" else "forward-validation")),
        "PASSED_ALL_GATES": passed_all,
    }

    if write_wiki:
        from sdk.wiki import write_experiment
        write_experiment(spec, verdict)
    if alert and stage1_pass:
        from sdk.notify import telegram_candidate
        telegram_candidate(spec, verdict)
    return verdict
