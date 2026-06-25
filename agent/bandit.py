"""Thompson bandit over the proposal ARMS (explore / refine / orthogonal / crossover).

SYNTHESIS_PLAN Stage 1c pre-registered a FIXED arm split *until the run_log holds >=60 arm-labelled
outcomes, then fit a Thompson bandit on the logged arm_reward*. That condition is now met, so this
replaces the fixed split with a data-driven allocation: budget flows to the arms that actually
produce gate-worthy ideas (the run_log shows `refine` = lowest screen-fail / highest reward) and
away from the proven-weak ones (`orthogonal`/`crossover` = ~0 promotes) — WITHOUT touching any gate.
It only changes WHICH ideas we spend a cycle on; everything that IS tested still faces the same
non-bypassable stack.

Guardrails (structural, not a one-off hand-tune):
  - EXPLORE_FLOOR (0.25): explore is HARD-floored so discovery (which found the only pass) can never
    be starved; the exploit arms feed on what explore finds.
  - ARM_EPS (0.03): every other arm keeps a small floor so an unlucky streak can't kill an arm — the
    bandit can always recover it.
  - N_MIN (60): below the pre-registered threshold -> fall back to the fixed split (fresh-machine safe).
  - reward = _arm_reward in [0, 2] (gate-progress weighted: 0.25 ran, 0.5 stage1, +DSR, 2.0 pass);
    normalized to [0,1] and fed to a Beta(1+sum r, 1+sum(1-r)) posterior per arm. The Thompson
    allocation = the fraction of K posterior draws each arm wins the argmax (posterior variance =
    automatic, principled exploration on top of the explore floor).

Self-correcting: refit from the run_log on every top-up, so the allocation tracks the pool/regime.
Reversible/steerable: CRUCIBLE_FORCE_ARM (handled in director) still overrides for directed runs.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from crucible_paths import RUN_LOG

ARMS = ("explore", "refine", "orthogonal", "crossover")
EXPLORE_FLOOR = 0.25     # operator-directed (2026-06-25): discovery insurance
ARM_EPS = 0.03           # no arm ever dies -> the bandit can always recover it
N_MIN = 60               # pre-registered data-first threshold (SYNTHESIS_PLAN Stage 1c)
REWARD_MAX = 2.0         # arm_reward full-pass floor; normalizes reward into [0,1]
FIXED_SPLIT = (("explore", 0.45), ("refine", 0.25), ("orthogonal", 0.15), ("crossover", 0.15))


def arm_reward(verdict: dict | None) -> float:
    """Scalar reward for the proposal arm that produced a run — the bandit's optimization target,
    and the SINGLE SOURCE of the reward definition (run_worker imports this; the bandit recomputes
    it from each run_log entry's verdict so a definition change applies consistently to all history).

    MONOTONE in gate-progress so the bandit optimizes IDEA QUALITY, not merely 'it ran'. A SCREEN_FAIL
    (no in-sample edge) MUST score strictly below a strategy that CLEARED the tier-0 screen (a real
    in-sample edge) but died at a later gate. The old flat 0.25-for-any-run conflated the two — which
    is exactly the no-edge waste this whole exercise is meant to down-weight.
        0.0  didn't run / no verdict (casualty)
        0.1  ran but SCREEN_FAIL  (|search Sharpe| < 0.3 — no in-sample edge)
        0.3  cleared the tier-0 screen (real in-sample edge) but failed before stage-1
        0.5 + DSR   stage-1 pass (full rails; DSR in [0,1])
        2.0  PASSED ALL GATES
    """
    if not verdict:
        return 0.0
    tier = str(verdict.get("tier") or "").upper()
    ssh = verdict.get("search_sharpe")
    no_edge = tier == "SCREEN_FAIL" or (isinstance(ssh, (int, float)) and abs(ssh) < 0.3)
    r = 0.1 if no_edge else 0.3
    if verdict.get("stage1_pass"):
        r = 0.5
        try:
            r += min(max(float(verdict.get("dsr") or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    if verdict.get("PASSED_ALL_GATES"):
        r = max(r, 2.0)
    return round(r, 3)


def _rewards_by_arm(run_log=RUN_LOG) -> dict:
    """arm -> list of rewards normalized to [0,1], recomputed from each entry's VERDICT via the
    canonical arm_reward() (NOT the stored cache) so the current reward definition applies to all
    history consistently."""
    by: dict[str, list[float]] = defaultdict(list)
    try:
        with open(run_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                a = d.get("arm")
                if a in ARMS:
                    by[a].append(min(max(arm_reward(d.get("verdict")) / REWARD_MAX, 0.0), 1.0))
    except FileNotFoundError:
        pass
    return by


def _thompson_alloc(by: dict, rng, draws: int) -> dict:
    """Beta posterior per arm (uniform prior); allocation = fraction of `draws` each arm wins the
    argmax. Arms with no data keep the uniform Beta(1,1) prior (max uncertainty -> sampled freely)."""
    post = {a: (1.0 + sum(by.get(a, [])), 1.0 + sum(1.0 - x for x in by.get(a, []))) for a in ARMS}
    wins = {a: 0 for a in ARMS}
    for _ in range(draws):
        best, bv = "explore", -1.0
        for a in ARMS:
            v = rng.betavariate(*post[a])
            if v > bv:
                bv, best = v, a
        wins[best] += 1
    return {a: wins[a] / draws for a in ARMS}


def _apply_floors(raw: dict) -> dict:
    """Hard explore floor + small eps on every other arm; the remaining mass is split among the
    exploit arms in proportion to their raw Thompson weight. Always sums to 1."""
    explore = max(raw.get("explore", 0.0), EXPLORE_FLOOR)
    others = [a for a in ARMS if a != "explore"]
    budget = 1.0 - explore
    base = ARM_EPS * len(others)
    extra = max(budget - base, 0.0)
    rawsum = sum(raw.get(a, 0.0) for a in others) or 1.0
    w = {"explore": explore}
    for a in others:
        w[a] = ARM_EPS + extra * (raw.get(a, 0.0) / rawsum)
    s = sum(w.values()) or 1.0
    return {a: w[a] / s for a in ARMS}


def arm_weights(run_log=RUN_LOG, rng=None, draws: int = 2000):
    """((arm, weight), ...) summing to 1. Data-driven Thompson allocation with the explore floor;
    falls back to the pre-registered FIXED_SPLIT below N_MIN labelled outcomes (fresh-machine safe).
    Order matches ARMS so the caller's cumulative-weight sampling is stable."""
    rng = rng or random
    by = _rewards_by_arm(run_log)
    if sum(len(v) for v in by.values()) < N_MIN:
        return FIXED_SPLIT
    w = _apply_floors(_thompson_alloc(by, rng, draws))
    return tuple((a, w[a]) for a in ARMS)


def format_alloc(weights) -> str:
    return " ".join(f"{a}={w*100:.0f}%" for a, w in weights)
