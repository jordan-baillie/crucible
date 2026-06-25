"""Stage 1 (SYNTHESIS_PLAN.md): MAP-Elites grid invariants — ≤1 entry per cell, closed-family
exclusion, beta-confound zeroing, legacy migration, sample/sample_pair semantics, arm selection."""
import json
import random

import pytest


@pytest.fixture()
def pool(tmp_path, monkeypatch):
    """Point the elite pool + closed-families at a temp dir; reload module wiring."""
    import crucible_paths
    import agent.elite as elite
    p = tmp_path / "elite.jsonl"
    cf = tmp_path / "closed-families.txt"
    monkeypatch.setattr(elite, "POOL", p)
    monkeypatch.setattr(elite, "CLOSED_FAMILIES", cf)
    monkeypatch.setattr("sdk.locks.FileLock.acquire", lambda self: self, raising=False)
    monkeypatch.setattr("sdk.locks.FileLock.release", lambda self: None, raising=False)
    monkeypatch.setattr("sdk.locks.FileLock.__enter__", lambda self: self, raising=False)
    monkeypatch.setattr("sdk.locks.FileLock.__exit__", lambda self, *a: None, raising=False)
    return elite, p, cf


def _outcome(title, market, dsr, n_trades=500, beta_confound=False, ts="2026-06-12"):
    return {"id": f"id-{title[:8]}", "title": title, "ts": ts,
            "proposal": {"title": title, "market": market, "premium": title},
            "verdict": {"dsr": dsr, "n_trades": n_trades, "beta_confound": beta_confound,
                        "holdout_sharpe": 1.0, "search_sharpe": 1.2, "scope": "broad", "tier": "A"}}


def test_one_entry_per_cell(pool):
    elite, p, _ = pool
    elite.record(_outcome("Amihud illiquidity premium v1", "US small-cap equities", 0.8))
    elite.record(_outcome("Amihud illiquidity premium v2 variant", "US small-cap equities", 0.9))
    elite.record(_outcome("Amihud illiquidity premium v3 variant", "US small-cap equities", 0.7))
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    cells = [i["cell"] for i in items]
    assert len(cells) == len(set(cells)), "duplicate cell occupancy"
    # same family+universe+band -> exactly ONE survives, the best
    assert len(items) == 1 and items[0]["fitness"] == 0.9


def test_weaker_never_displaces(pool):
    elite, p, _ = pool
    elite.record(_outcome("Carry premium strong", "futures", 0.95))
    elite.record(_outcome("Carry premium weak variant", "futures", 0.6))
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 1 and items[0]["fitness"] == 0.95


def test_different_families_coexist(pool):
    elite, p, _ = pool
    elite.record(_outcome("Amihud illiquidity premium", "US small-cap equities", 0.9))
    elite.record(_outcome("Seasonal turn-of-month effect", "US small-cap equities", 0.7))
    elite.record(_outcome("Carry roll yield premium", "futures", 0.8))
    assert len(p.read_text(encoding="utf-8").splitlines()) == 3


def test_min_fitness_and_beta_confound_rejected(pool):
    elite, p, _ = pool
    elite.record(_outcome("Quality premium weak", "US equities", 0.3))            # below MIN_FIT
    elite.record(_outcome("Value premium confounded", "US equities", 0.9, beta_confound=True))
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == ""


def test_holdout_failed_strategy_is_not_an_elite(pool):
    """META-LESSONS #7 (trust only the holdout): a strategy the holdout RAN and REJECTED must never be
    a high-fitness exploit parent, however strong its search DSR. Root cause of crypto basis-carry
    being re-spawned 7x (each time re-failing the holdout for decay). 2026-06-16."""
    elite, p, _ = pool
    o = _outcome("Crypto basis-carry decayed", "crypto", 0.99)  # great in-sample DSR...
    o["verdict"]["holdout_pass"] = False                        # ...but the holdout rejected it (decay)
    o["verdict"]["holdout_sharpe"] = 2.8
    assert elite._fitness(o["verdict"]) == 0.0
    elite.record(o)
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == ""


def test_holdout_passed_strategy_is_admitted_and_persists_flag(pool):
    elite, p, _ = pool
    o = _outcome("Amihud illiquidity holdout-clean", "US small-cap equities", 0.9)
    o["verdict"]["holdout_pass"] = True
    elite.record(o)
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 1 and items[0]["fitness"] == 0.9
    assert items[0]["summary"].get("holdout_pass") is True  # persisted so future pool-cleans can see it


def test_pbo_and_blocker_are_stored_for_refine(pool):
    """Audit 2026-06-25: the loop was PBO-blind. PBO + the primary blocker must be persisted in the
    summary so the refine prompt can ATTACK the actual wall (PBO is the #1 near-miss blocker)."""
    elite, p, _ = pool
    o = _outcome("Amihud illiquidity PBO-blocked", "US small-cap equities", 0.9999)
    o["verdict"].update(holdout_pass=True, pbo=0.81)         # holdout passed, but overfit config
    elite.record(o)
    s = json.loads(p.read_text(encoding="utf-8").splitlines()[0])["summary"]
    assert s.get("pbo") == 0.81
    assert "PBO" in s["blocker"] and "tranche" in s["blocker"].lower()   # steers toward the proven fix


def test_lower_pbo_supersedes_on_dsr_tie(pool):
    """DSR is saturated (~0.9999) across a family, so on a fitness TIE keep the LOWER-PBO occupant —
    it is genuinely closer to a pass. Without this, an arbitrary high-PBO incumbent blocks the cell."""
    elite, p, _ = pool
    hi = _outcome("Amihud high-PBO", "US small-cap equities", 0.9999)
    hi["verdict"].update(holdout_pass=True, pbo=0.80)
    lo = _outcome("Amihud low-PBO variant", "US small-cap equities", 0.9999)
    lo["verdict"].update(holdout_pass=True, pbo=0.30)
    elite.record(hi)
    elite.record(lo)                                        # same fitness, lower PBO -> must displace
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 1 and items[0]["summary"]["pbo"] == 0.30
    # and the reverse order must NOT let the high-PBO one displace the low-PBO incumbent
    elite.record(hi)
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 1 and items[0]["summary"]["pbo"] == 0.30


def test_nan_fitness_rejected(pool):
    """NaN DSR used to BYPASS the MIN_FIT guard (nan <= 0.5 is False) and pollute the pool."""
    elite, p, _ = pool
    o = _outcome("Degenerate NaN-DSR strat", "crypto", float("nan"))
    o["verdict"]["holdout_pass"] = True  # even with a 'passed' holdout, NaN fitness is junk
    assert elite._fitness(o["verdict"]) == 0.0
    elite.record(o)
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == ""


def test_holdout_not_run_is_unaffected(pool):
    """SCREEN/explore strategies that never reached the holdout (holdout_sharpe=None) keep the old
    behaviour — the new guard only fires when the holdout actually RAN and failed."""
    elite, p, _ = pool
    o = _outcome("Term-structure carry screened", "futures", 0.8)
    o["verdict"]["holdout_sharpe"] = None      # holdout never ran
    o["verdict"]["holdout_pass"] = False       # default-ish False, but holdout_sharpe None -> not a 'rejection'
    assert elite._fitness(o["verdict"]) == 0.8


def test_closed_family_never_recorded_or_sampled(pool):
    elite, p, cf = pool
    elite.record(_outcome("Momentum 12-1 premium", "US equities", 0.9))
    cf.write_text("momentum\n", encoding="utf-8")
    elite.record(_outcome("Momentum 12-1 again", "US equities", 0.95))  # post-close: rejected
    assert elite.sample(random.Random(1)) is None  # pre-close entry filtered at read time too
    assert elite.sample_pair(random.Random(1)) is None


def test_legacy_migration_rebins(pool):
    """Old-format items (no cell/summary keys) load, re-bin, and collide correctly."""
    elite, p, _ = pool
    legacy = [
        {"id": "a", "fitness": 1.0, "title": "Amihud illiquidity premium",
         "proposal": {"title": "Amihud illiquidity premium", "market": "US small + mid-cap equities"},
         "ts": "2026-06-01"},
        {"id": "b", "fitness": 0.96, "title": "Amihud illiquidity premium variant v2",
         "proposal": {"title": "Amihud illiquidity premium variant v2", "market": "US small + mid-cap equities"},
         "ts": "2026-06-02"},
        {"id": "c", "fitness": 0.98, "title": "Crypto funding-carry delta-neutral",
         "proposal": {"title": "Crypto funding-carry delta-neutral", "market": "crypto (BTC, ETH perps)"},
         "ts": "2026-06-03"},
    ]
    p.write_text("".join(json.dumps(i) + "\n" for i in legacy))
    top = elite.top()
    fams = {elite._family(i) for i in top}
    assert len(top) == 2 and {"id-newamih"} is not None  # amihud collision -> best kept; crypto kept
    assert {i["id"] for i in top} == {"a", "c"}
    # recording a new better same-cell item persists the migrated (deduped) grid
    elite.record(_outcome("Amihud illiquidity premium v9", "US small + mid-cap equities", 1.0, n_trades=400))
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    amihud = [i for i in items if elite._family(i) == "amihud" or "amihud" in i["title"].lower()]
    assert all("cell" in i for i in items)


def test_sample_uniform_over_cells(pool):
    elite, p, _ = pool
    elite.record(_outcome("Amihud illiquidity premium", "US small-cap equities", 1.0))
    elite.record(_outcome("Seasonal turn-of-month", "etf rotation", 0.6))
    rng = random.Random(42)
    seen = {elite.sample(rng)["id"] for _ in range(60)}
    assert len(seen) == 2, "uniform-over-cells must reach every occupied cell"


def test_sample_pair_different_families(pool):
    elite, p, _ = pool
    elite.record(_outcome("Amihud illiquidity premium", "US small-cap equities", 1.0))
    elite.record(_outcome("Amihud illiquidity variant lowturn", "US small-cap equities", 0.9, n_trades=100))
    assert elite.sample_pair(random.Random(1)) is None, "single family must not pair with itself"
    elite.record(_outcome("Carry roll-yield premium", "futures", 0.8))
    for seed in range(10):
        a, b = elite.sample_pair(random.Random(seed))
        assert elite._family(a) != elite._family(b)


def test_director_arm_split_and_fallback(pool, monkeypatch):
    elite, p, _ = pool
    import agent.director as director
    monkeypatch.setattr(director, "elite", elite)
    calls = []
    monkeypatch.setattr(director, "propose", lambda: calls.append("explore") or {"title": "x"})
    monkeypatch.setattr(director, "propose_mutate", lambda e: calls.append("refine") or {"title": "m"})
    monkeypatch.setattr(director, "propose_orthogonal", lambda e: calls.append("orthogonal") or {"title": "o"})
    monkeypatch.setattr(director, "propose_crossover", lambda a, b: calls.append("crossover") or {"title": "c"})
    rng = random.Random(7)
    # empty pool: every exploit arm must FALL BACK to explore
    arms = {director._propose_via_arm(rng)[1] for _ in range(40)}
    assert arms == {"explore"}
    # 2 families in pool: all four arms reachable, arm label matches the call made
    elite.record(_outcome("Amihud illiquidity premium", "US small-cap equities", 1.0))
    elite.record(_outcome("Carry roll-yield premium", "futures", 0.8))
    calls.clear()
    labels = [director._propose_via_arm(rng)[1] for _ in range(200)]
    assert set(labels) == {"explore", "refine", "orthogonal", "crossover"}
    assert calls.count("crossover") == labels.count("crossover")


def test_propose_via_arm_parent_ids_lineage(pool, monkeypatch):
    """Exploit arms must record EXPLICIT parent_ids (research-map lineage); explore must not."""
    elite, p, _ = pool
    import agent.director as director
    monkeypatch.setattr(director, "elite", elite)
    monkeypatch.setattr(director, "propose", lambda: {"title": "x"})
    monkeypatch.setattr(director, "propose_mutate", lambda e: {"title": "m"})
    monkeypatch.setattr(director, "propose_orthogonal", lambda e: {"title": "o"})
    monkeypatch.setattr(director, "propose_crossover", lambda a, b: {"title": "c"})
    elite.record(_outcome("Amihud illiquidity premium", "US small-cap equities", 1.0))
    elite.record(_outcome("Carry roll-yield premium", "futures", 0.8))
    pool_ids = {it["id"] for it in elite._grid(elite._load()).values()}
    rng = random.Random(11)
    for _ in range(200):
        _, arm, parents = director._propose_via_arm(rng)
        if arm == "explore":
            assert parents == []
        elif arm in ("refine", "orthogonal"):
            assert len(parents) == 1 and set(parents) <= pool_ids
        else:  # crossover
            assert len(parents) == 2 and set(parents) <= pool_ids


def test_arm_reward_shape():
    # single-sourced in agent.bandit; run_worker re-exports it. MONOTONE in gate-progress (2026-06-25):
    # a no-edge SCREEN_FAIL (0.1) must rank below a cleared-screen run (0.3) that failed a later gate.
    from agent.run_worker import _arm_reward
    assert _arm_reward(None) == 0.0
    assert _arm_reward({"tier": "SCREEN_FAIL"}) == 0.1                              # no in-sample edge
    assert _arm_reward({"tier": "FAIL", "search_sharpe": 0.9}) == 0.3                # had an edge, failed later
    assert _arm_reward({"stage1_pass": True, "dsr": 0.8, "search_sharpe": 1.0}) == 1.3
    assert _arm_reward({"stage1_pass": True, "dsr": 0.2, "search_sharpe": 1.0, "PASSED_ALL_GATES": True}) == 2.0
