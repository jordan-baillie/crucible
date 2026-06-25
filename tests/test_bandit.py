"""Thompson bandit over proposal arms (agent/bandit.py). Pure/deterministic — no LLM, no network.
Locks in: reward monotonicity, the 25% explore floor + no-arm-dies eps, the N_MIN fixed-split
fallback, and that the allocation is a valid distribution."""
import json
import random

from agent import bandit


def test_reward_is_monotone_in_gate_progress():
    R = bandit.arm_reward
    casualty = R(None)
    screen = R({"tier": "SCREEN_FAIL"})
    edge = R({"tier": "FAIL", "search_sharpe": 0.9})            # cleared tier-0 screen
    stage1 = R({"tier": "PROMOTE", "search_sharpe": 1.2, "stage1_pass": True, "dsr": 0.99})
    pass_all = R({"PASSED_ALL_GATES": True, "stage1_pass": True, "search_sharpe": 1.5, "dsr": 1.0})
    assert casualty == 0.0
    assert casualty < screen < edge < stage1 < pass_all        # strict gate-progress monotonicity
    assert screen == 0.1 and edge == 0.3 and pass_all == 2.0
    # a weak-but-present in-sample edge that fails a LATER gate still beats a no-edge screen-fail
    assert R({"tier": "FAIL", "search_sharpe": 0.4}) > R({"tier": "FAIL", "search_sharpe": 0.1})


def test_explore_floor_and_no_arm_dies():
    # a degenerate raw allocation that wants ~0 explore must still be floored at 25%
    w = bandit._apply_floors({"explore": 0.0, "refine": 0.9, "orthogonal": 0.05, "crossover": 0.05})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["explore"] >= bandit.EXPLORE_FLOOR - 1e-9
    assert all(v >= bandit.ARM_EPS - 1e-9 for k, v in w.items() if k != "explore")
    # a raw explore ABOVE the floor is preserved (floor is a minimum, not a cap)
    w2 = bandit._apply_floors({"explore": 0.6, "refine": 0.2, "orthogonal": 0.1, "crossover": 0.1})
    assert w2["explore"] >= 0.6 - 1e-9


def test_fixed_split_fallback_below_n_min(tmp_path):
    # too little data -> the pre-registered fixed split (fresh-machine safe), not a noisy bandit fit
    p = tmp_path / "rl.jsonl"
    p.write_text("\n".join(json.dumps({"arm": "explore", "verdict": {"tier": "FAIL", "search_sharpe": 0.9}})
                           for _ in range(bandit.N_MIN - 1)))
    assert bandit.arm_weights(run_log=str(p)) == bandit.FIXED_SPLIT


def test_bandit_favours_higher_reward_arm_over_floor():
    # enough data, refine clearly strong / explore clearly weak -> refine outweighs the floored explore
    rows = ([{"arm": "explore", "verdict": {"tier": "SCREEN_FAIL"}}] * 40 +
            [{"arm": "refine", "verdict": {"tier": "PROMOTE", "search_sharpe": 1.3,
                                           "stage1_pass": True, "dsr": 0.98}}] * 40)
    import tempfile, os
    fd, path = tempfile.mkstemp()
    os.write(fd, ("\n".join(json.dumps(r) for r in rows)).encode()); os.close(fd)
    try:
        w = dict(bandit.arm_weights(run_log=path, rng=random.Random(0)))
    finally:
        os.unlink(path)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["explore"] >= bandit.EXPLORE_FLOOR - 1e-9       # floor honoured
    assert w["refine"] > w["explore"]                        # strong arm beats the floored weak one


def test_arm_weights_order_matches_arms():
    w = bandit.arm_weights(rng=random.Random(0))
    assert tuple(a for a, _ in w) == bandit.ARMS            # stable order for cumulative sampling
