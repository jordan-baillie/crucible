"""Joint crucible+atlas snapshot + diversity brief (agent/joint_state.py). Pure — no LLM, no network.

Locks: build() is graceful and well-shaped; brief() is generator-only, bounded, names the spent/live
families, and returns "" (NO steer) when there is nothing to steer against — exactly like _focus().
"""
from agent import joint_state


def test_build_is_graceful_and_well_shaped():
    # Reads wiki/elite/atlas which may be absent in CI — must never raise, always the same shape.
    snap = joint_state.build()
    assert snap["schema_version"] == joint_state.SCHEMA_VERSION
    assert set(snap["research"].keys()) == {"fdr", "elite_cells", "closed_families"}
    assert isinstance(snap["execution"], list)


def test_brief_is_empty_when_nothing_to_steer():
    empty = {"research": {"fdr": {}, "elite_cells": [], "closed_families": []}, "execution": []}
    assert joint_state.brief(empty) == ""      # no steer == safe default (unset-focus behaviour)


def test_brief_names_closed_live_and_exploited_families():
    snap = {
        "research": {
            "fdr": {"bar": 0.9, "n_families": 3, "families": ["pead"]},
            "elite_cells": [{"family": "amihud_illiq", "universe": "us_small",
                             "turnover": "med", "fitness": 0.7}],
            "closed_families": ["value_x_mom", "crypto_basis_carry"],
        },
        "execution": [{"name": "val_mom_trend_smallcap", "state": "evidence",
                       "family": "val_mom", "days": 20, "cum_return_pct": 1.3}],
    }
    b = joint_state.brief(snap)
    assert "DIVERSITY BRIEF" in b
    assert "value_x_mom" in b and "crypto_basis_carry" in b     # closed
    assert "val_mom" in b                                        # live book family
    assert "amihud_illiq" in b                                   # heavily-exploited elite cell
    assert "ORTHOGONAL" in b                                     # the actual steer
    assert len(b) < 2000                                         # bounded — cannot bloat the prompt


def test_brief_is_bounded_under_many_families():
    snap = {
        "research": {"fdr": {}, "closed_families": [f"closed_fam_{i}" for i in range(50)],
                     "elite_cells": [{"family": f"elite_fam_{i}"} for i in range(50)]},
        "execution": [{"name": f"book_{i}", "family": f"live_fam_{i}"} for i in range(50)],
    }
    b = joint_state.brief(snap)
    # each list is capped at _MAX_LIST — a huge estate can't push the brief toward the call timeout
    assert b.count("closed_fam_") <= joint_state._MAX_LIST
    assert b.count("live_fam_") <= joint_state._MAX_LIST
    assert b.count("elite_fam_") <= joint_state._MAX_LIST
