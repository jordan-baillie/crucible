"""Rails hardening:
  #1 write-once holdout commit — atomic, FAIL-CLOSED, and (2026-06-22) INTRINSIC to the rail:
     the harness delegates the lock + re-check + append to ri.ledger_commit_once, so the
     single-use guarantee no longer depends on this caller holding a lock. These tests pin the
     harness's FAIL-CLOSED MAPPING at that seam; the rail's own atomicity is covered by
     test_harness_integration::test_ledger_commit_once_is_atomic_and_intrinsic.
  #2 CPCV purge wired to the strategy's holding horizon (not a fixed 1 bar).
Both are STRICTER-only changes."""
import numpy as np
import pytest

import sdk.harness as H


# ───────────────────────── #1 write-once holdout: FAIL-CLOSED + race-safe ─────────────────────────
def test_holdout_commit_success_records_and_passes(monkeypatch):
    # rail booked the single look (prior is None) -> the PASS stands, the look is recorded.
    seen = {}
    def commit_once(h, rec):
        seen["h"], seen["rec"] = h, rec
        return None
    monkeypatch.setattr(H.ri, "ledger_commit_once", commit_once)
    hp, reasons, raced = H._commit_holdout_look("s1", "hash1", {"config_hash": "hash1"}, True, [])
    assert hp is True and reasons == [] and raced is False
    assert seen == {"h": "hash1", "rec": {"config_hash": "hash1"}}   # the single look IS recorded


def test_holdout_commit_FAILS_CLOSED_when_append_errors(monkeypatch):
    # the crux: a holdout that cannot record its single use must NOT hand out a PASS
    def boom(h, rec): raise IOError("disk full")
    monkeypatch.setattr(H.ri, "ledger_commit_once", boom)
    hp, reasons, raced = H._commit_holdout_look("s1", "hash1", {}, True, [])   # was passing
    assert hp is False and raced is False                   # FORCED FAIL (was: WARN + keep PASS)
    assert any("UNRECORDABLE" in r for r in reasons)


def test_holdout_commit_FAILS_CLOSED_on_concurrent_race(monkeypatch):
    # a peer booked the same config_hash first -> the rail returns the PRIOR record -> force FAIL
    monkeypatch.setattr(H.ri, "ledger_commit_once", lambda h, rec: {"ts": "earlier"})
    hp, reasons, raced = H._commit_holdout_look("s1", "hash1", {}, True, [])
    assert hp is False and raced is True
    assert any("RACE" in r for r in reasons)


# ───────────────────────── #2 CPCV purge = holding horizon ─────────────────────────
def test_holding_horizon_bars():
    assert H._holding_horizon_bars([]) == 1                          # no trades -> 1
    assert H._holding_horizon_bars([{"pnl": 1}]) == 1               # no dates -> 1
    one_day = [{"entry_date": "2020-01-01", "exit_date": "2020-01-02"}] * 5
    assert H._holding_horizon_bars(one_day) == 1                    # 1-bar holds -> 1 (back-compat)
    weekly = [{"entry_date": "2020-01-01", "exit_date": "2020-01-08"}] * 5   # 7 calendar days
    assert H._holding_horizon_bars(weekly) == 5                     # ceil(7 * 5/7) = 5 trading bars


def test_cpcv_splits_honor_purge():
    from research_integrity import cpcv
    n = 120
    s1 = cpcv.cpcv_splits(n, n_groups=6, k_test=2, purge=1)
    s5 = cpcv.cpcv_splits(n, n_groups=6, k_test=2, purge=5)
    assert len(s1) == len(s5)
    # a larger purge removes MORE training rows around the test blocks (less label-overlap leakage)
    assert s5[0].train_idx.size < s1[0].train_idx.size


def test_cpcv_path_sharpes_and_assemble_bundle_accept_purge():
    import research_integrity as ri
    r = np.random.default_rng(0).normal(0.0005, 0.01, 400)
    # both entry points must accept the purge kwarg and run
    paths = ri.assemble_bundle(r, [], purge=5)
    assert "bundle" in paths
    from research_integrity.adapter import cpcv_path_sharpes
    assert isinstance(cpcv_path_sharpes(r, purge=5), list)
