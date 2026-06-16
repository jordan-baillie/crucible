"""Rails hardening (operator 2026-06-16):
  #1 write-once holdout commit — atomic, locked, FAIL-CLOSED.
  #2 CPCV purge wired to the strategy's holding horizon (not a fixed 1 bar).
Both are STRICTER-only changes; these tests pin the new guarantees."""
import numpy as np
import pytest

import sdk.harness as H


class _NoLock:
    """No-op stand-in for FileLock so the commit logic is testable without a real cross-process lock."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ───────────────────────── #1 write-once holdout: FAIL-CLOSED + race-safe ─────────────────────────
def test_holdout_commit_success_records_and_passes(monkeypatch):
    monkeypatch.setattr(H, "FileLock", _NoLock)
    appended = []
    monkeypatch.setattr(H.ri, "ledger_lookup", lambda h: None)
    monkeypatch.setattr(H.ri, "ledger_append", lambda rec: appended.append(rec))
    hp, reasons, raced = H._commit_holdout_look("s1", "hash1", {"config_hash": "hash1"}, True, [])
    assert hp is True and reasons == [] and raced is False
    assert appended == [{"config_hash": "hash1"}]            # the single look IS recorded


def test_holdout_commit_FAILS_CLOSED_when_append_errors(monkeypatch):
    # the crux: a holdout that cannot record its single use must NOT hand out a PASS
    monkeypatch.setattr(H, "FileLock", _NoLock)
    monkeypatch.setattr(H.ri, "ledger_lookup", lambda h: None)
    def boom(rec): raise IOError("disk full")
    monkeypatch.setattr(H.ri, "ledger_append", boom)
    hp, reasons, raced = H._commit_holdout_look("s1", "hash1", {}, True, [])   # was passing
    assert hp is False                                       # FORCED FAIL (was: WARN + keep PASS)
    assert any("UNRECORDABLE" in r for r in reasons)


def test_holdout_commit_FAILS_CLOSED_on_concurrent_race(monkeypatch):
    # a peer claimed the same config_hash while we computed -> re-lookup inside the lock catches it
    monkeypatch.setattr(H, "FileLock", _NoLock)
    monkeypatch.setattr(H.ri, "ledger_lookup", lambda h: {"ts": "earlier"})
    def must_not_append(rec): raise AssertionError("must not append over a concurrent claim")
    monkeypatch.setattr(H.ri, "ledger_append", must_not_append)
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
