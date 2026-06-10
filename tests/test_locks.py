"""FileLock smoke tests incl. the rename-based stale-steal (TOCTOU fix)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fresh_locks(monkeypatch, tmp_path):
    monkeypatch.setenv("CRUCIBLE_LOCK_DIR", str(tmp_path / "locks"))
    for m in list(sys.modules):
        if m.startswith(("sdk", "crucible_paths")):
            sys.modules.pop(m)
    from sdk.locks import FileLock, LockTimeout
    return FileLock, LockTimeout


def test_acquire_release(monkeypatch, tmp_path):
    FileLock, LockTimeout = _fresh_locks(monkeypatch, tmp_path)
    with FileLock("t1", ttl=5) as l:
        assert l.path.exists()
    assert not l.path.exists()


def test_contention_blocks_then_acquires(monkeypatch, tmp_path):
    FileLock, LockTimeout = _fresh_locks(monkeypatch, tmp_path)
    a = FileLock("t2", ttl=5).acquire()
    b = FileLock("t2", ttl=5, wait=0.2)
    t0 = time.time()
    try:
        b.acquire()
        raise AssertionError("second holder must not acquire a live lock")
    except LockTimeout:
        assert time.time() - t0 >= 0.15
    a.release()
    b.wait = 1
    b.acquire(); b.release()


def test_stale_lock_is_stolen_not_live(monkeypatch, tmp_path):
    FileLock, LockTimeout = _fresh_locks(monkeypatch, tmp_path)
    # plant an EXPIRED lock
    stale = FileLock("t3", ttl=5)
    stale.path.parent.mkdir(parents=True, exist_ok=True)
    stale.path.write_text(json.dumps({"owner": "dead", "acquired": 0, "expires": 1}))
    got = FileLock("t3", ttl=5, wait=2).acquire()   # must steal
    assert got.path.exists()
    got.release()
    # a LIVE lock must survive a steal attempt (rename-then-inspect restores it)
    live = FileLock("t4", ttl=60).acquire()
    thief = FileLock("t4", ttl=60, wait=0.2)
    try:
        thief.acquire()
        raise AssertionError("live lock must not be stolen")
    except LockTimeout:
        pass
    assert live.path.exists(), "live lock must be restored after failed steal"
    live.release()
