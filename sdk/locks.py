"""Cross-process file locks for the multi-agent forge.

Atomic O_CREAT|O_EXCL create + owner/expiry + stale-steal. Zero deps. Used to serialize
the shared FDR registry read-modify-write and the work-queue claim across N agents.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from crucible_paths import LOCKS
LOCK_DIR = Path(os.environ.get("CRUCIBLE_LOCK_DIR", os.environ.get("HEPH_LOCK_DIR", LOCKS)))


def _me() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class LockTimeout(Exception):
    pass


class FileLock:
    """Context-manager file lock. Reentrancy NOT supported (don't nest the same name)."""

    def __init__(self, name: str, ttl: float = 120.0, owner: str | None = None,
                 wait: float = 120.0, poll: float = 0.05):
        self.name = name
        self.ttl = ttl
        self.owner = owner or _me()
        self.wait = wait
        self.poll = poll
        self.path = LOCK_DIR / f"{name}.lock"
        self._held = False

    def _payload(self) -> bytes:
        now = time.time()
        return json.dumps({"owner": self.owner, "acquired": now, "expires": now + self.ttl}).encode()

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, self._payload())
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False

    def _steal_if_stale(self) -> None:
        """Steal an EXPIRED lock without the read->check->unlink TOCTOU.

        Naive unlink races: between B reading an expired payload and unlinking, holder A can
        release and C can acquire -- B then deletes C's LIVE lock and two holders coexist.
        Fix: atomically RENAME the lock to a unique name first (only one stealer wins the
        rename), then inspect the renamed file and delete it only if it really was expired
        (if it was live after all, restore it)."""
        claim = self.path.with_suffix(f".steal.{os.getpid()}.{time.monotonic_ns()}")
        try:
            os.rename(self.path, claim)  # atomic: exactly one stealer can win
        except FileNotFoundError:
            return  # released (or stolen) meanwhile
        try:
            data = json.loads(claim.read_text(encoding="utf-8"))
            expired = time.time() > float(data.get("expires", 0))
        except ValueError:
            expired = True  # corrupt payload -> treat as dead
        if expired:
            claim.unlink(missing_ok=True)
        else:
            # live lock grabbed by mistake -- put it back (acquire() will keep waiting)
            try:
                os.rename(claim, self.path)
            except OSError:
                claim.unlink(missing_ok=True)

    def acquire(self) -> "FileLock":
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.wait
        while True:
            if self._try_create():
                self._held = True
                return self
            self._steal_if_stale()
            if self._try_create():
                self._held = True
                return self
            if time.time() >= deadline:
                raise LockTimeout(f"could not acquire lock '{self.name}' within {self.wait}s")
            time.sleep(self.poll)

    def release(self) -> None:
        if not self._held:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("owner") == self.owner:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass
        self._held = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
