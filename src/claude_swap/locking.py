"""File locking for concurrent access protection."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import IO

# Platform-specific imports for file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from claude_swap.exceptions import LockError


class FileLock:
    """Cross-process file lock using platform-specific APIs."""

    def __init__(self, lock_path: Path, timeout: float = 10.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_file: IO | None = None
        self._locked = False

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire exclusive lock with timeout.

        Args:
            timeout: Maximum seconds to wait for lock. Defaults to the
                timeout given at construction.

        Returns:
            True if lock acquired, False if timeout.
        """
        if timeout is None:
            timeout = self.timeout
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "w")

        start = time.monotonic()
        while True:
            try:
                if sys.platform == "win32":
                    # Windows: use msvcrt for file locking
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # POSIX: use fcntl for file locking
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
                return True
            except (BlockingIOError, OSError):
                if time.monotonic() - start > timeout:
                    self._lock_file.close()
                    self._lock_file = None
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        """Release the lock."""
        if self._lock_file and self._locked:
            if sys.platform == "win32":
                # Windows: unlock using msvcrt
                try:
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # File may already be unlocked
            else:
                # POSIX: unlock using fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            self._locked = False

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise LockError("Failed to acquire lock - another instance may be running")
        return self

    def __exit__(self, *args) -> None:
        self.release()


class InstanceLock:
    """Held for a process's lifetime so only one instance of something runs.

    The lock is the operating system's, so it ends with the process however
    it ends. The holder writes its pid to ``pid_path``. Checking whether the
    lock is held takes it for an instant without writing a pid, so a holder
    starting at that moment retries briefly (``try_acquire(timeout=...)``).

        lock = InstanceLock(root / "autoswitch.lock", root / "worker.pid")
        if not lock.try_acquire(timeout=3):
            print("already running")
    """

    def __init__(self, lock_path: Path, pid_path: Path):
        self.lock_path = lock_path
        self.pid_path = pid_path
        self._fd: int | None = None

    def _take(self) -> int | None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if sys.platform == "win32":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _drop(fd: int) -> None:
        if sys.platform == "win32":
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def try_acquire(self, timeout: float = 0.0) -> bool:
        """Take the lock, retrying for up to ``timeout`` seconds."""
        if self._fd is not None:
            return True
        deadline = time.monotonic() + timeout
        while True:
            fd = self._take()
            if fd is not None:
                self._fd = fd
                self.pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if self.pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.pid_path.unlink()
        except (OSError, ValueError):
            pass
        self._drop(self._fd)
        self._fd = None

    def held_elsewhere(self) -> bool:
        """Whether another holder has the lock right now."""
        if self._fd is not None:
            return False
        fd = self._take()
        if fd is None:
            return True
        self._drop(fd)
        return False
