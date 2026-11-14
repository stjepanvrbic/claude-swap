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

    The lock is the operating system's, so it ends with the process whatever
    way it ends; nothing has to clean up after a crash. The holder's pid is
    written next to it for ``holder_pid``.

        lock = InstanceLock(root / "autoswitch.lock")
        if not lock.try_acquire():
            print("already running, pid", lock.holder_pid())
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.pid_path = lock_path.with_suffix(".pid")
        self._fd: int | None = None

    def try_acquire(self) -> bool:
        """Take the lock without waiting; False when another process has it."""
        if self._fd is not None:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if sys.platform == "win32":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        self.pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            self.pid_path.unlink()
        except OSError:
            pass
        if sys.platform == "win32":
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def held_elsewhere(self) -> bool:
        """Whether another process holds the lock right now."""
        if self._fd is not None:
            return False
        if not self.try_acquire():
            return True
        self.release()
        return False

    def holder_pid(self) -> int | None:
        try:
            pid = int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return pid if pid > 0 else None
