"""Detached auto-switch worker management.

``start``/``stop``/``pause``/``resume``/``status`` manage one worker process
per backup root (``cswap auto start`` launches ``cswap auto _worker``
detached). The worker holds the auto-switch lock while it runs, so a second
one exits at once and "running" is whether the lock is held. The worker writes its events through
:func:`event_logger`, a size-rotated log, and anything it prints to stderr
(a crash traceback) lands in a separate file.

    st = start(backup_root)       # BackgroundStatus(running=True, pid=...)
    emit = event_logger(backup_root)
    emit(event)                   # one timestamped line per engine event
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from claude_swap.locking import InstanceLock
from claude_swap.settings import load_settings, set_setting

PID_FILENAME = "autoswitch_background.pid"
LOG_FILENAME = "autoswitch_background.log"
STDERR_FILENAME = "autoswitch_background.err"
# A poll line every tick is ~2 MB a day at the 15 s minimum interval, so
# this keeps roughly a week of history.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
STDERR_MAX_BYTES = 1024 * 1024
# How long `start` waits to see the new worker take the lock.
START_WAIT_S = 5.0


@dataclass(frozen=True)
class BackgroundStatus:
    enabled: bool
    pid: int | None
    running: bool
    pid_path: Path
    log_path: Path
    paused: bool = False

    def to_json(self) -> dict:
        return {
            "enabled": self.enabled,
            "paused": self.paused,
            "pid": self.pid,
            "running": self.running,
            "pidPath": str(self.pid_path),
            "logPath": str(self.log_path),
        }


def pid_path(backup_root: Path) -> Path:
    return backup_root / PID_FILENAME


def log_path(backup_root: Path) -> Path:
    return backup_root / LOG_FILENAME


def stderr_path(backup_root: Path) -> Path:
    return backup_root / STDERR_FILENAME


class EventLog:
    """Engine-event sink for the worker: one timestamped line per event in
    the rotated worker log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        self._logger = logging.getLogger(f"claude-swap-worker.{path}")
        self._logger.handlers[:] = [handler]
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def __call__(self, event) -> None:
        self.write(event.human())

    def write(self, line: str) -> None:
        self._logger.info(line)


def event_logger(backup_root: Path) -> EventLog:
    return EventLog(log_path(backup_root))


def _lock(backup_root: Path) -> InstanceLock:
    # Imported here: autoswitch imports this module's settings helpers.
    from claude_swap.autoswitch import LOCK_FILENAME

    return InstanceLock(backup_root / LOCK_FILENAME)


def _remove_legacy_pid(backup_root: Path) -> None:
    try:
        (backup_root / PID_FILENAME).unlink()
    except OSError:
        pass


def status(backup_root: Path) -> BackgroundStatus:
    """Running means some process holds the auto-switch lock: a worker, a
    foreground ``cswap auto``, or a TUI/menu bar engine."""
    lock = _lock(backup_root)
    settings = load_settings(backup_root)
    running = lock.held_elsewhere()
    return BackgroundStatus(
        enabled=settings.enabled,
        paused=settings.paused,
        pid=lock.holder_pid() if running else None,
        running=running,
        pid_path=lock.pid_path,
        log_path=log_path(backup_root),
    )


def start(backup_root: Path, *, debug: bool = False) -> BackgroundStatus:
    current = status(backup_root)
    set_setting(backup_root, "autoswitch.enabled", "true")
    if current.running:
        return status(backup_root)
    _remove_legacy_pid(backup_root)

    err = stderr_path(backup_root)
    err.parent.mkdir(parents=True, exist_ok=True)
    try:
        err_mode = "wb" if err.stat().st_size > STDERR_MAX_BYTES else "ab"
    except FileNotFoundError:
        err_mode = "ab"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    args = [sys.executable, "-m", "claude_swap", "auto", "_worker"]
    if debug:
        args.append("--debug")

    creationflags = 0
    popen_kwargs = {}
    if sys.platform == "win32":
        # A hidden console, not DETACHED_PROCESS: the venv launcher's child
        # python would otherwise open a visible console that kills it on close.
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    with open(err, err_mode, buffering=0) as err_file:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            close_fds=True,
            creationflags=creationflags,
            env=env,
            **popen_kwargs,
        )
    # Report what actually happened: the worker holds the lock once its
    # engine starts, or has exited (another engine won the race, or it
    # failed and said why in the error log).
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        st = status(backup_root)
        if st.running or proc.poll() is not None:
            return st
        time.sleep(0.1)
    return status(backup_root)


def stop(
    backup_root: Path,
    *,
    timeout: float = 15.0,
    persist: bool = True,
) -> BackgroundStatus:
    """Stop whatever holds the auto-switch lock: SIGTERM, which ends the
    loop between ticks, then a kill if it is still there at ``timeout``."""
    if persist:
        set_setting(backup_root, "autoswitch.enabled", "false")
    _remove_legacy_pid(backup_root)
    current = status(backup_root)
    if not current.running or current.pid is None:
        return current
    try:
        os.kill(current.pid, signal.SIGTERM)
    except OSError:
        return status(backup_root)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not status(backup_root).running:
            return status(backup_root)
        time.sleep(0.1)
    if sys.platform != "win32":
        try:
            os.kill(current.pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.5)
    return status(backup_root)


def pause(backup_root: Path) -> BackgroundStatus:
    """Pause switching without stopping a process manager's worker."""
    set_setting(backup_root, "autoswitch.paused", "true")
    return status(backup_root)


def resume(backup_root: Path, *, debug: bool = False) -> BackgroundStatus:
    """Resume switching and restore a previously enabled worker."""
    was_enabled = load_settings(backup_root).enabled
    set_setting(backup_root, "autoswitch.paused", "false")
    if was_enabled:
        return start(backup_root, debug=debug)
    return status(backup_root)
