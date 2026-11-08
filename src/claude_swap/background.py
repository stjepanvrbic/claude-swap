"""Detached auto-switch worker management.

``start``/``stop``/``pause``/``resume``/``status`` manage one worker process
per backup root (``cswap auto start`` launches ``cswap auto _worker``
detached), recorded in a pid file. The worker writes its events through
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

from claude_swap.settings import load_settings, set_setting

PID_FILENAME = "autoswitch_background.pid"
LOG_FILENAME = "autoswitch_background.log"
STDERR_FILENAME = "autoswitch_background.err"
# A poll line every tick is ~2 MB a day at the 15 s minimum interval, so
# this keeps roughly a week of history.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
STDERR_MAX_BYTES = 1024 * 1024


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


def _read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(path, 0o600)


def _remove_pid(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _process_running(pid: int | None) -> bool:
    if pid is None:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            code = ctypes.c_ulong()
            try:
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == 259
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(backup_root: Path) -> BackgroundStatus:
    path = pid_path(backup_root)
    settings = load_settings(backup_root)
    pid = _read_pid(path)
    return BackgroundStatus(
        enabled=settings.enabled,
        paused=settings.paused,
        pid=pid,
        running=_process_running(pid),
        pid_path=path,
        log_path=log_path(backup_root),
    )


def start(backup_root: Path, *, debug: bool = False) -> BackgroundStatus:
    current = status(backup_root)
    set_setting(backup_root, "autoswitch.enabled", "true")
    if current.running:
        return status(backup_root)
    _remove_pid(current.pid_path)

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
    _write_pid(current.pid_path, proc.pid)
    return status(backup_root)


def stop(
    backup_root: Path,
    *,
    timeout: float = 5.0,
    persist: bool = True,
) -> BackgroundStatus:
    if persist:
        set_setting(backup_root, "autoswitch.enabled", "false")
    current = status(backup_root)
    if not current.running or current.pid is None:
        _remove_pid(current.pid_path)
        return status(backup_root)

    try:
        os.kill(current.pid, signal.SIGTERM)
    except OSError:
        _remove_pid(current.pid_path)
        return status(backup_root)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_running(current.pid):
            _remove_pid(current.pid_path)
            return status(backup_root)
        time.sleep(0.1)
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
