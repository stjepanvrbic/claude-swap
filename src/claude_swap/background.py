"""Detached auto-switch worker management."""

from __future__ import annotations

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


@dataclass(frozen=True)
class BackgroundStatus:
    enabled: bool
    pid: int | None
    running: bool
    pid_path: Path
    log_path: Path

    def to_json(self) -> dict:
        return {
            "enabled": self.enabled,
            "pid": self.pid,
            "running": self.running,
            "pidPath": str(self.pid_path),
            "logPath": str(self.log_path),
        }


def pid_path(backup_root: Path) -> Path:
    return backup_root / PID_FILENAME


def log_path(backup_root: Path) -> Path:
    return backup_root / LOG_FILENAME


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
    pid = _read_pid(path)
    return BackgroundStatus(
        enabled=load_settings(backup_root).enabled,
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

    log = log_path(backup_root)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    args = [sys.executable, "-m", "claude_swap", "auto", "_worker"]
    if debug:
        args.append("--debug")

    creationflags = 0
    popen_kwargs = {}
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    with open(log, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
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
