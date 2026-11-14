"""Auto-switch worker management.

``start``/``stop``/``pause``/``resume``/``status`` manage one worker process
per backup root, ``cswap auto _worker``, recorded in a pid file. ``start``
hands it to the platform's service manager (``claude_swap.service``), which
starts it at login and again when it dies, and launches it detached where
there is none. The worker writes its events through :func:`event_logger`, a
size-rotated log, and anything it prints to stderr (a crash traceback) lands
in a separate file.

    st = start(backup_root)       # BackgroundStatus(running=True, pid=..., supervisor="systemd")
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
from dataclasses import dataclass, replace
from pathlib import Path

from claude_swap import service
from claude_swap.locking import InstanceLock
from claude_swap.settings import load_settings, set_setting

PID_FILENAME = "autoswitch_background.pid"
LOCK_FILENAME = "autoswitch_background.lock"
LOG_FILENAME = "autoswitch_background.log"
STDERR_FILENAME = "autoswitch_background.err"
# Written by a worker a service manager runs: its pid and the manager's name.
SUPERVISOR_FILENAME = "autoswitch_background.service"
# A poll line every tick is ~2 MB a day at the 15 s minimum interval, so
# this keeps roughly a week of history.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
STDERR_MAX_BYTES = 1024 * 1024
# How long `start` waits to see the new worker take the lock (a loaded host
# or a Windows venv launcher can take seconds).
START_WAIT_S = 15.0
# How long `start` lets a worker the service did not start finish its tick.
HANDOVER_WAIT_S = 60.0


@dataclass(frozen=True)
class BackgroundStatus:
    enabled: bool
    pid: int | None
    running: bool
    pid_path: Path
    log_path: Path
    paused: bool = False
    # The service manager that runs the worker ("systemd", "task-scheduler"),
    # or None when it runs detached.
    supervisor: str | None = None
    # What `start` could not do as it should, for the user to read.
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict:
        out = {
            "enabled": self.enabled,
            "paused": self.paused,
            "pid": self.pid,
            "running": self.running,
            "supervisor": self.supervisor,
            "pidPath": str(self.pid_path),
            "logPath": str(self.log_path),
        }
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


def pid_path(backup_root: Path) -> Path:
    return backup_root / PID_FILENAME


def worker_lock(backup_root: Path) -> InstanceLock:
    """Held by the running worker for its lifetime; it writes its own pid to
    the pid file, so the file names the worker that actually runs."""
    return InstanceLock(backup_root / LOCK_FILENAME, pid_path(backup_root))


def supervisor_path(backup_root: Path) -> Path:
    return backup_root / SUPERVISOR_FILENAME


def mark_supervised(backup_root: Path, manager: str) -> None:
    """Record that this process is the worker ``manager`` runs."""
    supervisor_path(backup_root).write_text(f"{os.getpid()} {manager}\n", encoding="utf-8")


def clear_supervised(backup_root: Path) -> None:
    path = supervisor_path(backup_root)
    try:
        pid, _manager = path.read_text(encoding="utf-8").split()
        if int(pid) == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def _supervisor(backup_root: Path, pid: int | None) -> str | None:
    try:
        marked_pid, manager = supervisor_path(backup_root).read_text(encoding="utf-8").split()
    except (OSError, ValueError):
        return None
    return manager if pid is not None and marked_pid == str(pid) else None


def open_stderr(backup_root: Path):
    """The worker's stderr file, appended to, emptied once it passes
    ``STDERR_MAX_BYTES``."""
    err = stderr_path(backup_root)
    err.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = "wb" if err.stat().st_size > STDERR_MAX_BYTES else "ab"
    except FileNotFoundError:
        mode = "ab"
    return open(err, mode, buffering=0)


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
    running = worker_lock(backup_root).held_elsewhere() or _process_running(pid)
    return BackgroundStatus(
        enabled=settings.enabled,
        paused=settings.paused,
        pid=pid,
        running=running,
        pid_path=path,
        log_path=log_path(backup_root),
        supervisor=_supervisor(backup_root, pid) if running else None,
    )


def start(backup_root: Path, *, debug: bool = False) -> BackgroundStatus:
    """Enable auto-switch and leave its worker running, under the service
    manager when there is one. ``debug`` launches a detached worker with
    debug logging when none runs."""
    current = status(backup_root)
    set_setting(backup_root, "autoswitch.enabled", "true")
    warnings: list[str] = []
    manager = None
    if not debug:
        try:
            manager = service.manager(backup_root)
        except service.Unavailable as e:
            if str(e):
                warnings.append(
                    f"No service manager runs the worker here ({e}): it runs "
                    "detached, and nothing restarts it if it dies."
                )
    # The pid file names a worker only while its lock is held: a killed
    # worker's pid can belong to another process by now.
    held = worker_lock(backup_root).held_elsewhere()
    if held and (manager is None or current.supervisor == manager.name):
        return replace(status(backup_root), warnings=tuple(warnings))

    if manager is not None:
        if held:
            # A worker the service manager did not start: it gives way to
            # the service's, which comes back if it dies.
            if not _end_worker(backup_root, current.pid):
                warnings.append(
                    f"The running worker (pid {current.pid}) did not stop within "
                    f"{HANDOVER_WAIT_S:.0f}s; the service's worker takes over "
                    "when it ends."
                )
        try:
            manager.install()
            manager.activate()
        except service.ServiceError as e:
            warnings.append(
                f"The {manager.describe()} did not start ({e}): the worker runs "
                "detached, and nothing restarts it if it dies."
            )
        else:
            try:
                warnings.extend(manager.warnings())
            except service.ServiceError:
                pass
            if wait_for_worker(backup_root, START_WAIT_S):
                return replace(status(backup_root), warnings=tuple(warnings))
            warnings.append(
                f"The {manager.describe()} has not started the worker within "
                f"{START_WAIT_S:.0f}s, so it runs detached for now; the service "
                "takes over when that one ends."
            )
    return _launch_detached(backup_root, debug, warnings)


def wait_for_worker(backup_root: Path, timeout: float) -> bool:
    """Whether a worker holds the lock within ``timeout`` seconds."""
    lock = worker_lock(backup_root)
    deadline = time.monotonic() + timeout
    while not lock.held_elsewhere():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _end_worker(backup_root: Path, pid: int | None) -> bool:
    """Stop the worker holding the lock; whether its lock freed in time."""
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    lock = worker_lock(backup_root)
    deadline = time.monotonic() + HANDOVER_WAIT_S
    while lock.held_elsewhere():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)
    return True


def _launch_detached(
    backup_root: Path, debug: bool, warnings: list[str]
) -> BackgroundStatus:
    _remove_pid(pid_path(backup_root))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    args = [sys.executable, "-m", "claude_swap", "auto", "_worker"]
    if debug:
        args.append("--debug")

    launch_flags = [0]
    popen_kwargs = {}
    if sys.platform == "win32":
        # A hidden console, not DETACHED_PROCESS: the venv launcher's child
        # python would otherwise open a visible console that kills it on close.
        hidden = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        # Out of the caller's job object as well: an SSH session runs in a
        # job that kills every process in it when the session closes. A job
        # that forbids breakaway refuses the launch, and the worker then
        # lives only as long as its caller.
        launch_flags = [
            hidden | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0),
            hidden,
        ]
    else:
        popen_kwargs["start_new_session"] = True

    with open_stderr(backup_root) as err_file:
        for attempt, creationflags in enumerate(launch_flags, start=1):
            try:
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
                break
            except PermissionError:
                if attempt == len(launch_flags):
                    raise
                warnings.append(
                    "This session's job object keeps the worker, so it stops "
                    "when this session ends."
                )
    # The worker writes its own pid once it holds the lock: the launched
    # process may be a launcher (a Windows venv's python.exe), or a duplicate
    # that leaves at once. Report what happened rather than what was spawned.
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        if worker_lock(backup_root).held_elsewhere() or proc.poll() is not None:
            break
        time.sleep(0.1)
    return replace(status(backup_root), warnings=tuple(warnings))


def stop(
    backup_root: Path,
    *,
    timeout: float = 30.0,
    persist: bool = True,
) -> BackgroundStatus:
    """Stop the worker and keep the service manager from starting it again;
    ``persist`` also turns auto-switch off. A worker in the middle of a check
    finishes it first, so this waits up to ``timeout`` seconds."""
    if persist:
        set_setting(backup_root, "autoswitch.enabled", "false")
    try:
        manager = service.manager(backup_root)
    except service.Unavailable:
        manager = None
    warnings: list[str] = []
    try:
        if manager is not None and manager.installed():
            manager.deactivate()
    except service.ServiceError as e:
        warnings.append(f"The {manager.describe()} could not be disabled ({e}).")
    current = status(backup_root)
    # Only the lock's holder owns the pid file's pid: a killed worker's pid
    # can belong to another process by now.
    if current.pid is None or not worker_lock(backup_root).held_elsewhere():
        _remove_pid(current.pid_path)
        return replace(status(backup_root), warnings=tuple(warnings))

    try:
        os.kill(current.pid, signal.SIGTERM)
    except OSError:
        _remove_pid(current.pid_path)
        return replace(status(backup_root), warnings=tuple(warnings))

    # Stopped means the lock is free, not just the process gone: Windows
    # reports a terminated process exited a moment before it closes the
    # process's handles, the lock file's among them. The pid file goes as
    # soon as the process does, since its pid can be reused.
    lock = worker_lock(backup_root)
    deadline = time.monotonic() + timeout
    while True:
        if not _process_running(current.pid):
            _remove_pid(current.pid_path)
            if not lock.held_elsewhere():
                break
        if time.monotonic() >= deadline:
            warnings.append(
                f"The worker (pid {current.pid}) was asked to stop but still "
                f"holds its lock after {timeout:.0f}s."
            )
            break
        time.sleep(0.1)
    return replace(status(backup_root), warnings=tuple(warnings))


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
