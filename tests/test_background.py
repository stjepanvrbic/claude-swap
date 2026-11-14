"""Tests for detached auto-switch background worker management."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_swap import background
from claude_swap.locking import InstanceLock
from claude_swap.paths import get_backup_root
from claude_swap.settings import load_settings, set_setting


HOLDER = """
import os, sys, time
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from claude_swap.locking import InstanceLock
lock = InstanceLock(Path(sys.argv[1]) / "autoswitch.lock")
assert lock.try_acquire()
print(os.getpid(), flush=True)
time.sleep(60)
"""


class Holder:
    """A lock-holding process. ``pid`` is the interpreter that holds the
    lock: on Windows a venv's python.exe is a launcher that runs it as a
    child, so the Popen pid is not it."""

    def __init__(self, proc: subprocess.Popen, pid: int):
        self.proc = proc
        self.pid = pid

    def kill(self) -> None:
        try:
            os.kill(self.pid, signal.SIGTERM)  # TerminateProcess on Windows
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def holder(tmp_path: Path):
    """A separate process holding the auto-switch lock, as a worker does."""
    src = Path(background.__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(tmp_path), str(src)],
        stdout=subprocess.PIPE, text=True,
    )
    holder = Holder(proc, int(proc.stdout.readline().strip()))
    yield holder
    holder.kill()


def _exited_proc() -> Mock:
    proc = Mock(pid=12345)
    proc.poll.return_value = 0
    return proc


def test_status_defaults_to_disabled(tmp_path: Path):
    st = background.status(tmp_path)

    assert st.enabled is False
    assert st.pid is None
    assert st.running is False
    assert st.pid_path == tmp_path / "autoswitch.pid"
    assert st.log_path == tmp_path / background.LOG_FILENAME


def test_status_reports_the_process_holding_the_lock(tmp_path: Path, holder):
    st = background.status(tmp_path)

    assert st.running is True
    assert st.pid == holder.pid


def test_a_second_holder_cannot_take_the_lock(tmp_path: Path, holder):
    assert InstanceLock(tmp_path / "autoswitch.lock").try_acquire() is False


def test_the_lock_frees_when_its_holder_dies(tmp_path: Path, holder):
    holder.kill()
    lock = InstanceLock(tmp_path / "autoswitch.lock")
    assert lock.try_acquire(timeout=10) is True
    lock.release()


def test_start_enables_and_launches_worker(tmp_path: Path):
    with patch("subprocess.Popen", return_value=_exited_proc()) as popen:
        background.start(tmp_path)

    assert load_settings(tmp_path).enabled is True
    popen.assert_called_once()
    args = popen.call_args.args[0]
    assert args[-3:] == ["claude_swap", "auto", "_worker"]


def test_start_does_not_launch_a_second_worker(tmp_path: Path, holder):
    with patch("subprocess.Popen") as popen:
        st = background.start(tmp_path)

    popen.assert_not_called()
    assert st.running is True
    assert st.pid == holder.pid


def test_start_on_windows_uses_hidden_console(tmp_path: Path):
    # DETACHED_PROCESS leaves no console to inherit, so the venv launcher's
    # child python allocates a visible one; closing it kills the worker.
    stopped = background.status(tmp_path)
    # sys.platform is one module attribute: patched here it reaches the lock
    # too, which cannot run Windows calls on this host.
    with patch("claude_swap.background.status", return_value=stopped), \
         patch("claude_swap.background.sys.platform", "win32"), \
         patch("subprocess.CREATE_NO_WINDOW", 0x08000000, create=True), \
         patch("subprocess.DETACHED_PROCESS", 0x00000008, create=True), \
         patch("subprocess.CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True), \
         patch("subprocess.Popen", return_value=_exited_proc()) as popen:
        background.start(tmp_path)

    flags = popen.call_args.kwargs["creationflags"]
    assert flags & 0x08000000
    assert not flags & 0x00000008


def test_stop_never_kills_a_holder(tmp_path: Path, holder):
    # A foreground `cswap auto` or the TUI holds the lock the same way and
    # may be mid-refresh; stop only disables, and says it is still running.
    st = background.stop(tmp_path, timeout=1)

    assert holder.proc.poll() is None
    assert st.running is True
    assert load_settings(tmp_path).enabled is False


@pytest.fixture
def worker_root(tmp_path: Path, monkeypatch):
    """A home of its own for real worker processes, and their backup root."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    with patch("pathlib.Path.home", return_value=home):
        root = get_backup_root()
    root.mkdir(parents=True, exist_ok=True)
    yield root
    background.stop(root)


def _wait_running(root: Path, timeout: float = 60.0) -> background.BackgroundStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = background.status(root)
        if st.running:
            return st
        time.sleep(0.2)
    return background.status(root)


def test_a_real_worker_runs_alone_and_stops_between_ticks(worker_root: Path):
    background.start(worker_root)
    st = _wait_running(worker_root)
    assert st.running and st.pid

    second = subprocess.run(
        [sys.executable, "-m", "claude_swap", "auto", "_worker"],
        capture_output=True, timeout=120,
    )
    assert second.returncode == 0
    assert background.status(worker_root).pid == st.pid

    stopped = background.stop(worker_root)
    assert stopped.running is False
    assert load_settings(worker_root).enabled is False


LEGACY_WORKER = """
import json, sys, time
from pathlib import Path
settings = Path(sys.argv[1]) / "settings.json"
print("up", flush=True)
for _ in range(600):
    try:
        if not json.loads(settings.read_text())["autoswitch"]["enabled"]:
            break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(0.2)
"""


def test_start_retires_a_worker_from_a_build_before_the_lock(tmp_path: Path):
    # The old worker holds no lock and leaves on reading enabled=false, as
    # the previous build's did; "claude_swap" in its command line marks it.
    set_setting(tmp_path, "autoswitch.enabled", "true")
    legacy = subprocess.Popen(
        [sys.executable, "-c", LEGACY_WORKER, str(tmp_path), "claude_swap"],
        stdout=subprocess.PIPE, text=True,
    )
    assert legacy.stdout.readline().strip() == "up"
    (tmp_path / background.PID_FILENAME).write_text(f"{legacy.pid}\n")
    real_popen = subprocess.Popen
    spawned = []

    def popen(args, **kwargs):
        # Only the worker launch is faked: on macOS the old-worker check
        # itself runs `ps` through subprocess.
        if args[-3:] == ["claude_swap", "auto", "_worker"]:
            spawned.append(args)
            return _exited_proc()
        return real_popen(args, **kwargs)

    try:
        with patch("subprocess.Popen", side_effect=popen):
            background.start(tmp_path)

        assert legacy.wait(timeout=30) is not None
        assert len(spawned) == 1
        assert load_settings(tmp_path).enabled is True
        assert not (tmp_path / background.PID_FILENAME).exists()
    finally:
        legacy.kill()
        legacy.wait()


def test_stop_with_nothing_running_disables(tmp_path: Path):
    (tmp_path / background.PID_FILENAME).write_text("12345\n")

    st = background.stop(tmp_path)

    assert load_settings(tmp_path).enabled is False
    assert st.running is False
    assert not (tmp_path / background.PID_FILENAME).exists()


def test_pause_preserves_enabled_worker_state(tmp_path: Path, holder):
    set_setting(tmp_path, "autoswitch.enabled", "true")

    st = background.pause(tmp_path)

    settings = load_settings(tmp_path)
    assert settings.enabled is True
    assert settings.paused is True
    assert st.running is True


def test_resume_restarts_enabled_worker(tmp_path: Path):
    set_setting(tmp_path, "autoswitch.enabled", "true")
    set_setting(tmp_path, "autoswitch.paused", "true")
    status = background.BackgroundStatus(
        enabled=True,
        pid=12345,
        running=True,
        pid_path=tmp_path / "autoswitch.pid",
        log_path=tmp_path / background.LOG_FILENAME,
        paused=False,
    )

    with patch("claude_swap.background.start", return_value=status) as start:
        st = background.resume(tmp_path)

    start.assert_called_once_with(tmp_path, debug=False)
    assert st is status
    assert load_settings(tmp_path).paused is False

def test_event_log_rotates_by_size(tmp_path: Path):
    with patch.object(background, "LOG_MAX_BYTES", 200):
        emit = background.event_logger(tmp_path)
        for i in range(20):
            emit.write(f"line {i:02d} " + "x" * 40)
    log = tmp_path / background.LOG_FILENAME
    rotated = sorted(tmp_path.glob(background.LOG_FILENAME + ".*"))
    assert log.stat().st_size <= 200
    assert len(rotated) == background.LOG_BACKUPS
    assert "line 19" in log.read_text()


def test_worker_output_goes_to_stderr_file_not_the_event_log(tmp_path: Path):
    with patch("subprocess.Popen", return_value=_exited_proc()) as popen:
        background.start(tmp_path)
    kwargs = popen.call_args.kwargs
    assert kwargs["stderr"].name == str(tmp_path / background.STDERR_FILENAME)
    assert kwargs["stdout"] is background.subprocess.DEVNULL
