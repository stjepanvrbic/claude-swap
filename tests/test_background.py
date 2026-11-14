"""Tests for detached auto-switch background worker management."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_swap import background
from claude_swap.locking import InstanceLock
from claude_swap.settings import load_settings, set_setting


HOLDER = """
import sys, time
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from claude_swap.locking import InstanceLock
lock = InstanceLock(Path(sys.argv[1]) / "autoswitch.lock")
assert lock.try_acquire()
print("held", flush=True)
time.sleep(60)
"""


@pytest.fixture
def holder(tmp_path: Path):
    """A separate process holding the auto-switch lock, as a worker does."""
    src = Path(background.__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(tmp_path), str(src)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    yield proc
    proc.kill()
    proc.wait()


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
    holder.wait()
    lock = InstanceLock(tmp_path / "autoswitch.lock")
    assert lock.try_acquire() is True
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


def test_stop_ends_the_lock_holder_and_disables(tmp_path: Path, holder):
    st = background.stop(tmp_path, timeout=10)

    assert holder.wait(timeout=10) is not None
    assert st.running is False
    assert load_settings(tmp_path).enabled is False


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
