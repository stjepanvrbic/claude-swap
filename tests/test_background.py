"""Tests for detached auto-switch background worker management."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_swap import background
from claude_swap.settings import load_settings, set_setting


def test_status_defaults_to_disabled(tmp_path: Path):
    st = background.status(tmp_path)

    assert st.enabled is False
    assert st.pid is None
    assert st.running is False
    assert st.pid_path == tmp_path / background.PID_FILENAME
    assert st.log_path == tmp_path / background.LOG_FILENAME


def test_start_enables_and_launches_worker(tmp_path: Path):
    proc = Mock(pid=12345)

    with patch("claude_swap.background._process_running", return_value=False), \
         patch("subprocess.Popen", return_value=proc) as popen:
        background.start(tmp_path)

    assert load_settings(tmp_path).enabled is True
    # The worker writes its own pid once it holds the lock, not the starter.
    assert not (tmp_path / background.PID_FILENAME).exists()
    popen.assert_called_once()
    args = popen.call_args.args[0]
    assert args[-3:] == ["claude_swap", "auto", "_worker"]


def test_start_on_windows_uses_hidden_console(tmp_path: Path):
    # DETACHED_PROCESS leaves no console to inherit, so the venv launcher's
    # child python allocates a visible one; closing it kills the worker.
    proc = Mock(pid=12345)

    # sys.platform is one attribute for every module: patched here it would
    # send the worker-lock check down the Windows path on this host.
    with patch("claude_swap.background.worker_lock") as lock, \
         patch("claude_swap.background.sys.platform", "win32"), \
         patch("claude_swap.background._process_running", return_value=False), \
         patch("subprocess.CREATE_NO_WINDOW", 0x08000000, create=True), \
         patch("subprocess.DETACHED_PROCESS", 0x00000008, create=True), \
         patch("subprocess.CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True), \
         patch("subprocess.CREATE_BREAKAWAY_FROM_JOB", 0x01000000, create=True), \
         patch("subprocess.Popen", return_value=proc) as popen:
        lock.return_value.held_elsewhere.return_value = False
        st = background.start(tmp_path)

    assert st.warnings == ()
    flags = popen.call_args.kwargs["creationflags"]
    assert flags & 0x08000000
    assert not flags & 0x00000008
    assert flags & 0x01000000  # out of the caller's job object


def test_start_on_windows_stays_in_a_job_that_forbids_breakaway(tmp_path: Path):
    proc = Mock(pid=12345)

    with patch("claude_swap.background.worker_lock") as lock, \
         patch("claude_swap.background.sys.platform", "win32"), \
         patch("claude_swap.background._process_running", return_value=False), \
         patch("subprocess.CREATE_NO_WINDOW", 0x08000000, create=True), \
         patch("subprocess.CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True), \
         patch("subprocess.CREATE_BREAKAWAY_FROM_JOB", 0x01000000, create=True), \
         patch("subprocess.Popen", side_effect=[PermissionError(13, "denied"), proc]) as popen:
        lock.return_value.held_elsewhere.return_value = False
        st = background.start(tmp_path)

    first, second = (c.kwargs["creationflags"] for c in popen.call_args_list)
    assert first & 0x01000000
    assert second == 0x08000000 | 0x00000200
    assert any("stops when this session ends" in w for w in st.warnings)
    assert st.to_json()["warnings"] == list(st.warnings)


def test_stop_disables_and_removes_dead_pid(tmp_path: Path):
    (tmp_path / background.PID_FILENAME).write_text("12345\n")

    with patch("claude_swap.background._process_running", return_value=False):
        st = background.stop(tmp_path)

    assert load_settings(tmp_path).enabled is False
    assert st.running is False
    assert not (tmp_path / background.PID_FILENAME).exists()


def test_pause_preserves_enabled_worker_state(tmp_path: Path):
    set_setting(tmp_path, "autoswitch.enabled", "true")
    (tmp_path / background.PID_FILENAME).write_text("12345\n")

    with patch("claude_swap.background._process_running", return_value=True):
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
        pid_path=tmp_path / background.PID_FILENAME,
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
    proc = Mock(pid=12345)
    with patch("claude_swap.background._process_running", return_value=False), \
         patch("subprocess.Popen", return_value=proc) as popen:
        background.start(tmp_path)
    kwargs = popen.call_args.kwargs
    assert kwargs["stderr"].name == str(tmp_path / background.STDERR_FILENAME)
    assert kwargs["stdout"] is background.subprocess.DEVNULL



def test_stop_reports_stopped_only_once_the_lock_is_free(tmp_path: Path):
    # Windows reports a killed worker exited a moment before its lock frees.
    lock = background.worker_lock(tmp_path)
    assert lock.try_acquire(timeout=1.0)
    (tmp_path / background.PID_FILENAME).write_text("12345\n")
    try:
        with patch("claude_swap.background._process_running", return_value=False), \
             patch("claude_swap.background.os.kill",
                   side_effect=lambda *_: threading.Timer(0.3, lock.release).start()):
            st = background.stop(tmp_path)
    finally:
        lock.release()

    assert st.running is False
    assert st.warnings == ()
    assert not (tmp_path / background.PID_FILENAME).exists()


@pytest.mark.parametrize("case", ["pid file already gone", "process already gone"])
def test_stop_waits_for_the_lock_however_the_worker_is_ending(tmp_path: Path, case):
    # A worker shutting down drops its pid file before its lock; one that
    # exited before the signal makes the signal fail.
    lock = background.worker_lock(tmp_path)
    assert lock.try_acquire(timeout=1.0)
    pid_file = tmp_path / background.PID_FILENAME
    if case == "pid file already gone":
        pid_file.unlink()
    else:
        pid_file.write_text("12345\n")
    threading.Timer(0.3, lock.release).start()
    try:
        with patch("claude_swap.background._process_running", return_value=False), \
             patch("claude_swap.background.os.kill", side_effect=ProcessLookupError):
            st = background.stop(tmp_path)
    finally:
        lock.release()

    assert st.running is False
    assert st.warnings == ()
    assert not pid_file.exists()


def test_stop_leaves_the_pid_file_of_a_worker_started_since(tmp_path: Path):
    lock = background.worker_lock(tmp_path)
    assert lock.try_acquire(timeout=1.0)
    pid_file = tmp_path / background.PID_FILENAME
    pid_file.write_text("12345\n")

    def replaced(*_):
        # The signalled worker is gone and another one holds the lock now.
        pid_file.write_text("67890\n")

    try:
        with patch("claude_swap.background._process_running", side_effect=lambda pid: pid == 67890), \
             patch("claude_swap.background.os.kill", side_effect=replaced):
            st = background.stop(tmp_path, timeout=0.5)
    finally:
        lock.release()

    assert pid_file.read_text().strip() == "67890"
    assert st.warnings == ("A worker still holds the auto-switch lock after 0s.",)
