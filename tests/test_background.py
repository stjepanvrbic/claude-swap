"""Tests for detached auto-switch background worker management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

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
         patch("subprocess.Popen", return_value=proc) as popen:
        lock.return_value.held_elsewhere.return_value = False
        background.start(tmp_path)

    flags = popen.call_args.kwargs["creationflags"]
    assert flags & 0x08000000
    assert not flags & 0x00000008


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
