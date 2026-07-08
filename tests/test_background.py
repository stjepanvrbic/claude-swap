"""Tests for detached auto-switch background worker management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from claude_swap import background
from claude_swap.settings import load_settings


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
        st = background.start(tmp_path)

    assert load_settings(tmp_path).enabled is True
    assert (tmp_path / background.PID_FILENAME).read_text().strip() == "12345"
    assert st.pid == 12345
    popen.assert_called_once()
    args = popen.call_args.args[0]
    assert args[-3:] == ["claude_swap", "auto", "_worker"]


def test_stop_disables_and_removes_dead_pid(tmp_path: Path):
    (tmp_path / background.PID_FILENAME).write_text("12345\n")

    with patch("claude_swap.background._process_running", return_value=False):
        st = background.stop(tmp_path)

    assert load_settings(tmp_path).enabled is False
    assert st.running is False
    assert not (tmp_path / background.PID_FILENAME).exists()

