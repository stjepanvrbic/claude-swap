"""The auto-switch worker under systemd and Task Scheduler.

The definitions are checked by the real tools where CI has them
(``systemd-analyze verify`` on Linux, ``schtasks`` on Windows); how
``background.start``/``stop`` drive a manager is checked against a recording
stand-in, since neither CI runner has a logged-in session to run one in.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_swap import background, service
from claude_swap.settings import load_settings, set_setting

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
WORKER = [sys.executable, "-m", "claude_swap", "auto", "_worker", "--service", "systemd"]


class TestSystemdUnit:
    def test_runs_the_worker_and_restarts_it_when_it_dies(self, tmp_path: Path):
        text = service.SystemdUser(unit_dir=tmp_path).render(
            ["/home/me/.local/bin/cswap", "auto", "_worker", "--service", "systemd"], {}
        )

        assert 'ExecStart="/home/me/.local/bin/cswap" "auto" "_worker" "--service" "systemd"\n' in text
        assert "Restart=on-failure\n" in text
        assert "WantedBy=default.target\n" in text

    def test_carries_the_starting_shells_profile(self, tmp_path: Path):
        text = service.SystemdUser(unit_dir=tmp_path).render(
            WORKER, {"CLAUDE_CONFIG_DIR": '/a b/50%$x"y'}
        )

        # Environment= expands % specifiers, not $ variables.
        assert 'Environment="CLAUDE_CONFIG_DIR=/a b/50%%$x\\"y"\n' in text

    @pytest.mark.skipif(sys.platform != "linux", reason="systemd is Linux-only")
    def test_systemd_accepts_the_unit(self, tmp_path: Path):
        unit = service.SystemdUser(unit=f"cswap-test-{uuid.uuid4().hex}.service", unit_dir=tmp_path)
        unit.path.write_text(
            unit.render(WORKER, {"CLAUDE_CONFIG_DIR": "/a b/50%$x"}), encoding="utf-8"
        )

        result = subprocess.run(
            ["systemd-analyze", "verify", str(unit.path)], capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr

    def test_install_writes_only_a_changed_unit(self, tmp_path: Path):
        unit = service.SystemdUser(unit_dir=tmp_path / "user")
        ran = []
        with patch.object(service, "_run", side_effect=lambda argv: ran.append(argv) or Mock(returncode=0)), \
             patch.object(service, "worker_argv", return_value=WORKER):
            unit.install()
            unit.install()

        assert unit.path.read_text(encoding="utf-8") == unit.render(WORKER, {})
        assert ran == [["systemctl", "--user", "daemon-reload"]]


def _task_xml(tmp_path: Path) -> ET.Element:
    xml = service.TaskScheduler(tmp_path).render(
        [r"C:\Users\me\.local\bin\cswap.exe", "auto", "_worker", "--service", "task-scheduler"],
        r"PC\me",
    )
    return ET.fromstring(xml.split("\n", 1)[1])


class TestScheduledTask:
    def test_runs_the_worker_without_a_console_or_a_time_limit(self, tmp_path: Path):
        task = _task_xml(tmp_path)

        assert task.findtext("t:Actions/t:Exec/t:Command", namespaces=NS).endswith(r"System32\conhost.exe")
        assert task.findtext("t:Actions/t:Exec/t:Arguments", namespaces=NS) == (
            r'--headless C:\Users\me\.local\bin\cswap.exe auto _worker --service task-scheduler'
        )
        assert task.findtext("t:Settings/t:ExecutionTimeLimit", namespaces=NS) == "PT0S"
        assert task.findtext("t:Principals/t:Principal/t:LogonType", namespaces=NS) == "InteractiveToken"

    def test_starts_at_logon_and_rechecks_every_two_minutes(self, tmp_path: Path):
        task = _task_xml(tmp_path)

        assert task.findtext("t:Triggers/t:LogonTrigger/t:UserId", namespaces=NS) == r"PC\me"
        # Task Scheduler's own restart-on-failure does not fire when the
        # worker is killed: the repeating trigger is what brings it back,
        # and a run while it is alive does nothing.
        assert task.findtext("t:Triggers/t:TimeTrigger/t:Repetition/t:Interval", namespaces=NS) == "PT2M"
        assert task.find("t:Triggers/t:TimeTrigger/t:Repetition/t:Duration", NS) is None
        assert task.findtext("t:Settings/t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"

    @pytest.mark.real_service_manager
    @pytest.mark.skipif(sys.platform != "win32", reason="Task Scheduler is Windows-only")
    def test_task_scheduler_accepts_the_task(self, tmp_path: Path):
        task = service.TaskScheduler(tmp_path, task=f"cswap-test-{uuid.uuid4().hex}")
        # A command that does nothing, in case the scheduler runs it.
        harmless = [sys.executable, "-c", "pass"]
        try:
            with patch.object(service, "worker_argv", return_value=harmless):
                task.install()
            assert task.installed()
            registered = ET.fromstring(service._run(
                ["schtasks", "/Query", "/TN", task.task, "/XML"]
            ).stdout.split("\n", 1)[1])
            assert registered.findtext(
                "t:Triggers/t:TimeTrigger/t:Repetition/t:Interval", namespaces=NS
            ) == "PT2M"
            assert registered.findtext("t:Settings/t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"
            assert not list(tmp_path.glob("*.xml"))

            task.deactivate()
            disabled = ET.fromstring(service._run(
                ["schtasks", "/Query", "/TN", task.task, "/XML"]
            ).stdout.split("\n", 1)[1])
            assert disabled.findtext("t:Settings/t:Enabled", namespaces=NS) == "false"
        finally:
            subprocess.run(["schtasks", "/Delete", "/TN", task.task, "/F"], capture_output=True)


class FakeManager:
    """Stands in for a service manager whose worker takes the lock when it
    is activated."""

    name = "systemd"

    def __init__(self, *, fails: bool = False, starts_worker: bool = True):
        self.calls: list[str] = []
        self.fails = fails
        self.starts_worker = starts_worker

    def describe(self) -> str:
        return "systemd user unit cswap-auto.service"

    def install(self) -> None:
        self.calls.append("install")

    def installed(self) -> bool:
        return True

    def activate(self) -> None:
        self.calls.append("activate")
        if self.fails:
            raise service.ServiceError("systemctl --user enable --now exited 1: denied")

    def deactivate(self) -> None:
        self.calls.append("deactivate")

    def warnings(self) -> list[str]:
        return []


@pytest.fixture
def no_detached_launch():
    with patch("subprocess.Popen") as popen:
        popen.return_value.poll.return_value = 0
        yield popen


class TestStartUnderAManager:
    def test_start_hands_the_worker_to_the_manager(self, tmp_path: Path, no_detached_launch):
        manager = FakeManager()
        with patch.object(service, "manager", return_value=manager), \
             patch.object(background, "wait_for_worker", return_value=True):
            st = background.start(tmp_path)

        assert manager.calls == ["install", "activate"]
        no_detached_launch.assert_not_called()
        assert load_settings(tmp_path).enabled is True
        assert st.warnings == ()

    def test_a_refusing_manager_still_leaves_a_worker_running(self, tmp_path: Path, no_detached_launch):
        with patch.object(service, "manager", return_value=FakeManager(fails=True)):
            st = background.start(tmp_path)

        no_detached_launch.assert_called_once()
        assert "did not start" in st.warnings[0] and "denied" in st.warnings[0]

    def test_a_manager_that_never_starts_the_worker_is_covered(self, tmp_path: Path, no_detached_launch):
        with patch.object(service, "manager", return_value=FakeManager()), \
             patch.object(background, "wait_for_worker", return_value=False):
            st = background.start(tmp_path)

        no_detached_launch.assert_called_once()
        assert "runs detached for now" in st.warnings[0]

    def test_no_manager_says_why(self, tmp_path: Path, no_detached_launch):
        with patch.object(service, "manager", side_effect=service.Unavailable("no user bus")):
            st = background.start(tmp_path)

        no_detached_launch.assert_called_once()
        assert "no user bus" in st.warnings[0]

    def test_the_managers_running_worker_is_left_alone(self, tmp_path: Path, no_detached_launch):
        manager = FakeManager()
        running = background.BackgroundStatus(
            enabled=True, pid=4321, running=True, pid_path=tmp_path / "p",
            log_path=tmp_path / "l", supervisor="systemd",
        )
        with patch.object(service, "manager", return_value=manager), \
             patch.object(background, "status", return_value=running), \
             patch.object(background, "_end_worker") as end:
            background.start(tmp_path)

        end.assert_not_called()
        assert manager.calls == []

    def test_a_worker_the_manager_did_not_start_gives_way(self, tmp_path: Path, no_detached_launch):
        manager = FakeManager()
        stray = background.BackgroundStatus(
            enabled=True, pid=4321, running=True, pid_path=tmp_path / "p",
            log_path=tmp_path / "l", supervisor=None,
        )
        with patch.object(service, "manager", return_value=manager), \
             patch.object(background, "status", return_value=stray), \
             patch.object(background, "_end_worker") as end, \
             patch.object(background, "wait_for_worker", return_value=True):
            background.start(tmp_path)

        end.assert_called_once_with(tmp_path, 4321)
        assert manager.calls == ["install", "activate"]

    def test_stop_keeps_the_manager_from_starting_it_again(self, tmp_path: Path):
        manager = FakeManager()
        set_setting(tmp_path, "autoswitch.enabled", "true")
        with patch.object(service, "manager", return_value=manager):
            background.stop(tmp_path)

        assert manager.calls == ["deactivate"]
        assert load_settings(tmp_path).enabled is False


def test_supervisor_names_only_the_worker_that_wrote_it(tmp_path: Path):
    background.mark_supervised(tmp_path, "task-scheduler")

    assert background._supervisor(tmp_path, os.getpid()) == "task-scheduler"
    assert background._supervisor(tmp_path, os.getpid() + 1) is None
    background.clear_supervised(tmp_path)
    assert not background.supervisor_path(tmp_path).exists()


@pytest.mark.real_service_manager
def test_manager_on_linux_needs_a_user_systemd(tmp_path: Path):
    refused = Mock(returncode=1, stderr="Failed to connect to bus", stdout="")
    with patch.object(service.sys, "platform", "linux"), \
         patch.object(service.shutil, "which", return_value="/usr/bin/systemctl"), \
         patch.object(service, "_run", return_value=refused):
        with pytest.raises(service.Unavailable, match="Failed to connect to bus"):
            service.manager(tmp_path)
