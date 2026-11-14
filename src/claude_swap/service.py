"""Run the auto-switch worker under the platform's service manager.

A worker launched from a shell runs until something kills it: a closed SSH
session, a crash, a logout. A service manager starts it at login and starts
it again when it dies. On Linux that is a systemd user unit, on Windows a
Task Scheduler task; :func:`manager` returns the one usable here, or raises
:class:`Unavailable` (macOS, no user systemd, no ``schtasks``).

    try:
        m = service.manager(backup_root)
    except service.Unavailable:
        ...                         # launch the worker detached instead
    else:
        m.install()                 # write the definition, if it changed
        m.activate()                # enabled and running
        m.deactivate()              # disabled and stopped

The worker the service runs is ``cswap auto _worker --service <name>``; it
records ``<name>`` beside its pid (see ``background.supervisor``), which is
how ``cswap auto status`` tells the service's worker from a stray one.

*Task Scheduler restarts nothing by itself.* Its "restart on failure" does not
fire when the worker process is killed, and a repetition on a logon trigger
only starts at the next logon. So the task also has a time trigger that
repeats every two minutes; with "do not start a new instance" a run while
the worker is alive is a no-op, and one after it died starts a new worker.
"""

from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from claude_swap import launch_agent
from claude_swap.exceptions import ClaudeSwitchError

SYSTEMD_UNIT = "cswap-auto.service"
TASK_NAME = "cswap auto-switch"
# How soon a dead worker comes back on Windows.
TASK_REPEAT = "PT2M"
_COMMAND_TIMEOUT_S = 30.0


class Unavailable(Exception):
    """No service manager can run the worker here; ``str()`` says why, or is
    empty where cswap does not use one (macOS)."""


class ServiceError(ClaudeSwitchError):
    """The service manager refused a step; ``str()`` names the command and
    what it printed."""


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Run one service-manager command; :class:`ServiceError` when it cannot
    run or does not finish."""
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, errors="replace",
            timeout=_COMMAND_TIMEOUT_S, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ServiceError(f"{' '.join(argv)}: {e}") from e


def _checked(argv: list[str]) -> subprocess.CompletedProcess:
    result = _run(argv)
    if result.returncode != 0:
        said = (result.stderr or result.stdout or "").strip()
        raise ServiceError(f"{' '.join(argv)} exited {result.returncode}: {said}")
    return result


def worker_argv(name: str) -> list[str]:
    """The worker invocation a service runs: the console script, which keeps
    its path across upgrades (see ``launch_agent.resolve_program``)."""
    return [*launch_agent.resolve_program(), "auto", "_worker", "--service", name]


class SystemdUser:
    """A systemd user unit; ``unit_dir`` is where its file lives."""

    name = "systemd"

    def __init__(self, unit: str = SYSTEMD_UNIT, unit_dir: Path | None = None):
        self.unit = unit
        config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        self.unit_dir = unit_dir or Path(config) / "systemd" / "user"

    @property
    def path(self) -> Path:
        return self.unit_dir / self.unit

    def describe(self) -> str:
        return f"systemd user unit {self.unit}"

    def render(self, argv: list[str], env: dict[str, str]) -> str:
        exec_start = " ".join(_systemd_quote(arg, expands_variables=True) for arg in argv)
        environment = "".join(
            f"Environment={_systemd_quote(f'{key}={value}')}\n"
            for key, value in sorted(env.items())
        )
        return (
            "# Written by `cswap auto start`, which rewrites it: change it with\n"
            "# `systemctl --user edit` instead.\n"
            "[Unit]\n"
            "Description=Claude Swap automatic account switcher\n"
            "\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={exec_start}\n"
            f"{environment}"
            "Restart=on-failure\n"
            "RestartSec=10s\n"
            "TimeoutStopSec=60s\n"
            "UMask=0077\n"
            "\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )

    def install(self) -> None:
        text = self.render(worker_argv(self.name), _forwarded_env())
        try:
            current = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current == text:
            return
        try:
            self.unit_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            raise ServiceError(f"writing {self.path}: {e}") from e
        _checked(["systemctl", "--user", "daemon-reload"])

    def installed(self) -> bool:
        return self.path.exists()

    def activate(self) -> None:
        if self._failed():
            # A unit that hit its start limit stays failed until reset.
            _checked(["systemctl", "--user", "reset-failed", self.unit])
        _checked(["systemctl", "--user", "enable", "--now", self.unit])

    def deactivate(self) -> None:
        _checked(["systemctl", "--user", "disable", "--now", self.unit])

    def _failed(self) -> bool:
        return _run(["systemctl", "--user", "is-failed", "--quiet", self.unit]).returncode == 0

    def warnings(self) -> list[str]:
        """What keeps a user unit from running while you are logged out."""
        user = os.environ.get("USER") or os.environ.get("LOGNAME")
        if not user or shutil.which("loginctl") is None:
            return []
        result = _run(["loginctl", "show-user", user, "--property=Linger", "--value"])
        if result.returncode == 0 and result.stdout.strip() == "no":
            return [
                "Lingering is off for this user, so systemd stops the worker "
                "when you log out. `loginctl enable-linger` keeps it running."
            ]
        return []


class TaskScheduler:
    """A Windows Task Scheduler task that runs the worker in your logon
    session; its XML is written to ``scratch_dir`` while it registers."""

    name = "task-scheduler"

    def __init__(self, scratch_dir: Path, task: str = TASK_NAME):
        self.task = task
        self.scratch_dir = scratch_dir

    def describe(self) -> str:
        return f"Task Scheduler task '{self.task}'"

    def render(self, argv: list[str], user: str) -> str:
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        conhost = ntpath.join(system_root, "System32", "conhost.exe")
        # conhost --headless gives the console script a console nobody sees.
        arguments = "--headless " + subprocess.list2cmdline(argv)
        return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Claude Swap automatic account switcher. Written by `cswap auto start`, which rewrites it.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{escape(user)}</UserId>
    </LogonTrigger>
    <TimeTrigger>
      <Repetition>
        <Interval>{TASK_REPEAT}</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2000-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(user)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(conhost)}</Command>
      <Arguments>{escape(arguments)}</Arguments>
    </Exec>
  </Actions>
</Task>
"""

    def install(self) -> None:
        env = [f"--env={key}={value}" for key, value in sorted(_forwarded_env().items())]
        xml = self.render([*worker_argv(self.name), *env], _windows_account())
        path = self.scratch_dir / "autoswitch_task.xml"
        try:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(xml, encoding="utf-16")
        except OSError as e:
            raise ServiceError(f"writing {path}: {e}") from e
        try:
            _checked(["schtasks", "/Create", "/TN", self.task, "/XML", str(path), "/F"])
        finally:
            path.unlink(missing_ok=True)

    def installed(self) -> bool:
        return _run(["schtasks", "/Query", "/TN", self.task]).returncode == 0

    def activate(self) -> None:
        _checked(["schtasks", "/Change", "/TN", self.task, "/ENABLE"])
        _checked(["schtasks", "/Run", "/TN", self.task])

    def deactivate(self) -> None:
        _checked(["schtasks", "/Change", "/TN", self.task, "/DISABLE"])
        # Fails when the task is not running, which is the state we want.
        _run(["schtasks", "/End", "/TN", self.task])

    def warnings(self) -> list[str]:
        return []


Manager = SystemdUser | TaskScheduler


def manager(backup_root: Path) -> Manager:
    """The service manager that can run the worker of ``backup_root`` here.

    Raises :class:`Unavailable` when there is none: systemd without a user
    manager to talk to (a container, WSL without systemd, ``su``), no
    ``schtasks``, or macOS, where the worker is not run as a service.
    """
    if sys.platform == "win32":
        if shutil.which("schtasks") is None:
            raise Unavailable("schtasks is not on PATH")
        return TaskScheduler(backup_root)
    if sys.platform == "darwin":
        raise Unavailable("")
    if shutil.which("systemctl") is None:
        raise Unavailable("systemctl is not on PATH")
    try:
        probe = _run(["systemctl", "--user", "show", "--property=Version"])
    except ServiceError as e:
        raise Unavailable(str(e)) from e
    if probe.returncode != 0:
        said = (probe.stderr or probe.stdout or "").strip()
        raise Unavailable(f"no systemd user manager to talk to ({said})")
    return SystemdUser()


def _windows_account() -> str:
    """This user as ``DOMAIN\\name`` (``MACHINE\\name`` for a local account),
    the form Task Scheduler maps to the account. ``USERDOMAIN`` will not do:
    an SSH session sets it to ``WORKGROUP``."""
    import ctypes

    name_sam_compatible = 2
    secur32 = ctypes.WinDLL("secur32")
    size = ctypes.c_ulong(0)
    secur32.GetUserNameExW(name_sam_compatible, None, ctypes.byref(size))
    buffer = ctypes.create_unicode_buffer(size.value)
    if not secur32.GetUserNameExW(name_sam_compatible, buffer, ctypes.byref(size)):
        raise ServiceError(f"GetUserNameExW failed: error {ctypes.GetLastError()}")
    return buffer.value


def _forwarded_env() -> dict[str, str]:
    """The profile, proxy and CA variables of the shell that starts the
    service, which a user unit does not inherit (same set the menu bar's
    LaunchAgent carries)."""
    return {
        key: os.environ[key]
        for key in launch_agent._FORWARDED_ENV_KEYS
        if key in os.environ
    }


def _systemd_quote(word: str, *, expands_variables: bool = False) -> str:
    """One unit-file word, double-quoted, with systemd's ``%`` specifiers
    escaped, and its ``$`` variables too where the setting expands them
    (ExecStart does, Environment does not)."""
    escaped = word.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if expands_variables:
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'
