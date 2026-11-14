"""The active account is read from the rate-limit headers of a one-token
request, and any account the usage endpoint refuses with 429 falls back to
that probe.

The engine, the collector, the usage store and ``switch_to`` are the real
ones on the Linux file backend; only the two requests are answered from a
per-token table, in the raw shapes the API sends.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import background, oauth
from claude_swap.autoswitch import AutoSwitchEngine, SwitchEvent, TickOutcome
from claude_swap.models import Platform
from claude_swap.paths import get_backup_root
from claude_swap.settings import AutoSwitchSettings, load_settings, set_setting
from claude_swap.switcher import ClaudeAccountSwitcher

HOUR = 3600.0
DAY = 24 * HOUR


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _http_429() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.anthropic.com/api/oauth/usage", 429, "Too Many Requests",
        {"Retry-After": "0"}, None,
    )


class Account:
    """One account's server-side usage; ``five_hour`` None is Ultra."""

    def __init__(self, *, seven_day=0.0, fable=0.0, five_hour=None,
                 weekly_reset_in=3 * DAY):
        self.seven_day = seven_day
        self.fable = fable
        self.five_hour = five_hour
        self.weekly_reset = time.time() + weekly_reset_in

    def endpoint_body(self) -> dict:
        body = {
            "five_hour": None,
            "seven_day": {"utilization": self.seven_day, "resets_at": _iso(self.weekly_reset)},
            "limits": [{
                "kind": "weekly_scoped",
                "scope": {"model": {"display_name": "Fable"}},
                "percent": self.fable,
                "resets_at": _iso(self.weekly_reset),
            }],
        }
        if self.five_hour is not None:
            body["five_hour"] = {"utilization": self.five_hour, "resets_at": _iso(time.time() + HOUR)}
        return body

    def headers(self, model: str) -> dict:
        """The response headers a probe of ``model`` gets back: Fable's
        window (``7d_oi``) only when Fable was asked."""
        h = {
            "anthropic-ratelimit-unified-7d-utilization": str(self.seven_day / 100),
            "anthropic-ratelimit-unified-7d-reset": str(int(self.weekly_reset)),
        }
        if self.five_hour is not None:
            h["anthropic-ratelimit-unified-5h-utilization"] = str(self.five_hour / 100)
            h["anthropic-ratelimit-unified-5h-reset"] = str(int(time.time() + HOUR))
        if model == oauth.FABLE_PROBE_MODEL:
            h["anthropic-ratelimit-unified-7d_oi-utilization"] = str(self.fable / 100)
            h["anthropic-ratelimit-unified-7d_oi-reset"] = str(int(self.weekly_reset))
        return h


class Network:
    def __init__(self):
        self.accounts: dict[str, Account] = {}
        self.endpoint_calls: list[str] = []
        self.probe_calls: list[tuple[str, str]] = []
        self.endpoint_refuses = False
        self.probe_error: Exception | None = None

    def endpoint(self, token: str) -> dict:
        self.endpoint_calls.append(token)
        if self.endpoint_refuses:
            raise _http_429()
        return self.accounts[token].endpoint_body()

    def probe(self, token: str, model: str = oauth.USAGE_PROBE_MODEL) -> dict:
        self.probe_calls.append((token, model))
        if self.probe_error is not None:
            raise self.probe_error
        return oauth.parse_unified_ratelimit_headers(self.accounts[token].headers(model))


@pytest.fixture
def network(monkeypatch) -> Network:
    net = Network()
    monkeypatch.setattr(oauth, "request_usage_data", net.endpoint)
    monkeypatch.setattr(oauth, "request_usage_via_probe", net.probe)
    return net


def _creds(token: str) -> str:
    return json.dumps({"claudeAiOauth": {
        "accessToken": token, "refreshToken": f"rt-{token}",
        "expiresAt": int((time.time() + 8 * HOUR) * 1000),
    }})


class TestTryFetch:
    def test_the_active_account_is_read_by_the_probe_alone(self, network):
        network.accounts["sk-1"] = Account(fable=42)

        outcome = oauth.try_fetch_usage_for_account(
            "1", "a@example.com", _creds("sk-1"), is_active=True,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert network.endpoint_calls == []
        assert network.probe_calls == [("sk-1", oauth.FABLE_PROBE_MODEL)]
        assert outcome.usage["scoped"][0]["name"] == "Fable"
        assert outcome.usage["scoped"][0]["pct"] == 42.0

    def test_an_unanswered_probe_falls_back_to_the_endpoint(self, network):
        # 529 is the API being overloaded, not the account being spent.
        network.accounts["sk-1"] = Account(fable=20)
        network.probe_error = urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages", 529, "Overloaded", {}, None
        )

        outcome = oauth.try_fetch_usage_for_account(
            "1", "a@example.com", _creds("sk-1"), is_active=True,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert network.endpoint_calls == ["sk-1"]
        assert outcome.error is None
        assert outcome.usage["scoped"][0]["pct"] == 20.0

    def test_a_probe_without_the_fable_window_falls_back_to_the_endpoint(
        self, network, monkeypatch
    ):
        network.accounts["sk-1"] = Account(fable=20)
        monkeypatch.setattr(
            oauth, "request_usage_via_probe",
            lambda token, model=None: oauth.parse_unified_ratelimit_headers(
                network.accounts[token].headers(oauth.USAGE_PROBE_MODEL)
            ),
        )

        outcome = oauth.try_fetch_usage_for_account(
            "1", "a@example.com", _creds("sk-1"), is_active=True,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert network.endpoint_calls == ["sk-1"]
        assert outcome.usage["scoped"][0]["pct"] == 20.0

    def test_a_failed_probe_and_a_refused_endpoint_record_the_probe_failure(
        self, network
    ):
        # The endpoint's hour-long 429 would park the account, probe and
        # all; the probe's failure is the short, honest record.
        network.accounts["sk-1"] = Account(fable=20)
        network.probe_error = urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages", 529, "Overloaded", {}, None
        )
        network.endpoint_refuses = True

        outcome = oauth.try_fetch_usage_for_account(
            "1", "a@example.com", _creds("sk-1"), is_active=True,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert outcome.error == "http-529"
        assert outcome.retry_after_s is None
        assert len(network.probe_calls) == 1

    def test_another_account_is_read_from_the_endpoint(self, network):
        network.accounts["sk-2"] = Account(fable=10)

        outcome = oauth.try_fetch_usage_for_account(
            "2", "b@example.com", _creds("sk-2"), is_active=False,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert network.endpoint_calls == ["sk-2"]
        assert network.probe_calls == []
        assert outcome.error is None

    def test_a_429_from_the_endpoint_falls_back_to_the_probe(self, network):
        network.accounts["sk-2"] = Account(fable=70)
        network.endpoint_refuses = True

        outcome = oauth.try_fetch_usage_for_account(
            "2", "b@example.com", _creds("sk-2"), is_active=False,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert outcome.error is None
        assert outcome.usage["source"] == oauth.USAGE_SOURCE_HEADERS
        assert outcome.usage["scoped"][0]["pct"] == 70.0

    def test_a_failed_fallback_keeps_the_429(self, network):
        network.accounts["sk-2"] = Account()
        network.endpoint_refuses = True
        network.probe_error = urllib.error.URLError(OSError("down"))

        outcome = oauth.try_fetch_usage_for_account(
            "2", "b@example.com", _creds("sk-2"), is_active=False,
            probe_model=oauth.FABLE_PROBE_MODEL,
        )

        assert outcome.usage is None
        assert outcome.error == "http-429"
        assert outcome.retry_after_s == 0.0

    def test_without_a_probe_model_the_endpoint_is_used_as_before(self, network):
        network.accounts["sk-1"] = Account()

        oauth.try_fetch_usage_for_account(
            "1", "a@example.com", _creds("sk-1"), is_active=True,
        )

        assert network.endpoint_calls == ["sk-1"]
        assert network.probe_calls == []


def test_a_window_the_server_refuses_reads_as_spent():
    headers = {
        "anthropic-ratelimit-unified-7d-utilization": "0.5",
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.97",
        "anthropic-ratelimit-unified-7d_oi-status": "rejected",
    }
    usage = oauth.build_usage_result(oauth.parse_unified_ratelimit_headers(headers))

    assert usage["seven_day"]["pct"] == 50.0
    assert usage["scoped"][0]["pct"] == 100.0


class Fleet:
    """Accounts 1..n on a temp home, account 1 active."""

    def __init__(self, temp_home: Path, n: int = 4, **settings):
        self.home = temp_home
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(temp_home / ".local" / "share")}):
            self.switcher = ClaudeAccountSwitcher()
        self.switcher.platform = Platform.LINUX
        self.switcher._setup_directories()
        self.switcher._init_sequence_file()
        for number in range(1, n + 1):
            self._seed(number)
        self._make_live(1)
        base = {"threshold": 95.0, "weekly_threshold": 98.0, "model": "Fable",
                "strategy": "weekly-first", "cooldown_seconds": 0.0}
        self.settings = AutoSwitchSettings(**{**base, **settings})
        set_setting(self.switcher.backup_dir, "autoswitch.model", self.settings.model)
        self.events: list = []
        self.engine = AutoSwitchEngine(self.switcher, self.settings, self.events.append)

    @staticmethod
    def email(number: int) -> str:
        return f"account{number}@example.com"

    def _seed(self, number: int) -> None:
        email = self.email(number)
        self.switcher._write_account_credentials(str(number), email, _creds(f"sk-{number}"))
        self.switcher._write_account_config(str(number), email, json.dumps({
            "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{number}"},
        }))
        data = self.switcher._get_sequence_data()
        data["accounts"][str(number)] = {
            "email": email, "uuid": f"uuid-{number}", "organizationUuid": "",
            "organizationName": "", "added": "2024-01-01T00:00:00Z",
        }
        data["sequence"] = sorted(set(data["sequence"]) | {number})
        if data["activeAccountNumber"] is None:
            data["activeAccountNumber"] = number
        self.switcher._write_json(self.switcher.sequence_file, data)

    def _make_live(self, number: int) -> None:
        (self.home / ".claude" / ".credentials.json").write_text(_creds(f"sk-{number}"))
        (self.home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": self.email(number), "accountUuid": f"uuid-{number}"},
        }))

    def active(self) -> int:
        return int(self.switcher.current_account_number())

    def switches(self) -> list[SwitchEvent]:
        return [e for e in self.events if isinstance(e, SwitchEvent)]


class TestEngineWhileTheProbeFails:
    def test_a_healthy_account_is_kept_when_only_the_probe_fails(
        self, temp_home, network
    ):
        fleet = Fleet(temp_home)
        network.probe_error = urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages", 529, "Overloaded", {}, None
        )
        network.accounts["sk-1"] = Account(fable=20)
        for n in (2, 3, 4):
            network.accounts[f"sk-{n}"] = Account(fable=0)

        for _ in range(4):
            fleet.engine.tick()

        assert fleet.switches() == []
        assert fleet.active() == 1


class TestEngineWhileTheEndpointRefuses:
    """The failure in the field: the endpoint answered 429 for hours, the
    active account's usage was unknown, and nothing switched."""

    def test_the_active_account_is_still_read_and_kept_below_its_line(
        self, temp_home, network
    ):
        fleet = Fleet(temp_home)
        network.endpoint_refuses = True
        network.accounts["sk-1"] = Account(fable=60)
        for n in (2, 3, 4):
            network.accounts[f"sk-{n}"] = Account(fable=0)

        assert fleet.engine.tick() is TickOutcome.NO_ACTION

        assert ("sk-1", oauth.FABLE_PROBE_MODEL) in network.probe_calls
        assert "sk-1" not in network.endpoint_calls
        assert fleet.active() == 1
        assert any(
            e.kind == "no-switch" and e.reason == "below-threshold" for e in fleet.events
        )

    def test_crossing_the_fable_line_lands_on_ultra_with_the_soonest_reset(
        self, temp_home, network
    ):
        fleet = Fleet(temp_home)
        network.endpoint_refuses = True
        network.accounts["sk-1"] = Account(seven_day=50, fable=99)
        network.accounts["sk-2"] = Account(five_hour=0, fable=0, weekly_reset_in=10 * HOUR)
        network.accounts["sk-3"] = Account(fable=30, weekly_reset_in=2 * DAY)
        network.accounts["sk-4"] = Account(fable=0, weekly_reset_in=5 * DAY)

        outcome = fleet.engine.tick()

        assert outcome is TickOutcome.SWITCHED
        assert fleet.active() == 3


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
    set_setting(root, "autoswitch.enabled", "false")
    lock = background.worker_lock(root)
    deadline = time.monotonic() + 30
    while lock.held_elsewhere() and time.monotonic() < deadline:
        time.sleep(0.2)


def _holder_pid(root: Path) -> int:
    """The pid the worker writes just after it takes the lock."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            return int(background.pid_path(root).read_text().strip())
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError("the worker never wrote its pid")


def _worker() -> list[str]:
    return [sys.executable, "-m", "claude_swap", "auto", "_worker"]


def test_start_reports_the_worker_that_holds_the_lock(worker_root: Path):
    st = background.start(worker_root)
    lock = background.worker_lock(worker_root)
    deadline = time.monotonic() + 60
    while not lock.held_elsewhere() and time.monotonic() < deadline:
        time.sleep(0.2)
    holder = _holder_pid(worker_root)

    with patch("subprocess.Popen") as popen:
        again = background.start(worker_root)

    popen.assert_not_called()
    assert again.running and again.pid == holder
    assert st.running


def test_only_one_worker_runs_per_backup_root(worker_root: Path):
    set_setting(worker_root, "autoswitch.enabled", "true")
    first = subprocess.Popen(_worker(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        lock = background.worker_lock(worker_root)
        deadline = time.monotonic() + 60
        while not lock.held_elsewhere() and time.monotonic() < deadline:
            time.sleep(0.2)
        holder = _holder_pid(worker_root)
        assert background.status(worker_root).running

        second = subprocess.run(_worker(), capture_output=True, timeout=120)

        assert second.returncode == 0
        assert int(background.pid_path(worker_root).read_text().strip()) == holder
        assert lock.held_elsewhere()

        # Disabling ends the worker between ticks, and the lock with it.
        set_setting(worker_root, "autoswitch.enabled", "false")
        assert first.wait(timeout=60) == 0
        assert not lock.held_elsewhere()
        assert load_settings(worker_root).enabled is False
    finally:
        if first.poll() is None:
            try:
                os.kill(int(background.pid_path(worker_root).read_text()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
            first.kill()
            first.wait()
