"""The auto-switch engine against a real account store on the Linux file
backend. Only the network is replaced: the probe and the usage endpoint
answer from a per-token table, in the raw shapes the API sends, so parsing,
recording in the usage store and the real ``switch_to`` all run.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.autoswitch import (
    LOCK_FILENAME,
    MAX_SLEEP_S,
    AllExhaustedEvent,
    AutoSwitchEngine,
    EngineBusy,
    NoSwitchEvent,
    PollEvent,
    QuarantineEvent,
    SwitchEvent,
    TickOutcome,
    account_windows,
)
from claude_swap.locking import InstanceLock
from claude_swap.models import Platform
from claude_swap.settings import AutoSwitchSettings
from claude_swap.switcher import ClaudeAccountSwitcher

HOUR = 3600.0
DAY = 24 * HOUR
NOW = 1_800_000_000.0


class FakeClock:
    def __init__(self, now: float = NOW):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class Account:
    """One account's server-side usage. ``five_hour`` None is an Ultra
    account (no 5-hour window)."""

    def __init__(self, *, seven_day=0.0, fable=0.0, five_hour=None,
                 weekly_reset=NOW + 3 * DAY, refused=False, opus=None):
        self.seven_day = seven_day
        self.fable = fable
        self.five_hour = five_hour
        self.weekly_reset = weekly_reset
        self.refused = refused
        self.opus = opus  # a per-model window only the endpoint reports

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
            body["five_hour"] = {"utilization": self.five_hour, "resets_at": _iso(NOW + 2 * HOUR)}
        if self.opus is not None:
            body["limits"].append({
                "kind": "weekly_scoped",
                "scope": {"model": {"display_name": "Opus"}},
                "percent": self.opus,
                "resets_at": _iso(self.weekly_reset),
            })
        return body

    def probe_headers(self, model: str) -> dict:
        """What :func:`oauth.parse_unified_ratelimit_headers` makes of the
        probe's response headers: no Fable window unless Fable was asked."""
        data = {"seven_day": {"utilization": self.seven_day, "resets_at": _iso(self.weekly_reset)}}
        if self.five_hour is not None:
            data["five_hour"] = {"utilization": self.five_hour, "resets_at": _iso(NOW + 2 * HOUR)}
        if model == oauth.FABLE_PROBE_MODEL:
            data["limits"] = [{
                "kind": "weekly_scoped",
                "scope": {"model": {"display_name": "Fable"}},
                "percent": self.fable,
                "resets_at": _iso(self.weekly_reset),
            }]
        if self.refused:
            data["rejected"] = {"resets_at": _iso(self.weekly_reset), "retry_after_s": None}
        return data


class Network:
    """The two usage requests, answered per access token."""

    def __init__(self):
        self.accounts: dict[str, Account] = {}
        self.probe_calls: list[tuple[str, str]] = []
        self.endpoint_calls: list[str] = []
        self.endpoint_error: dict[str, Exception] = {}
        self.probe_error: dict[str, Exception] = {}

    def probe(self, token: str, model: str = oauth.USAGE_PROBE_MODEL) -> dict:
        self.probe_calls.append((token, model))
        if token in self.probe_error:
            raise self.probe_error[token]
        return self.accounts[token].probe_headers(model)

    def endpoint(self, token: str) -> dict:
        self.endpoint_calls.append(token)
        if token in self.endpoint_error:
            raise self.endpoint_error[token]
        return self.accounts[token].endpoint_body()

    def probed(self) -> list[str]:
        return [token for token, _ in self.probe_calls]


def _http_429() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.anthropic.com/api/oauth/usage", 429, "Too Many Requests",
        {"Retry-After": "0"}, None,
    )


class Fleet:
    """Accounts 1..n on a temp home, account 1 active, and an engine."""

    def __init__(self, temp_home: Path, network: Network, n: int = 4, **settings):
        self.home = temp_home
        self.network = network
        self.clock = FakeClock()
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(temp_home / ".local" / "share")}):
            self.switcher = ClaudeAccountSwitcher()
        self.switcher.platform = Platform.LINUX
        self.switcher._setup_directories()
        self.switcher._init_sequence_file()
        self.switcher._usage_store.clock = self.clock
        for number in range(1, n + 1):
            self._seed(number)
        self._make_live(1)
        defaults = {"threshold": 95.0, "weekly_threshold": 98.0, "model": "Fable",
                    "cooldown_seconds": 0.0}
        self.settings = AutoSwitchSettings(**{**defaults, **settings})
        self.events: list = []
        self.engine = self.make_engine()

    def make_engine(self, **kwargs) -> AutoSwitchEngine:
        return AutoSwitchEngine(
            self.switcher, self.settings, self.events.append, clock=self.clock, **kwargs
        )

    @staticmethod
    def email(number: int) -> str:
        return f"account{number}@example.com"

    def _creds(self, number: int) -> str:
        return json.dumps({"claudeAiOauth": {
            "accessToken": f"sk-{number}",
            "refreshToken": f"rt-{number}",
            "expiresAt": int((NOW + 8 * HOUR) * 1000),
        }})

    def _seed(self, number: int) -> None:
        email = self.email(number)
        self.switcher._write_account_credentials(str(number), email, self._creds(number))
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
        (self.home / ".claude" / ".credentials.json").write_text(self._creds(number))
        (self.home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": self.email(number), "accountUuid": f"uuid-{number}"},
        }))

    def set(self, number: int, **usage) -> None:
        self.network.accounts[f"sk-{number}"] = Account(**usage)

    def tick(self) -> TickOutcome:
        return self.engine.tick()

    def active(self) -> int:
        return int(self.switcher.current_account_number())

    def switches(self) -> list[SwitchEvent]:
        return [e for e in self.events if isinstance(e, SwitchEvent)]

    def holds(self) -> list[str]:
        return [e.reason for e in self.events if isinstance(e, NoSwitchEvent)]

    def read_all(self) -> None:
        """Tick with the clock moved on until every inactive account has
        been read once, as the loop does over its first minutes."""
        for _ in range(len(self.network.accounts)):
            self.tick()
            self.clock.advance(self.settings.interval_seconds)


@pytest.fixture
def network(monkeypatch) -> Network:
    net = Network()
    monkeypatch.setattr(oauth, "request_usage_via_probe", net.probe)
    monkeypatch.setattr(oauth, "request_usage_data", net.endpoint)
    return net


@pytest.fixture
def fleet(temp_home, network) -> Fleet:
    return Fleet(temp_home, network)


class TestStayingPut:
    def test_the_active_account_is_kept_below_its_lines(self, fleet):
        # 3 has all its quota and resets in an hour; leaving 1 would still
        # cost every session its prompt cache, so 1 stays.
        fleet.set(1, seven_day=40, fable=90)
        fleet.set(2, seven_day=10, fable=10)
        fleet.set(3, seven_day=0, fable=0, weekly_reset=NOW + HOUR)
        fleet.set(4, seven_day=0, fable=0)
        fleet.read_all()

        assert fleet.switches() == []
        assert fleet.active() == 1
        assert "below-threshold" in fleet.holds()

    def test_unknown_active_usage_never_moves_the_account(self, fleet, network):
        fleet.set(1, fable=50)
        fleet.set(2)
        fleet.set(3)
        fleet.set(4)
        network.probe_error["sk-1"] = urllib.error.URLError(OSError("down"))

        for _ in range(10):
            fleet.tick()
            fleet.clock.advance(60)

        assert fleet.switches() == []
        assert fleet.active() == 1
        assert "active-usage-unknown" in fleet.holds()


class TestReading:
    def test_the_active_account_is_read_by_a_fable_probe_not_the_endpoint(self, fleet, network):
        fleet.set(1, fable=20)
        for n in (2, 3, 4):
            fleet.set(n)

        fleet.tick()

        assert ("sk-1", oauth.FABLE_PROBE_MODEL) in network.probe_calls
        assert "sk-1" not in network.endpoint_calls
        entry = fleet.switcher.usage_store().entries(
            fleet.switcher.usage_identities(["1"]), ("Fable",)
        )["1"]
        assert entry.last_good["scoped"][0]["pct"] == 20.0
        assert entry.held_until is not None

    def test_the_active_account_is_probed_faster_near_its_line(self, fleet, network):
        fleet.set(1, fable=95)
        for n in (2, 3, 4):
            fleet.set(n)
        fleet.tick()
        network.probe_calls.clear()

        fleet.clock.advance(25)
        fleet.tick()

        assert network.probed().count("sk-1") == 1

    def test_other_accounts_are_read_one_per_tick_from_the_endpoint(self, fleet, network):
        for n in (1, 2, 3, 4):
            fleet.set(n)

        fleet.tick()
        assert len(network.endpoint_calls) == 1
        fleet.clock.advance(15)
        fleet.tick()
        assert len(network.endpoint_calls) == 2

    def test_a_refused_endpoint_read_waits_out_its_backoff(self, fleet, network):
        for n in (1, 2, 3, 4):
            fleet.set(n)
        network.endpoint_error["sk-2"] = _http_429()

        for _ in range(6):
            fleet.tick()
            fleet.clock.advance(15)

        assert network.endpoint_calls.count("sk-2") == 1


class TestLeaving:
    def test_crossing_the_fable_line_moves_to_an_ultra_account_first(self, fleet):
        fleet.set(1, seven_day=50, fable=97)
        fleet.set(2, five_hour=0, seven_day=0, fable=0)
        fleet.set(3, seven_day=60, fable=40)
        fleet.set(4, five_hour=0, seven_day=5, fable=5)
        fleet.read_all()
        fleet.set(1, seven_day=51, fable=98.5)
        fleet.clock.advance(60)

        assert fleet.tick() is TickOutcome.SWITCHED
        assert fleet.active() == 3
        assert fleet.switches()[-1].trigger == "threshold"

    def test_among_ultra_accounts_the_soonest_weekly_reset_goes_first(self, fleet):
        fleet.set(1, fable=97)
        fleet.set(2, fable=0, weekly_reset=NOW + 5 * DAY)
        fleet.set(3, fable=60, weekly_reset=NOW + 10 * HOUR)
        fleet.set(4, fable=30, weekly_reset=NOW + 2 * DAY)
        fleet.read_all()
        fleet.set(1, fable=99)
        fleet.clock.advance(60)

        fleet.tick()

        assert fleet.active() == 3

    def test_a_priority_tier_comes_before_ultra(self, fleet):
        fleet.switcher.set_account_priority("2", "5")
        fleet.set(1, fable=97)
        fleet.set(2, five_hour=0, fable=0)
        fleet.set(3, fable=0, weekly_reset=NOW + HOUR * 3)
        fleet.set(4, fable=0)
        fleet.read_all()
        fleet.set(1, fable=99)
        fleet.clock.advance(60)

        fleet.tick()

        assert fleet.active() == 2

    def test_a_target_is_confirmed_by_a_fresh_probe(self, fleet, network):
        fleet.set(1, fable=97)
        fleet.set(2, fable=10, weekly_reset=NOW + DAY)
        fleet.set(3, fable=20, weekly_reset=NOW + 2 * DAY)
        fleet.set(4, fable=90)
        fleet.read_all()
        # 2 was spent elsewhere since it was read.
        fleet.set(2, fable=100, weekly_reset=NOW + DAY)
        fleet.set(1, fable=99)
        fleet.clock.advance(60)

        fleet.tick()

        assert fleet.active() == 3
        assert "sk-2" in network.probed()

    def test_an_account_never_read_is_probed_before_it_is_used(self, fleet, network):
        fleet.set(1, fable=99)
        fleet.set(2, fable=10)
        fleet.set(3, fable=100)
        fleet.set(4, fable=100)

        fleet.tick()

        assert fleet.active() == 2
        assert "sk-2" in network.probed()

    def test_a_window_past_its_reset_counts_as_empty(self, fleet):
        fleet.set(1, fable=97)
        fleet.set(2, fable=100, weekly_reset=NOW + 30 * 60)
        fleet.set(3, fable=100)
        fleet.set(4, fable=100)
        fleet.read_all()
        fleet.clock.advance(HOUR)
        fleet.set(2, fable=0, weekly_reset=NOW + 7 * DAY)
        fleet.set(1, fable=99)

        fleet.tick()

        assert fleet.active() == 2

    def test_cooldown_holds_a_threshold_switch_but_not_a_refusal(self, temp_home, network):
        fleet = Fleet(temp_home, network, cooldown_seconds=600.0)
        fleet.set(1, fable=97)
        fleet.set(2, fable=0)
        fleet.set(3, fable=0)
        fleet.set(4, fable=0)
        fleet.read_all()
        fleet.engine._mutate_state(lambda s: s.update(lastSwitchAt=fleet.clock.now))
        fleet.set(1, fable=99)
        fleet.clock.advance(60)

        fleet.tick()
        assert fleet.active() == 1
        assert "cooldown" in fleet.holds()

        fleet.set(1, fable=100, refused=True)
        fleet.clock.advance(60)
        fleet.tick()
        assert fleet.active() != 1
        assert fleet.switches()[-1].trigger == "at-limit"


class TestNothingRoomy:
    def test_a_nearly_spent_target_waits_until_the_active_is_refused(self, fleet):
        fleet.set(1, fable=97)
        fleet.set(2, fable=95)
        fleet.set(3, fable=100)
        fleet.set(4, fable=100)
        fleet.read_all()
        fleet.set(1, fable=98.5)
        fleet.clock.advance(60)

        assert fleet.tick() is TickOutcome.BLOCKED
        assert fleet.active() == 1
        assert "no-roomy-target" in fleet.holds()

        fleet.set(1, fable=100, refused=True)
        fleet.clock.advance(60)
        assert fleet.tick() is TickOutcome.SWITCHED
        assert fleet.active() == 2
        assert fleet.switches()[-1].trigger == "at-limit"

    def test_every_account_spent_waits_for_the_earliest_reset(self, fleet):
        fleet.set(1, fable=100, refused=True)
        fleet.set(2, fable=100, weekly_reset=NOW + 4 * HOUR)
        fleet.set(3, fable=100, weekly_reset=NOW + 20 * 60)
        fleet.set(4, fable=100, weekly_reset=NOW + 2 * DAY)
        fleet.read_all()

        assert fleet.tick() is TickOutcome.BLOCKED
        exhausted = [e for e in fleet.events if isinstance(e, AllExhaustedEvent)]
        assert exhausted and exhausted[-1].earliest_reset_at.startswith(
            _iso(NOW + 20 * 60)[:16]
        )
        assert fleet.engine._next_delay(TickOutcome.BLOCKED) <= MAX_SLEEP_S


class TestDeadLogins:
    def test_a_target_whose_login_is_dead_is_quarantined_and_skipped(self, fleet, monkeypatch):
        fleet.set(1, fable=97)
        fleet.set(2, fable=0, weekly_reset=NOW + HOUR * 5)
        fleet.set(3, fable=0)
        fleet.set(4, fable=100)
        fleet.read_all()
        # 2's stored token has to be refreshed before use, and its login is gone.
        expired = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2", "refreshToken": "rt-2",
            "expiresAt": int((fleet.clock.now + 60) * 1000),
        }})
        fleet.switcher._write_account_credentials("2", fleet.email(2), expired)
        monkeypatch.setattr(
            fleet.switcher, "consume_backup_grant",
            lambda *a: oauth.RefreshOutcome(credentials=None, error="invalid_grant"),
        )
        fleet.set(1, fable=99)
        fleet.clock.advance(60)

        fleet.tick()

        assert fleet.active() == 3
        assert any(
            isinstance(e, QuarantineEvent) and e.number == "2" for e in fleet.events
        )


class TestEndingLogins:
    def _login(self, fleet, number: int, left_s: float) -> None:
        creds = json.dumps({"claudeAiOauth": {
            "accessToken": f"sk-{number}", "refreshToken": f"rt-{number}",
            "expiresAt": int((fleet.clock.now + left_s) * 1000),
            "refreshTokenExpiresAt": int((fleet.clock.now + left_s) * 1000),
        }})
        fleet.switcher._write_account_credentials(str(number), fleet.email(number), creds)
        if number == fleet.active():
            (fleet.home / ".claude" / ".credentials.json").write_text(creds)

    def test_an_ending_login_moves_to_one_that_lasts(self, fleet):
        for n in (1, 2, 3, 4):
            fleet.set(n, fable=10)
        self._login(fleet, 1, 10 * 60)
        self._login(fleet, 2, 10 * 60)

        fleet.tick()

        assert fleet.active() in (3, 4)
        assert fleet.switches()[-1].trigger == "login-expiring"

    def test_two_ending_logins_do_not_bounce(self, temp_home, network):
        fleet = Fleet(temp_home, network, n=2, cooldown_seconds=300.0)
        fleet.set(1, fable=10)
        fleet.set(2, fable=10)
        self._login(fleet, 1, 14 * 60)
        self._login(fleet, 2, 14 * 60)

        for _ in range(4):
            fleet.tick()
            fleet.clock.advance(15)

        assert fleet.switches() == []
        assert fleet.active() == 1
        assert "no-target" in fleet.holds()


class TestSessionOwnedAccounts:
    def test_an_account_a_cswap_run_session_owns_is_never_refreshed(
        self, fleet, monkeypatch
    ):
        for n in (1, 2, 3, 4):
            fleet.set(n, fable=10)
        expired = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2", "refreshToken": "rt-2",
            "expiresAt": int((fleet.clock.now - 60) * 1000),
        }})
        fleet.switcher._write_account_credentials("2", fleet.email(2), expired)
        monkeypatch.setattr(
            fleet.switcher, "_live_session_pids",
            lambda num, email: [4242] if num == "2" else [],
        )
        consumed: list[str] = []
        monkeypatch.setattr(
            fleet.switcher, "consume_backup_grant",
            lambda num, *a: consumed.append(num),
        )

        fleet.read_all()
        fleet.set(1, fable=99)
        fleet.clock.advance(60)
        fleet.tick()

        assert "2" not in consumed
        assert fleet.active() != 2


class TestSettingsReload:
    def test_naming_fable_later_switches_the_probe_to_fable(self, temp_home, network):
        fleet = Fleet(temp_home, network, model=None)
        assert fleet.engine.reader.probe_model == oauth.USAGE_PROBE_MODEL

        fleet.engine.apply_settings(fleet.settings.__class__(
            **{**fleet.settings.__dict__, "model": "Fable"}
        ))

        assert fleet.engine.reader.probe_model == oauth.FABLE_PROBE_MODEL


class TestRanking:
    def test_a_refusal_only_reading_is_not_taken_for_ultra(self, fleet):
        # 2's last reading is a refusal that has since ended: no windows at
        # all, which once read as "no 5-hour window", so Ultra.
        fleet.set(1, fable=97)
        fleet.set(2, five_hour=0, fable=0, weekly_reset=NOW + 6 * DAY)
        fleet.set(3, fable=100)
        fleet.set(4, five_hour=0, fable=0, weekly_reset=NOW + DAY)
        fleet.read_all()
        store = fleet.switcher.usage_store()
        ids = fleet.switcher.usage_identities(["2"])
        store.adopt({"2": ({"rejected": {"resets_at": _iso(NOW + 60)}}, 0.0)}, ids)
        fleet.clock.advance(120)
        fleet.set(1, fable=99)

        fleet.tick()

        assert fleet.active() == 4


class TestEndpointOnlyWindows:
    def test_an_opus_line_is_watched_through_the_endpoint(self, temp_home, network):
        fleet = Fleet(temp_home, network, model="Fable,Opus")
        fleet.set(1, fable=10, opus=50)
        for n in (2, 3, 4):
            fleet.set(n, fable=0, opus=0)
        fleet.read_all()
        assert fleet.switches() == []
        fleet.set(1, fable=10, opus=99)

        for _ in range(int(fleet.settings.account_interval_seconds / 15) + 2):
            fleet.clock.advance(15)
            fleet.tick()
            if fleet.switches():
                break

        assert fleet.active() != 1
        assert fleet.switches()[-1].trigger == "threshold"

    def test_a_model_no_account_reports_is_named_once(self, temp_home, network):
        fleet = Fleet(temp_home, network, model="Fabel")
        for n in (1, 2, 3, 4):
            fleet.set(n)
        fleet.read_all()
        fleet.clock.advance(60)
        fleet.tick()

        warnings = [e for e in fleet.events if e.kind == "error" and "Fabel" in e.message]
        assert len(warnings) == 1


class TestDryRun:
    def test_a_dry_run_spends_nothing_on_the_target(self, fleet, network, monkeypatch):
        fleet.engine = fleet.make_engine(dry_run=True)
        fleet.set(1, fable=97)
        for n in (2, 3, 4):
            fleet.set(n, fable=0)
        fleet.read_all()
        consumed = []
        monkeypatch.setattr(
            fleet.switcher, "consume_backup_grant", lambda *a: consumed.append(a)
        )
        network.probe_calls.clear()
        fleet.set(1, fable=99)

        for _ in range(3):
            fleet.clock.advance(20)
            fleet.tick()

        assert consumed == []
        assert set(network.probed()) == {"sk-1"}
        would = [e for e in fleet.events if isinstance(e, SwitchEvent)]
        assert len(would) == 1 and would[0].dry_run
        assert fleet.active() == 1


_HOLDER = """
import os, sys, time
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from claude_swap.locking import InstanceLock
assert InstanceLock(Path(sys.argv[1]) / "autoswitch.lock").try_acquire()
print(os.getpid(), flush=True)
time.sleep(120)
"""


class TestOneEngine:
    def test_a_second_live_engine_refuses_to_run(self, fleet):
        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(fleet.switcher.backup_dir),
             str(Path(oauth.__file__).resolve().parents[1])],
            stdout=subprocess.PIPE, text=True,
        )
        holder_pid = int(holder.stdout.readline().strip())
        try:
            with pytest.raises(EngineBusy) as busy:
                fleet.engine.run_loop()
            assert busy.value.pid == holder_pid
        finally:
            try:
                os.kill(holder_pid, signal.SIGTERM)
            except OSError:
                pass
            holder.kill()
            holder.wait()

    def test_a_restarted_engine_in_the_same_process_takes_over(self, fleet):
        for n in (1, 2, 3, 4):
            fleet.set(n, fable=10)
        first = fleet.make_engine()
        second = fleet.make_engine()
        errors: list[BaseException] = []

        def run(engine):
            try:
                engine.run_loop()
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        t1 = threading.Thread(target=run, args=(first,))
        t1.start()
        lock = InstanceLock(fleet.switcher.backup_dir / LOCK_FILENAME)
        deadline = time.monotonic() + 30
        while not lock.held_elsewhere() and time.monotonic() < deadline:
            time.sleep(0.05)
        t2 = threading.Thread(target=run, args=(second,))
        t2.start()
        time.sleep(0.5)
        first.stop()
        t1.join(timeout=30)
        deadline = time.monotonic() + 30
        while (second._last_decision is None) and time.monotonic() < deadline:
            time.sleep(0.05)
        second.stop()
        t2.join(timeout=30)

        assert errors == []
        assert second._last_decision is not None

    def test_a_dry_run_engine_runs_beside_a_live_one(self, fleet):
        held = InstanceLock(fleet.switcher.backup_dir / LOCK_FILENAME)
        assert held.try_acquire()
        try:
            engine = fleet.make_engine(dry_run=True)
            engine.stop()
            assert engine.run_loop() == 0
        finally:
            held.release()


class TestEvents:
    def test_an_idle_tick_reports_nothing_new(self, fleet):
        for n in (1, 2, 3, 4):
            fleet.set(n, fable=10)
        fleet.read_all()
        fleet.tick()
        fleet.events.clear()
        fleet.clock.advance(1)

        fleet.tick()

        assert fleet.events == []

    def test_a_read_reports_the_windows(self, fleet):
        fleet.set(1, fable=42, seven_day=12)
        for n in (2, 3, 4):
            fleet.set(n)
        fleet.tick()

        poll = next(e for e in fleet.events if isinstance(e, PollEvent))
        assert poll.windows["1"] == {"7d": 12.0, "Fable": 42.0}
        assert "Fable 42% used (switch at 98%)" in poll.human()


def test_a_rejected_window_header_reads_as_spent():
    headers = {
        "anthropic-ratelimit-unified-7d-utilization": "0.5",
        "anthropic-ratelimit-unified-7d-reset": str(int(NOW + DAY)),
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.97",
        "anthropic-ratelimit-unified-7d_oi-status": "rejected",
        "anthropic-ratelimit-unified-7d_oi-reset": str(int(NOW + DAY)),
    }
    usage = oauth.build_usage_result(oauth.parse_unified_ratelimit_headers(headers))

    windows = account_windows(usage, ("Fable",), 95.0, 98.0, NOW)

    assert {w.label: w.pct for w in windows} == {"7d": 50.0, "Fable": 100.0}
