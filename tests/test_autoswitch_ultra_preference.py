"""Where the engine switches, it prefers accounts without a 5-hour window,
and it leaves a login before it ends.

An Ultra account reports only weekly windows, so it runs until its weekly
quota with no 5-hour stop. A login ends at a fixed time (only a fresh
``/login`` replaces it); an account whose login has ended stops working
mid-session. The fleets below are shaped after a real one: Ultra accounts at
0-22% weekly, two 5-hour accounts, and logins ending hours apart.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from claude_swap.autoswitch import (
    LOGIN_HANDOFF_S,
    LOGIN_LANDING_MIN_S,
    NoSwitchEvent,
    SwitchEvent,
    TickOutcome,
)
from tests.test_autoswitch import EngineHarness

HOUR = 3600.0
DAY = 24 * HOUR


def _harness(temp_home, *, accounts=3, **settings) -> EngineHarness:
    h = EngineHarness(temp_home, **settings)
    for number in range(1, accounts + 1):
        h.seed(number, f"account{number}@example.com")
    h.make_live("account1@example.com", 1)
    return h


def _ultra(weekly: float) -> dict:
    return {"seven_day": {"pct": weekly}}


def _five_hour(five: float, weekly: float) -> dict:
    return {"five_hour": {"pct": five}, "seven_day": {"pct": weekly}}


def _credential(h: EngineHarness, number: int, login_left_s: float | None,
                access_left_s: float = HOUR) -> str:
    blob = {
        "accessToken": f"sk-{number}",
        "refreshToken": f"rt-{number}",
        "expiresAt": int((h.clock.now + access_left_s) * 1000),
    }
    if login_left_s is not None:
        blob["refreshTokenExpiresAt"] = int((h.clock.now + login_left_s) * 1000)
    return json.dumps({"claudeAiOauth": blob})


def _set_login(h: EngineHarness, number: int, login_left_s: float | None,
               access_left_s: float = HOUR) -> None:
    """Give a slot (and, for the active one, Claude Code's live credential)
    a login ending ``login_left_s`` from now."""
    creds = _credential(h, number, login_left_s, access_left_s)
    h.switcher._write_account_credentials(
        str(number), f"account{number}@example.com", creds
    )
    if number == 1:
        (h.temp_home / ".claude" / ".credentials.json").write_text(creds)


def _switch(h: EngineHarness) -> SwitchEvent:
    return next(e for e in h.events if isinstance(e, SwitchEvent))


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


class TestUltraRanksFirst:
    def test_at_limit_lands_on_ultra_over_a_roomier_five_hour_account(self, temp_home):
        # 2 has more raw room (90 vs 70), but its 5-hour window stops it
        # within hours; the Ultra account runs until its weekly quota.
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _five_hour(100, 40),
            "2": _five_hour(0, 10),
            "3": _ultra(30),
        })
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "at-limit"
        assert h.active_number() == 3

    def test_failover_lands_on_ultra(self, temp_home):
        # The 14:21 failover in the source log: the dead active account's
        # usage is unreadable, and the worker picked the 5-hour account.
        h = _harness(temp_home, unhealthy_ticks=1)
        outcome = h.tick_with_usage({
            "1": None,
            "2": _five_hour(0, 5),
            "3": _ultra(22),
        })
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "failover"
        assert h.active_number() == 3

    def test_most_room_decides_among_ultra_accounts(self, temp_home):
        h = _harness(temp_home, accounts=4)
        outcome = h.tick_with_usage({
            "1": _five_hour(100, 40),
            "2": _ultra(22),
            "3": _ultra(0),
            "4": _five_hour(0, 0),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_every_account_past_its_lines_ranks_by_room_alone(self, temp_home):
        # Nothing is under its lines: room left is the only thing that buys
        # time, so the roomier 5-hour account beats the nearly spent Ultra.
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _five_hour(99, 50),
            "2": _ultra(97),
            "3": _five_hour(91, 20),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_consume_first_keeps_its_own_order(self, temp_home):
        h = _harness(temp_home, strategy="consume-first")
        h.engine = h._make_engine(dry_run=True)
        outcome = h.tick_with_usage({
            "1": _five_hour(100, 40),
            "2": _five_hour(0, 10),
            "3": _ultra(30),
        })
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).to_ref["number"] == 2


class TestNoSwitchJustToReachUltra:
    """Every switch costs the prompt cache, so a healthy account is never
    left just to reach an Ultra one; the next necessary switch lands there."""

    def test_healthy_five_hour_account_stays_put(self, temp_home):
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _five_hour(10, 20),
            "2": _five_hour(0, 0),
            "3": _ultra(0),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        assert _reasons(h) == ["below-threshold"]

    def test_the_next_necessary_switch_goes_back_to_ultra(self, temp_home):
        # The five-hour account reaches its line: the move it has to make
        # anyway lands on the Ultra account, not the roomier 5-hour peer.
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _five_hour(95, 20),
            "2": _five_hour(0, 0),
            "3": _ultra(40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "proactive"
        assert h.active_number() == 3

    def test_an_ultra_active_account_stays_put(self, temp_home):
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _ultra(50),
            "2": _ultra(0),
            "3": _five_hour(0, 0),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]


class TestLoginExpiry:
    def test_active_login_ending_soon_hands_off_below_its_lines(self, temp_home):
        h = _harness(temp_home)
        _set_login(h, 1, login_left_s=-6 * HOUR, access_left_s=LOGIN_HANDOFF_S - 60)
        outcome = h.tick_with_usage({
            "1": _ultra(20),
            "2": _ultra(10),
            "3": _five_hour(0, 0),
        })
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "login-expiring"
        assert h.active_number() == 2

    def test_a_dead_login_leaves_at_once_instead_of_counting_ticks(self, temp_home):
        # Usage is unreadable because the login is gone: no idle hold, no
        # three unhealthy ticks before failover.
        h = _harness(temp_home)
        _set_login(h, 1, login_left_s=-6 * HOUR, access_left_s=-60)
        outcome = h.tick_with_usage({"1": None, "2": _ultra(10), "3": _ultra(0)})
        assert outcome is TickOutcome.SWITCHED
        assert _switch(h).trigger == "login-expiring"
        assert h.active_number() == 3

    def test_a_login_with_time_left_changes_nothing(self, temp_home):
        h = _harness(temp_home)
        _set_login(h, 1, login_left_s=3 * DAY)
        outcome = h.tick_with_usage({
            "1": _ultra(20), "2": _ultra(0), "3": _ultra(0),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert _reasons(h) == ["below-threshold"]

    def test_an_access_token_outliving_the_login_counts(self, temp_home):
        # The login lapsed hours ago, but the access token already issued
        # works for three more hours: nothing to hand off yet.
        h = _harness(temp_home)
        _set_login(h, 1, login_left_s=-6 * HOUR, access_left_s=3 * HOUR)
        outcome = h.tick_with_usage({
            "1": _ultra(20), "2": _ultra(0), "3": _ultra(0),
        })
        assert outcome is TickOutcome.NO_ACTION

    def test_a_target_whose_login_ends_soon_ranks_last(self, temp_home):
        # The failover in the source log landed on an account whose login
        # ended 70 minutes later.
        h = _harness(temp_home)
        _set_login(h, 2, login_left_s=LOGIN_LANDING_MIN_S - 600)
        _set_login(h, 3, login_left_s=16 * DAY)
        outcome = h.tick_with_usage({
            "1": _ultra(100), "2": _ultra(0), "3": _ultra(60),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_an_escape_still_takes_a_short_login_when_nothing_else_has_room(
        self, temp_home
    ):
        h = _harness(temp_home)
        _set_login(h, 2, login_left_s=LOGIN_LANDING_MIN_S - 600)
        outcome = h.tick_with_usage({
            "1": _ultra(100), "2": _ultra(0), "3": _ultra(100),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_a_voluntary_move_never_picks_a_short_login(self, temp_home):
        # Over its line with room left, so leaving is voluntary: the only
        # target with room has a login about to end, so it holds instead.
        h = _harness(temp_home)
        _set_login(h, 3, login_left_s=LOGIN_LANDING_MIN_S - 600)
        outcome = h.tick_with_usage({
            "1": _ultra(92),
            "2": _ultra(95),
            "3": _ultra(0),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_a_paused_engine_ignores_an_ending_login(self, temp_home):
        h = _harness(temp_home)
        h.engine.apply_settings(replace(h.settings, paused=True))
        _set_login(h, 1, login_left_s=-HOUR, access_left_s=-60)
        assert h.tick_with_usage({"1": None, "2": _ultra(0)}) is TickOutcome.NO_ACTION
        assert h.active_number() == 1
