"""Subscription plan display (#350): fetch, label, expiry, pacing, surfaces.

The three subscription states below are the measured ones from #350; every
classification test is pinned to them rather than to guessed values.
"""

from __future__ import annotations

import dataclasses
import io
import json
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import json_output, subscription
from claude_swap.models import AccountSnapshot

ACTIVE_MAX = {
    "tier": "default_claude_max_20x",
    "billingType": "stripe_subscription",
    "subscriptionStatus": "active",
    "planDisplayName": None,
}
# Cancelled but still inside the paid period: indistinguishable from ACTIVE_MAX,
# which is the whole reason renewal is not shown.
CANCELLED_IN_PERIOD = dict(ACTIVE_MAX)
LAPSED = {
    "tier": "default_claude_ai",
    "billingType": "none",
    "subscriptionStatus": "canceled",
    "planDisplayName": None,
}
TEAM = {
    "tier": "default_claude_max_5x",
    "organizationType": "claude_team",
    "billingType": "stripe_subscription",
    "subscriptionStatus": "active",
    "planDisplayName": None,
}
# Measured on an Enterprise seat (2026-11-07): the profile sends no plan name.
ENTERPRISE = {
    "tier": "default_claude_zero",
    "organizationType": "claude_enterprise",
    "billingType": "stripe_subscription_contracted",
    "subscriptionStatus": "active",
    "planDisplayName": None,
}
# A plan stored before the name was kept: no key at all, unlike a null one.
UNNAMED = {"tier": "default_claude_max_5x"}
# Ultra shares the Max 20x tier string; only the server's plan name separates it.
ULTRA = {
    "tier": "default_claude_max_20x",
    "organizationType": "claude_max",
    "planDisplayName": "Ultra",
}


def _creds(token: str | None = "tok", expires_ms: int | None = None) -> str:
    blob: dict = {}
    if token is not None:
        blob["accessToken"] = token
    if expires_ms is not None:
        blob["expiresAt"] = expires_ms
    return json.dumps({"claudeAiOauth": blob})


def _ok(plan: dict, org: str = "o1", account: str = "") -> subscription.PlanFetch:
    return subscription.PlanFetch(
        plan=plan, account_uuid=account, organization_uuid=org
    )


def _future_ms() -> int:
    return int((time.time() + 3600) * 1000)


def _past_ms() -> int:
    return int((time.time() - 3600) * 1000)


# --- fetch_plan ---------------------------------------------------------------


def _response(payload) -> io.BytesIO:
    body = io.BytesIO(json.dumps(payload).encode())
    body.__enter__ = lambda self=body: self  # type: ignore[attr-defined]
    body.__exit__ = lambda *a: None  # type: ignore[attr-defined]
    return body


@pytest.mark.no_oauth_profile_fake
class TestFetchPlan:
    def test_keeps_the_four_plan_fields_in_camel_case(self):
        payload = {
            "account": {"uuid": "u", "email": "e@x"},
            "organization": {
                "rate_limit_tier": "default_claude_max_5x",
                "organization_type": "claude_team",
                "billing_type": "stripe_subscription",
                "subscription_status": "active",
                "name": "not kept",
                "uuid": "o",
            },
        }
        with patch("urllib.request.urlopen", return_value=_response(payload)):
            result = subscription.fetch_plan("tok")
        assert result.plan == TEAM
        assert (result.account_uuid, result.organization_uuid) == ("u", "o")

    def test_keeps_the_plan_display_name(self):
        payload = {
            "organization": {
                "rate_limit_tier": "default_claude_max_20x",
                "organization_type": "claude_max",
                "plan_display_name": "Ultra",
            }
        }
        with patch("urllib.request.urlopen", return_value=_response(payload)):
            assert subscription.fetch_plan("tok").plan == ULTRA

    def test_no_tier_is_no_answer(self):
        payload = {"organization": {"billing_type": "stripe_subscription"}}
        with patch("urllib.request.urlopen", return_value=_response(payload)):
            assert subscription.fetch_plan("tok").plan is None

    def test_empty_tier_is_no_answer(self):
        payload = {"organization": {"rate_limit_tier": ""}}
        with patch("urllib.request.urlopen", return_value=_response(payload)):
            assert subscription.fetch_plan("tok").plan is None

    def test_non_string_fields_are_dropped_not_kept(self):
        payload = {
            "organization": {
                "rate_limit_tier": "default_claude_ai",
                "subscription_status": None,
                "billing_type": 3,
                "plan_display_name": "",
            }
        }
        with patch("urllib.request.urlopen", return_value=_response(payload)):
            assert subscription.fetch_plan("tok").plan == {
                "tier": "default_claude_ai",
                "planDisplayName": None,
            }

    @pytest.mark.parametrize(
        "error",
        [
            urllib.error.HTTPError("u", 401, "Unauthorized", {}, None),
            urllib.error.URLError("down"),
            TimeoutError("slow"),
        ],
    )
    def test_any_transport_failure_is_none(self, error):
        with patch("urllib.request.urlopen", side_effect=error):
            assert subscription.fetch_plan("tok").plan is None

    def test_retry_after_is_carried_out_of_a_429(self):
        error = urllib.error.HTTPError(
            "u", 429, "Too Many", {"Retry-After": "3600"}, None
        )
        with patch("urllib.request.urlopen", side_effect=error):
            result = subscription.fetch_plan("tok")
        assert result.plan is None
        assert result.retry_after_s == 3600.0

    def test_malformed_body_is_none(self):
        body = io.BytesIO(b"not json")
        body.__enter__ = lambda self=body: self  # type: ignore[attr-defined]
        body.__exit__ = lambda *a: None  # type: ignore[attr-defined]
        with patch("urllib.request.urlopen", return_value=body):
            assert subscription.fetch_plan("tok").plan is None

    def test_non_dict_organization_is_none(self):
        with patch(
            "urllib.request.urlopen", return_value=_response({"organization": []})
        ):
            assert subscription.fetch_plan("tok").plan is None


# --- label and expiry ---------------------------------------------------------


class TestPlanLabel:
    @pytest.mark.parametrize(
        "plan, label", [(TEAM, "Team"), (ENTERPRISE, "Enterprise")]
    )
    def test_organization_wins_over_the_seat_tier(self, plan, label):
        assert subscription.plan_label(plan) == label
        assert subscription.plan_label({**plan, "planDisplayName": "Ultra"}) == label

    @pytest.mark.parametrize(
        "tier, name, label",
        [
            ("default_claude_max_20x", "Ultra", "Ultra"),
            ("default_claude_max_20x", "", "Max 20x"),
            ("default_claude_max_20x", None, "Max 20x"),
            ("default_claude_max_5x", "Max", "Max 5x"),
            ("default_claude_max_5x", "", "Max 5x"),
            ("default_claude_pro", "Pro", "Pro"),
            (None, "Ultra", "Ultra"),
        ],
    )
    def test_server_name_wins_unless_the_table_is_more_precise(
        self, tier, name, label
    ):
        plan = {**ULTRA, "tier": tier, "planDisplayName": name}
        assert subscription.plan_label(plan) == label

    def test_pro_organization_wins_over_the_default_tier(self):
        # A paying Pro subscription reports the default rate-limit bucket
        # (``default_claude_ai``), the same value a lapsed account falls back
        # to; the tier alone would print "Free" for a subscriber who pays.
        plan = {
            "tier": "default_claude_ai",
            "organizationType": "claude_pro",
            "billingType": "apple_subscription",
        }
        assert subscription.plan_label(plan) == "Pro"

    def test_pro_organization_does_not_mask_a_lapse(self):
        # The label is cosmetic; expiry still comes from is_expired() alone.
        plan = {
            "tier": "default_claude_ai",
            "organizationType": "claude_pro",
            "subscriptionStatus": "canceled",
        }
        assert subscription.is_expired(plan)

    @pytest.mark.parametrize(
        "tier, label",
        [
            ("default_claude_ai", "Free"),
            ("default_claude_max_5x", "Max 5x"),
            ("default_claude_max_20x", "Max 20x"),
        ],
    )
    def test_observed_tiers(self, tier, label):
        assert subscription.plan_label({"tier": tier}) == label

    def test_unobserved_tier_shows_raw_rather_than_a_guess(self):
        assert subscription.plan_label({"tier": "default_claude_pro"}) == (
            "default_claude_pro"
        )

    def test_unknown_organization_type_falls_through_to_the_tier(self):
        plan = {"tier": "default_claude_max_20x", "organizationType": "claude_max"}
        assert subscription.plan_label(plan) == "Max 20x"

    def test_no_plan_no_label(self):
        assert subscription.plan_label(None) is None


class TestIsExpired:
    def test_active_max_is_not_expired(self):
        assert not subscription.is_expired(ACTIVE_MAX)

    def test_cancelled_still_in_period_is_not_expired(self):
        assert not subscription.is_expired(CANCELLED_IN_PERIOD)

    def test_lapsed_is_expired(self):
        assert subscription.is_expired(LAPSED)

    def test_canceled_on_a_paid_tier_is_not_enough(self):
        assert not subscription.is_expired(
            {"tier": "default_claude_max_20x", "subscriptionStatus": "canceled"}
        )

    def test_free_tier_while_active_is_not_enough(self):
        assert not subscription.is_expired(
            {"tier": "default_claude_ai", "subscriptionStatus": "active"}
        )

    def test_no_plan_is_not_expired(self):
        assert not subscription.is_expired(None)


# --- records, due, tokens -----------------------------------------------------


class TestRecords:
    def test_missing_is_due(self):
        assert subscription.is_due(None, now=1000.0)

    @pytest.mark.parametrize("plan, due", [(TEAM, False), (UNNAMED, True)])
    def test_fresh_is_due_only_when_the_name_key_is_missing(self, plan, due):
        record = {"plan": plan, "fetchedAt": 1000.0}
        assert subscription.is_due(record, now=1000.0 + 60) is due

    def test_due_at_ttl(self):
        record = {"plan": TEAM, "fetchedAt": 1000.0}
        assert subscription.is_due(record, now=1000.0 + subscription.PLAN_TTL_S)

    def test_future_stamp_is_clock_skew_and_due(self):
        record = {"plan": TEAM, "fetchedAt": 5000.0}
        assert subscription.is_due(record, now=1000.0)

    @pytest.mark.parametrize(
        "record",
        [
            {"plan": TEAM},
            {"fetchedAt": 1.0},
            {"plan": {"label": "no tier"}, "fetchedAt": 1.0},
            {"plan": TEAM, "fetchedAt": True},
            "not a dict",
        ],
    )
    def test_malformed_record_is_invalid_and_due(self, record):
        assert not subscription.valid_record(record)
        assert subscription.is_due(record, now=1000.0)

    @pytest.mark.parametrize("plan", [TEAM, UNNAMED])
    def test_a_live_retry_holds_an_overdue_record(self, plan):
        record = {"plan": plan, "fetchedAt": 0.0, "retryAt": 2000.0}
        assert not subscription.is_due(record, now=1000.0)

    def test_a_passed_retry_does_not_hold_it(self):
        now = subscription.PLAN_TTL_S + 1000.0
        record = {"plan": TEAM, "fetchedAt": 0.0, "retryAt": now - 1}
        assert subscription.is_due(record, now=now)

    def test_an_impossibly_distant_retry_is_skew(self):
        now = subscription.PLAN_TTL_S + 1000.0
        record = {"plan": TEAM, "fetchedAt": 0.0, "retryAt": 1e12}
        assert subscription.is_due(record, now=now)

    def test_identity_key_separates_organizations(self):
        assert subscription.identity_key("a@x", "org-1") != (
            subscription.identity_key("a@x", "org-2")
        )


class TestMatchesSlot:
    @pytest.mark.parametrize(
        "uuid, org, expected",
        [
            ("a", "o", True),
            ("a", "", True),
            ("", "o", True),
            ("x", "o", False),
            ("a", "x", False),
            ("", "", False),
        ],
    )
    def test_every_recorded_uuid_must_match(self, uuid, org, expected):
        fetched = subscription.PlanFetch(
            plan=TEAM, account_uuid="a", organization_uuid="o"
        )
        assert subscription.matches_slot(fetched, uuid, org) is expected

    def test_a_recorded_uuid_the_response_lacks_does_not_match(self):
        fetched = subscription.PlanFetch(plan=TEAM, organization_uuid="o")
        assert not subscription.matches_slot(fetched, "a", "o")


class TestUsableAccessToken:
    def test_valid_token(self):
        assert subscription.usable_access_token(_creds("t", _future_ms())) == "t"

    def test_expired_token_is_not_usable(self):
        assert subscription.usable_access_token(_creds("t", _past_ms())) is None

    def test_missing_token(self):
        assert subscription.usable_access_token(_creds(None, _future_ms())) is None

    @pytest.mark.parametrize("creds", [None, "", "not json", "{}"])
    def test_garbage(self, creds):
        assert subscription.usable_access_token(creds) is None


# --- store --------------------------------------------------------------------


class TestPlanStore:
    def test_round_trip(self, tmp_path: Path):
        store = subscription.PlanStore(tmp_path / "p.json")
        store.save({"k": {"plan": TEAM, "fetchedAt": 1.0}})
        assert store.load() == {"k": {"plan": TEAM, "fetchedAt": 1.0}}

    def test_save_merges_with_another_writer(self, tmp_path: Path):
        """Two processes refreshing different accounts must both land."""
        path = tmp_path / "p.json"
        first = subscription.PlanStore(path)
        second = subscription.PlanStore(path)
        first.save({"a": {"plan": TEAM, "fetchedAt": 1.0}})
        second.save({"b": {"plan": LAPSED, "fetchedAt": 2.0}})
        assert set(first.load()) == {"a", "b"}

    def test_an_older_answer_cannot_replace_a_newer_plan(self, tmp_path: Path):
        store = subscription.PlanStore(tmp_path / "p.json")
        store.save({"k": {"plan": LAPSED, "fetchedAt": 200.0}})
        store.save({"k": {"plan": ACTIVE_MAX, "fetchedAt": 100.0}})
        assert store.load()["k"]["plan"] == LAPSED

    def test_an_older_failure_cannot_back_off_a_newer_reading(self, tmp_path: Path):
        store = subscription.PlanStore(tmp_path / "p.json")
        store.save({"k": {"plan": TEAM, "fetchedAt": 200.0}})
        store.save({"k": {"retryAt": 900.0, "attemptedAt": 100.0}})
        assert "retryAt" not in store.load()["k"]

    def test_a_failure_keeps_the_plan_it_failed_to_replace(self, tmp_path: Path):
        store = subscription.PlanStore(tmp_path / "p.json")
        store.save({"k": {"plan": TEAM, "fetchedAt": 100.0}})
        store.save({"k": {"retryAt": 900.0, "attemptedAt": 200.0}})
        record = store.load()["k"]
        assert record["plan"] == TEAM
        assert record["fetchedAt"] == 100.0
        assert record["retryAt"] == 900.0

    def test_an_older_failure_cannot_shorten_a_newer_retry(self, tmp_path: Path):
        """Two failures landing out of order: the later attempt's wait stands,
        so a slow timeout cannot cut an hour-long Retry-After to the floor."""
        store = subscription.PlanStore(tmp_path / "p.json", clock=lambda: 1000.0)
        store.save({"k": {"retryAt": 200.0 + 3600.0, "attemptedAt": 200.0}})
        store.save({"k": {"retryAt": 100.0 + 900.0, "attemptedAt": 100.0}})
        assert store.load()["k"] == {"retryAt": 3800.0, "attemptedAt": 200.0}

    def test_a_future_stamped_plan_yields_to_a_fresh_reading(self, tmp_path: Path):
        """The clock was set back after a save, so the stored stamp is ahead
        of ``now``. ``is_due`` asks for a fresh read; the merge must let it
        land, or the old label stands while the endpoint is asked every pass."""
        store = subscription.PlanStore(tmp_path / "p.json", clock=lambda: 1000.0)
        store.save({"k": {"plan": ACTIVE_MAX, "fetchedAt": 5000.0}})
        store.save({"k": {"plan": LAPSED, "fetchedAt": 1000.0}})
        assert store.load()["k"] == {"plan": LAPSED, "fetchedAt": 1000.0}

    def test_a_future_stamped_plan_still_takes_a_failures_backoff(
        self, tmp_path: Path
    ):
        store = subscription.PlanStore(tmp_path / "p.json", clock=lambda: 1000.0)
        store.save({"k": {"plan": ACTIVE_MAX, "fetchedAt": 5000.0}})
        store.save({"k": {"retryAt": 1900.0, "attemptedAt": 1000.0}})
        record = store.load()["k"]
        assert record["plan"] == ACTIVE_MAX
        assert record["retryAt"] == 1900.0
        assert not subscription.is_due(record, 1000.0)

    def test_a_future_stamped_failure_does_not_hold_a_fresh_reading(
        self, tmp_path: Path
    ):
        store = subscription.PlanStore(tmp_path / "p.json", clock=lambda: 1000.0)
        store.save({"k": {"retryAt": 9000.0, "attemptedAt": 5000.0}})
        store.save({"k": {"plan": TEAM, "fetchedAt": 1000.0}})
        assert store.load()["k"] == {"plan": TEAM, "fetchedAt": 1000.0}

    def test_a_record_landing_during_the_lock_wait_is_not_skew(
        self, tmp_path: Path, monkeypatch
    ):
        """Another writer saves a newer plan while this save waits for the
        lock. The skew check must use a clock read under the lock, or that
        record reads as ahead of ``now`` and older data overwrites it."""
        from claude_swap.settings import atomic_write_json

        path = tmp_path / "p.json"
        now = [1000.0]
        store = subscription.PlanStore(path, clock=lambda: now[0])
        real_enter = subscription.FileLock.__enter__
        landed = []

        def enter(lock):
            if not landed:  # the other writer lands while this one waits
                landed.append(True)
                atomic_write_json(path, {"k": {"plan": LAPSED, "fetchedAt": 1005.0}})
                now[0] = 1010.0
            return real_enter(lock)

        monkeypatch.setattr(subscription.FileLock, "__enter__", enter)
        store.save({"k": {"plan": ACTIVE_MAX, "fetchedAt": 990.0}})
        assert store.load()["k"] == {"plan": LAPSED, "fetchedAt": 1005.0}

    def test_writers_cannot_interleave_a_read_merge_write(self, tmp_path: Path):
        import threading

        path = tmp_path / "p.json"

        def write(key: str) -> None:
            subscription.PlanStore(path).save(
                {key: {"plan": TEAM, "fetchedAt": 1.0}}
            )

        threads = [
            threading.Thread(target=write, args=(f"k{i}",)) for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(subscription.PlanStore(path).load()) == 8

    def test_corrupt_file_reads_as_empty(self, tmp_path: Path):
        path = tmp_path / "p.json"
        path.write_text("{ not json")
        assert subscription.PlanStore(path).load() == {}

    def test_missing_file_reads_as_empty(self, tmp_path: Path):
        assert subscription.PlanStore(tmp_path / "absent.json").load() == {}


# --- collector ----------------------------------------------------------------


class _Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def switcher(temp_home):
    from claude_swap.switcher import ClaudeAccountSwitcher

    sw = ClaudeAccountSwitcher()
    sw._setup_directories()
    sw._usage_store.clock = _Clock(1_000_000.0)
    return sw


def _info(num, email, org, active, creds):
    return (num, email, "Org", org, active, creds, "")


class TestCollector:
    def test_fetches_a_due_slot_and_stores_it(self, switcher, monkeypatch):
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        plans = switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        assert calls == ["live"]
        assert plans["1"]["plan"] == TEAM
        assert plans["1"]["fetchedAt"] == 1_000_000.0

    def test_a_record_without_the_name_key_is_refetched_once(
        self, switcher, monkeypatch
    ):
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        key = subscription.identity_key("a@x", "o1")
        switcher._subscription_store().save(
            {key: {"plan": UNNAMED, "fetchedAt": 1_000_000.0}}
        )
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        assert switcher._collect_subscription_plans(info)["1"]["plan"] == TEAM
        switcher._collect_subscription_plans(info)
        assert calls == ["live"]

    def test_a_fresh_record_is_served_without_a_call(self, switcher, monkeypatch):
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(TEAM))
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        switcher._collect_subscription_plans(info)

        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(LAPSED)
        )
        switcher._usage_store.clock.now += 60
        plans = switcher._collect_subscription_plans(info)
        assert calls == []
        assert plans["1"]["plan"] == TEAM

    def test_a_clock_set_back_does_not_pin_the_old_plan(self, switcher, monkeypatch):
        """A record stamped ahead of the clock is due. The fresh reading must
        replace it, and the next pass must not ask again — the alternative is
        a refused reading that stays due: one request per pass, old label."""
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: _ok(ACTIVE_MAX)
        )
        switcher._collect_subscription_plans(info)
        switcher._usage_store.clock.now -= 3600  # the clock is set back an hour

        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(LAPSED)
        )
        plans = switcher._collect_subscription_plans(info)
        assert calls == ["live"]
        assert plans["1"]["plan"] == LAPSED
        plans = switcher._collect_subscription_plans(info)
        assert calls == ["live"], "a reading that landed is not due again"
        assert plans["1"]["plan"] == LAPSED

    def test_a_failed_fetch_keeps_the_reading_and_backs_off(
        self, switcher, monkeypatch
    ):
        """The last good plan and its age survive, and the endpoint is left
        alone until the retry time rather than asked on every snapshot."""
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(TEAM))
        switcher._collect_subscription_plans(info)

        calls = []
        monkeypatch.setattr(
            subscription,
            "fetch_plan",
            lambda t, *a, **k: calls.append(t) or subscription.PlanFetch(),
        )
        switcher._usage_store.clock.now += subscription.PLAN_TTL_S
        plans = switcher._collect_subscription_plans(info)
        assert plans["1"]["plan"] == TEAM
        assert plans["1"]["fetchedAt"] == 1_000_000.0

        switcher._usage_store.clock.now += 60
        switcher._collect_subscription_plans(info)
        assert calls == ["live"], "a failure must back off, not retry every pass"

        switcher._usage_store.clock.now += subscription.FAILURE_RETRY_S
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: _ok(LAPSED)
        )
        plans = switcher._collect_subscription_plans(info)
        assert plans["1"]["plan"] == LAPSED
        assert "retryAt" not in plans["1"]

    def test_retry_after_outlasts_the_default_backoff(self, switcher, monkeypatch):
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        calls = []
        monkeypatch.setattr(
            subscription,
            "fetch_plan",
            lambda t, *a, **k: calls.append(t)
            or subscription.PlanFetch(retry_after_s=7200.0),
        )
        switcher._collect_subscription_plans(info)
        switcher._usage_store.clock.now += subscription.FAILURE_RETRY_S + 1
        switcher._collect_subscription_plans(info)
        assert len(calls) == 1
        switcher._usage_store.clock.now += 7200
        switcher._collect_subscription_plans(info)
        assert len(calls) == 2

    def test_a_zero_retry_after_still_gets_the_floor(self):
        assert subscription.next_retry_at(100.0, 0.0) == (
            100.0 + subscription.FAILURE_RETRY_S
        )

    def test_a_response_for_another_organization_is_not_kept(
        self, switcher, monkeypatch
    ):
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: _ok(LAPSED, org="other")
        )
        plans = switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        assert plans == {}
        record = switcher._subscription_store().load()[
            subscription.identity_key("a@x", "o1")
        ]
        assert "plan" not in record and "retryAt" in record

    def test_a_response_for_another_account_is_not_kept(self, switcher, monkeypatch):
        monkeypatch.setattr(
            switcher,
            "_get_sequence_data",
            lambda: {"accounts": {"1": {"uuid": "mine"}}},
        )
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: _ok(LAPSED, account="theirs")
        )
        plans = switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        assert plans == {}

    def test_the_fallback_read_uses_the_display_cache(self, switcher, monkeypatch):
        """Otherwise every pass repeats a Keychain read per expired slot."""
        seen = []
        monkeypatch.setattr(
            switcher,
            "_read_account_credentials",
            lambda num, email, **k: seen.append(k) or _creds("x", _past_ms()),
        )
        switcher._collect_subscription_plans(
            [_info(2, "b@x", "o1", False, _creds("stale", _past_ms()))]
        )
        assert seen == [{"cached": True}]

    def test_an_empty_fetch_set_reads_stored_plans_only(self, switcher, monkeypatch):
        info = [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(TEAM))
        switcher._collect_subscription_plans(info)
        switcher._usage_store.clock.now += subscription.PLAN_TTL_S + 1
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: pytest.fail("network")
        )
        assert switcher._plans_for_pass(info, set())["1"]["plan"] == TEAM

    def test_a_restricted_fetch_set_still_refreshes(self, switcher, monkeypatch):
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        switcher._plans_for_pass(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))], {"2"}
        )
        assert calls == ["live"]

    def test_inactive_slot_rereads_a_token_the_usage_pass_refreshed(
        self, switcher, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        monkeypatch.setattr(
            switcher,
            "_read_account_credentials",
            lambda num, email, **k: _creds("refreshed", _future_ms()),
        )
        switcher._collect_subscription_plans(
            [_info(2, "b@x", "o1", False, _creds("stale", _past_ms()))]
        )
        assert calls == ["refreshed"]

    def test_no_valid_token_means_no_call_and_no_record(self, switcher, monkeypatch):
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        monkeypatch.setattr(
            switcher,
            "_read_account_credentials",
            lambda num, email, **k: _creds("still-stale", _past_ms()),
        )
        plans = switcher._collect_subscription_plans(
            [_info(2, "b@x", "o1", False, _creds("stale", _past_ms()))]
        )
        assert calls == []
        assert "2" not in plans

    def test_active_slot_never_falls_back_to_the_slot_backup(
        self, switcher, monkeypatch
    ):
        """The live store is the active slot's truth; its backup may be older."""
        read = []
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(TEAM))
        monkeypatch.setattr(
            switcher,
            "_read_account_credentials",
            lambda num, email, **k: read.append(num) or _creds("backup", _future_ms()),
        )
        switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _past_ms()))]
        )
        assert read == []

    def test_a_slot_without_oauth_is_skipped(self, switcher, monkeypatch):
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(TEAM)
        )
        monkeypatch.setattr(
            switcher, "_read_account_credentials", lambda num, email, **k: "sk-ant-api"
        )
        plans = switcher._collect_subscription_plans(
            [_info(3, "k@x", "", False, "sk-ant-api")]
        )
        assert calls == [] and plans == {}

    def test_the_record_follows_the_identity_when_slots_are_renumbered(
        self, switcher, monkeypatch
    ):
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(LAPSED))
        switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: pytest.fail("refetched")
        )
        plans = switcher._collect_subscription_plans(
            [_info(4, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        assert plans["4"]["plan"] == LAPSED

    def test_lands_in_the_switchers_own_backup_dir(self, switcher, monkeypatch):
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(TEAM))
        switcher._collect_subscription_plans(
            [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]
        )
        assert (
            switcher.backup_dir / "cache" / subscription.PLAN_FILENAME
        ).exists()


# --- surfaces -----------------------------------------------------------------


class TestJson:
    def _row(self, **kw):
        return json_output.account_row(1, "a@x", "Org", "o1", True, None, **kw)

    def test_no_plan_emits_no_key(self):
        assert "plan" not in self._row()

    def test_plan_object(self):
        row = self._row(subscription_plan={"plan": LAPSED, "fetchedAt": 0.0})
        assert row["plan"] == {
            "label": "Free",
            "tier": "default_claude_ai",
            "billingType": "none",
            "subscriptionStatus": "canceled",
            "expired": True,
            "fetchedAt": "1970-01-01T00:00:00Z",
        }

    def test_plan_display_name_is_emitted_when_kept(self):
        row = self._row(subscription_plan={"plan": ULTRA, "fetchedAt": 0.0})
        assert (row["plan"]["label"], row["plan"]["planDisplayName"]) == (
            "Ultra",
            "Ultra",
        )

    @pytest.mark.parametrize(
        "plan",
        [
            {"tier": "default_claude_ai"},
            {"tier": "default_claude_ai", "planDisplayName": None},
        ],
    )
    def test_absent_source_fields_are_omitted_not_nulled(self, plan):
        row = self._row(subscription_plan={"plan": plan, "fetchedAt": 0.0})
        assert set(row["plan"]) == {"label", "tier", "expired", "fetchedAt"}


class TestListMarker:
    def test_label(self):
        from claude_swap.switcher import _plan_marker

        marker = _plan_marker({"plan": TEAM, "fetchedAt": 1000.0}, now=1000.0)
        assert "Team" in marker and "expired" not in marker

    def test_lapsed_is_free_without_the_word_expired(self):
        from claude_swap.switcher import _plan_marker

        marker = _plan_marker({"plan": LAPSED, "fetchedAt": 1000.0}, now=1000.0)
        assert "Free" in marker and "expired" not in marker

    def test_age_only_once_stale(self):
        from claude_swap.switcher import _plan_marker

        fresh = _plan_marker({"plan": TEAM, "fetchedAt": time.time()}, now=time.time())
        old_at = time.time() - 3 * subscription.PLAN_TTL_S
        stale = _plan_marker({"plan": TEAM, "fetchedAt": old_at}, now=time.time())
        assert "ago" not in fresh
        assert "ago" in stale

    def test_no_record_no_marker(self):
        from claude_swap.switcher import _plan_marker

        assert _plan_marker(None, now=0.0) == ""


class TestSnapshotField:
    def test_appended_after_policy(self):
        names = [f.name for f in dataclasses.fields(AccountSnapshot)]
        # Immediately after `policy`; fields added since are appended after it,
        # so "last" is no longer the right proxy (same as the policy test).
        assert names[names.index("subscription_plan") - 1] == "policy"

    def test_defaults_to_none(self):
        from claude_swap.usage_store import UsageEntry

        snap = AccountSnapshot(
            number="1",
            email="a@x",
            org_name="",
            org_uuid="",
            is_active=True,
            kind="oauth",
            switchable=True,
            usage=UsageEntry(),
        )
        assert snap.subscription_plan is None


def _snap(**kw) -> AccountSnapshot:
    from claude_swap.usage_store import UsageEntry

    base = {
        "number": "1",
        "email": "a@x",
        "org_name": "",
        "org_uuid": "",
        "is_active": True,
        "kind": "oauth",
        "switchable": True,
        "usage": UsageEntry(),
    }
    base.update(kw)
    return AccountSnapshot(**base)


class _Palette:
    muted = "dim"
    sev_warn = "yellow"


class TestTuiMarker:
    def _render(self, record) -> str:
        from rich.text import Text

        from claude_swap.tui.widgets import _append_plan

        text = Text()
        _append_plan(text, _snap(subscription_plan=record), "  ", _Palette())
        return text.plain

    def test_label(self):
        assert self._render({"plan": TEAM, "fetchedAt": time.time()}) == "  · Team"

    def test_lapsed_is_a_highlighted_free_with_no_extra_words(self):
        from rich.text import Text

        from claude_swap.tui.widgets import _append_plan

        text = Text()
        record = {"plan": LAPSED, "fetchedAt": time.time()}
        _append_plan(text, _snap(subscription_plan=record), "  ", _Palette())
        assert text.plain == "  · Free"
        assert any(
            span.style == _Palette.sev_warn
            and text.plain[span.start : span.end] == "Free"
            for span in text.spans
        )

    def test_stale_age_uses_the_same_wording_as_list(self):
        """Two TTLs old is the first stale moment, worded as list words it."""
        old_at = time.time() - 2 * subscription.PLAN_TTL_S - 60
        rendered = self._render({"plan": TEAM, "fetchedAt": old_at})
        assert "(2h ago)" in rendered

    def test_nothing_without_a_record(self):
        assert self._render(None) == ""


class TestSnapshotSourceRefresh:
    def _source(self, refresh):
        from claude_swap.snapshot_source import SnapshotSource

        class Switcher:
            refresh_subscription_plans = staticmethod(refresh)

        return SnapshotSource(Switcher())

    def test_overlapping_refreshes_collapse(self):
        calls = []
        holder = {}

        def refresh():
            calls.append(1)
            holder["source"].refresh_plans()

        holder["source"] = self._source(refresh)
        holder["source"].refresh_plans()
        assert calls == [1]

    def test_a_failure_goes_no_further_and_releases(self):
        calls = []

        def refresh():
            calls.append(1)
            raise RuntimeError("keychain")

        source = self._source(refresh)
        source.refresh_plans()
        source.refresh_plans()
        assert len(calls) == 2


# --- the 5-hour limit a plan does or does not have ---------------------------
#
# Measured 2026-11-16 across ten accounts: every Ultra one answers
# ``five_hour: null`` with no ``session`` entry in ``limits``, while Max 5x and
# Team answer both. An account upgraded to Ultra keeps being handed the session
# window that was open at the upgrade until that window resets.


MAX_5X = {
    "tier": "default_claude_max_5x",
    "organizationType": "claude_max",
    "subscriptionStatus": "active",
    "planDisplayName": "Max 5x",
}


class TestHasSessionLimit:
    @pytest.mark.parametrize(
        "plan, limited",
        [
            (ULTRA, False),
            ({"tier": "x", "planDisplayName": "ultra"}, False),
            ({"tier": "x", "planDisplayName": " Ultra "}, False),
            (MAX_5X, True),
            (TEAM, True),
            (ENTERPRISE, True),
            (UNNAMED, True),
            (ACTIVE_MAX, True),
            ({"tier": "x", "planDisplayName": "Ultra Pro Deluxe"}, True),
            (None, True),
            ({}, True),
        ],
    )
    def test_only_ultra_has_no_session_limit(self, plan, limited):
        assert subscription.has_session_limit(plan) is limited


class TestReadingForPlan:
    READING = {
        "five_hour": {"pct": 19.0, "resets_at": "2026-11-16T22:40:00+00:00"},
        "seven_day": {"pct": 5.0},
        "scoped": [{"name": "Fable", "pct": 6.0}],
    }

    def test_ultra_is_served_without_the_session_window(self):
        served = subscription.reading_for_plan(self.READING, ULTRA)
        assert "five_hour" not in served
        assert served["seven_day"] == {"pct": 5.0}
        assert served["scoped"] == self.READING["scoped"]

    def test_the_stored_reading_is_not_mutated(self):
        subscription.reading_for_plan(self.READING, ULTRA)
        assert "five_hour" in self.READING

    @pytest.mark.parametrize("plan", [MAX_5X, TEAM, UNNAMED, None])
    def test_every_other_plan_keeps_its_window(self, plan):
        assert subscription.reading_for_plan(self.READING, plan) is self.READING

    def test_a_reading_without_the_window_is_returned_unchanged(self):
        reading = {"seven_day": {"pct": 5.0}}
        assert subscription.reading_for_plan(reading, ULTRA) is reading

    @pytest.mark.parametrize("usage", [None, "re-login needed", 3])
    def test_a_non_reading_is_returned_unchanged(self, usage):
        assert subscription.reading_for_plan(usage, ULTRA) is usage


class TestContradictsPlan:
    def test_a_session_window_on_a_plan_without_one(self):
        assert subscription.contradicts_plan({"five_hour": {"pct": 1.0}}, ULTRA)

    def test_no_session_window_on_a_plan_with_one(self):
        assert subscription.contradicts_plan({"seven_day": {"pct": 1.0}}, MAX_5X)

    @pytest.mark.parametrize(
        "usage, plan",
        [
            ({"five_hour": {"pct": 1.0}}, MAX_5X),
            ({"seven_day": {"pct": 1.0}}, ULTRA),
        ],
    )
    def test_agreement_is_not_a_contradiction(self, usage, plan):
        assert not subscription.contradicts_plan(usage, plan)

    @pytest.mark.parametrize("usage", [None, "re-login needed"])
    def test_no_reading_disputes_nothing(self, usage):
        assert not subscription.contradicts_plan(usage, ULTRA)

    def test_no_record_is_not_disputed(self):
        assert not subscription.contradicts_plan({"five_hour": {"pct": 1.0}}, None)


def _seed_reading(switcher, usage, num="1", email="a@x", org="o1"):
    from claude_swap.usage_store import FetchRecord

    switcher._usage_store.record({num: FetchRecord(usage=usage)}, {num: (email, org)})


class TestServedWindows:
    """The collector hands every surface and the auto engine the same windows."""

    READING = {"five_hour": {"pct": 19.0}, "seven_day": {"pct": 5.0}}

    def _entry(self, switcher, plan):
        info = [_info(1, "a@x", "o1", False, _creds("live", _future_ms()))]
        _seed_reading(switcher, self.READING)
        if plan is not None:
            switcher._subscription_store().save({
                subscription.identity_key("a@x", "o1"): {
                    "plan": plan, "fetchedAt": switcher._usage_store.clock(),
                }
            })
        return switcher._collect_usage_entries(info, fetch=set())["1"]

    def test_ultra_is_served_weekly_only(self, switcher):
        from claude_swap import oauth

        entry = self._entry(switcher, ULTRA)
        assert "five_hour" not in entry.last_good
        assert not oauth.has_five_hour_window(entry.last_good)
        assert [w[0] for w in oauth.relevant_windows(entry.last_good)] == ["7d"]

    @pytest.mark.parametrize("plan", [MAX_5X, TEAM, None])
    def test_a_plan_with_a_session_limit_keeps_it(self, switcher, plan):
        entry = self._entry(switcher, plan)
        assert entry.last_good["five_hour"] == {"pct": 19.0}

    def test_the_stored_row_keeps_the_reading_as_fetched(self, switcher):
        self._entry(switcher, ULTRA)
        stored = switcher._usage_store.entries({"1": ("a@x", "o1")})["1"]
        assert stored.last_good["five_hour"] == {"pct": 19.0}

    def test_a_record_too_old_to_refresh_stops_deciding_limits(self, switcher):
        """A profile endpoint that keeps refusing must not leave an Ultra
        record hiding a 5-hour limit the account has again."""
        info = [_info(1, "a@x", "o1", False, _creds("live", _future_ms()))]
        _seed_reading(switcher, self.READING)
        switcher._subscription_store().save({
            subscription.identity_key("a@x", "o1"): {
                "plan": ULTRA,
                "fetchedAt": switcher._usage_store.clock(),
                # Parked on a retry, so no pass can refresh it.
                "retryAt": switcher._usage_store.clock() + 24 * 3600,
            }
        })
        switcher._usage_store.clock.now += subscription.PLAN_TRUST_MAX_AGE_S

        entry = switcher._collect_usage_entries(info, fetch=set())["1"]
        assert entry.last_good["five_hour"] == {"pct": 19.0}

    def test_a_record_inside_the_trust_age_still_decides(self, switcher):
        info = [_info(1, "a@x", "o1", False, _creds("live", _future_ms()))]
        _seed_reading(switcher, self.READING)
        switcher._subscription_store().save({
            subscription.identity_key("a@x", "o1"): {
                "plan": ULTRA,
                "fetchedAt": switcher._usage_store.clock(),
                "retryAt": switcher._usage_store.clock() + 24 * 3600,
            }
        })
        switcher._usage_store.clock.now += subscription.PLAN_TRUST_MAX_AGE_S - 1

        entry = switcher._collect_usage_entries(info, fetch=set())["1"]
        assert "five_hour" not in entry.last_good


class TestTheCollectingPassKeepsThePlanCurrent:
    """The auto engine reaches no other surface: collecting a reading has to
    be what refreshes the plan that decides the reading's windows."""

    def _info(self):
        return [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]

    def _record(self, switcher, plan):
        switcher._subscription_store().save({
            subscription.identity_key("a@x", "o1"): {
                "plan": plan, "fetchedAt": switcher._usage_store.clock(),
            }
        })

    @pytest.fixture(autouse=True)
    def _no_usage_network(self, switcher, monkeypatch):
        """These passes are about the plan: the usage row is served from the
        store, and nothing here reaches the usage endpoint."""
        monkeypatch.setattr(switcher, "_run_usage_fetches", lambda infos, pre: {})

    def test_a_downgrade_restores_the_window_on_the_engines_own_path(
        self, switcher, monkeypatch
    ):
        self._record(switcher, ULTRA)
        _seed_reading(switcher, {"five_hour": {"pct": 99.0}, "seven_day": {"pct": 5.0}})
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(MAX_5X))
        switcher._usage_store.clock.now += subscription.CONTRADICTED_PLAN_TTL_S

        entry = switcher._collect_usage_entries(self._info())["1"]
        assert entry.last_good["five_hour"] == {"pct": 99.0}

    def test_the_refreshed_record_decides_the_same_pass_that_read_it(
        self, switcher, monkeypatch
    ):
        """No pass may serve a window the plan it just read says is not there."""
        self._record(switcher, MAX_5X)
        _seed_reading(switcher, {"five_hour": {"pct": 19.0}, "seven_day": {"pct": 5.0}})
        monkeypatch.setattr(subscription, "fetch_plan", lambda t, *a, **k: _ok(ULTRA))
        switcher._usage_store.clock.now += subscription.PLAN_TTL_S

        entry = switcher._collect_usage_entries(self._info())["1"]
        assert "five_hour" not in entry.last_good

    def test_a_store_only_pass_makes_no_profile_call(self, switcher, monkeypatch):
        self._record(switcher, ULTRA)
        _seed_reading(switcher, {"five_hour": {"pct": 19.0}})
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(MAX_5X)
        )
        switcher._usage_store.clock.now += 24 * 3600

        switcher._collect_usage_entries(self._info(), fetch=set())
        assert calls == []


class TestContradictedRecordIsReread:
    def _info(self):
        return [_info(1, "a@x", "o1", True, _creds("live", _future_ms()))]

    def _store(self, switcher, plan):
        switcher._subscription_store().save({
            subscription.identity_key("a@x", "o1"): {
                "plan": plan, "fetchedAt": switcher._usage_store.clock(),
            }
        })

    def test_an_ultra_record_whose_account_reports_a_session_window(
        self, switcher, monkeypatch
    ):
        self._store(switcher, ULTRA)
        _seed_reading(switcher, {"five_hour": {"pct": 19.0}, "seven_day": {"pct": 5.0}})
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(MAX_5X)
        )
        switcher._usage_store.clock.now += subscription.CONTRADICTED_PLAN_TTL_S
        plans = switcher._collect_subscription_plans(self._info())
        assert calls == ["live"]
        assert plans["1"]["plan"] == MAX_5X

    def test_a_record_with_a_session_limit_whose_account_reports_none(
        self, switcher, monkeypatch
    ):
        self._store(switcher, MAX_5X)
        _seed_reading(switcher, {"seven_day": {"pct": 5.0}})
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(ULTRA)
        )
        switcher._usage_store.clock.now += subscription.CONTRADICTED_PLAN_TTL_S
        plans = switcher._collect_subscription_plans(self._info())
        assert calls == ["live"]
        assert plans["1"]["plan"] == ULTRA

    def test_an_undisputed_record_waits_the_full_ttl(self, switcher, monkeypatch):
        self._store(switcher, ULTRA)
        _seed_reading(switcher, {"seven_day": {"pct": 5.0}})
        calls = []
        monkeypatch.setattr(
            subscription, "fetch_plan", lambda t, *a, **k: calls.append(t) or _ok(MAX_5X)
        )
        switcher._usage_store.clock.now += subscription.CONTRADICTED_PLAN_TTL_S
        assert switcher._collect_subscription_plans(self._info())["1"]["plan"] == ULTRA
        assert calls == []

    def test_a_contradiction_still_waits_out_a_failures_retry(
        self, switcher, monkeypatch
    ):
        """A disagreement must not turn a failing profile read into one call
        per pass: the retry the failure recorded still governs."""
        self._store(switcher, ULTRA)
        _seed_reading(switcher, {"five_hour": {"pct": 19.0}})
        calls = []
        monkeypatch.setattr(
            subscription,
            "fetch_plan",
            lambda t, *a, **k: calls.append(t) or subscription.PlanFetch(),
        )
        switcher._usage_store.clock.now += subscription.PLAN_TTL_S
        switcher._collect_subscription_plans(self._info())
        assert calls == ["live"], "premise: the read was attempted and failed"

        switcher._usage_store.clock.now += subscription.CONTRADICTED_PLAN_TTL_S
        switcher._collect_subscription_plans(self._info())
        assert calls == ["live"]

        switcher._usage_store.clock.now += subscription.FAILURE_RETRY_S
        switcher._collect_subscription_plans(self._info())
        assert len(calls) == 2
