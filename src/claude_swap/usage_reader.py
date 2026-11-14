"""Usage readings for the auto-switch engine.

The active account is read from the rate-limit headers of a one-token
request (the probe). Every response carries the account's live 5-hour and
7-day utilization, and a response to a Fable request also carries Fable's own
weekly window. The usage endpoint answers only a few requests per account
before it starts refusing with 429, and every running Claude Code spends that
allowance on the active account itself, so the endpoint is used only for the
other accounts, at a slow pace, and a target is confirmed with a probe right
before a switch.

Every reading is recorded in the usage store, so ``cswap list`` and the TUI
show what the engine decided on, and is held there so those surfaces do not
spend the endpoint allowance on the same account again.

    reader = UsageReader(switcher, probe_model="claude-fable-5-2")
    reader.read_active("2")    # Reading(usage={...}, error=None, at=...)
    reader.read_account("4")   # endpoint read of an inactive account, or None
    reader.probe_account("4", token)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from claude_swap import oauth
from claude_swap.usage_store import FetchRecord

# How long the live access token must still be valid to be used for a probe.
# Claude Code refreshes it on its next request once it is inside its own
# five-minute buffer; a token that close to expiry is still accepted.
_LIVE_TOKEN_MARGIN_MS = 30 * 1000


def _reset_passed(resets_at: object, now: float) -> bool:
    if not isinstance(resets_at, str):
        return False
    try:
        when = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp() <= now


@dataclass(frozen=True)
class Reading:
    """One usage read. ``usage`` is set on success (the shape
    :func:`oauth.build_usage_result` produces, plus ``rejected`` when the
    server refused the probe); ``error`` names the failure otherwise:
    ``token-expired`` (the active login's access token has lapsed),
    ``no-credentials``, ``invalid_grant`` / ``no_refresh_token`` (the stored
    login is dead), or a request failure kind such as ``http-429`` or
    ``network``."""

    usage: dict | None
    error: str | None = None
    retry_after_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.usage is not None


class UsageReader:
    """Reads and records usage for one switcher's accounts.

    ``hold_s(number)`` says how long a successful reading keeps other
    surfaces from fetching that account themselves; it must end before the
    engine's own next read, which the collector would otherwise refuse.
    """

    def __init__(
        self,
        switcher,
        *,
        probe_model: str,
        hold_s: Callable[[str], float] = lambda _number: 120.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.switcher = switcher
        self.probe_model = probe_model
        self.hold_s = hold_s
        self.clock = clock

    def read_active(self, number: str) -> Reading:
        """Probe the active account with Claude Code's live access token.

        Never refreshes it: the live login belongs to Claude Code, which
        refreshes it before its next request. A lapsed token therefore means
        nothing is using the account right now (``token-expired``)."""
        creds = self.switcher.active_credentials()
        data = oauth.extract_oauth_data(creds) if creds else None
        token = data.get("accessToken") if data else None
        if not token:
            return Reading(None, error="no-credentials")
        expires_at = data.get("expiresAt")
        if (
            isinstance(expires_at, (int, float))
            and expires_at <= self.clock() * 1000 + _LIVE_TOKEN_MARGIN_MS
        ):
            return Reading(None, error="token-expired")
        return self.probe_account(number, token)

    def read_account(self, number: str) -> Reading | None:
        """Read an inactive account from the usage endpoint through the
        switcher's own collector, which owns everything about that account's
        token: the consume gate for a lapsed one, read-only use of a live
        ``cswap run`` session's profile, fetch leases against concurrent
        surfaces, backoff and dead-login strikes. None when the collector
        did not fetch it (in backoff, or another surface is fetching it)."""
        started = self.clock()
        entry = self.switcher.usage_entries_by_account(fetch={number}).get(number)
        if entry is None or (entry.last_attempt_at or 0.0) < started:
            return None
        if entry.fetched_at is not None and entry.fetched_at >= started:
            self.switcher.usage_store().adopt(
                {number: (entry.last_good, 0.0)},
                self.switcher.usage_identities([number]),
                hold_s=self.hold_s(number),
            )
            return Reading(entry.last_good)
        return Reading(None, error=entry.last_error or "bad-response")

    def read_active_endpoint(self, number: str) -> Reading:
        """Read the active account from the usage endpoint with Claude Code's
        live token, never refreshing it. Only needed for per-model windows
        the probe does not report."""
        creds = self.switcher.active_credentials()
        data = oauth.extract_oauth_data(creds) if creds else None
        if not data or not data.get("accessToken"):
            return Reading(None, error="no-credentials")
        expires_at = data.get("expiresAt")
        if (
            isinstance(expires_at, (int, float))
            and expires_at <= self.clock() * 1000 + _LIVE_TOKEN_MARGIN_MS
        ):
            return Reading(None, error="token-expired")
        outcome = oauth.try_fetch_usage_for_account(
            number, self.switcher.account_email(number), creds, is_active=True
        )
        error = outcome.error
        if error is None and outcome.usage is None:
            error = "bad-response"
        return self._record(
            number,
            Reading(
                outcome.usage if error is None else None,
                error=error,
                retry_after_s=outcome.retry_after_s,
            ),
        )

    def probe_account(self, number: str, access_token: str) -> Reading:
        """Probe one account with the given access token and record it."""
        outcome = oauth._usage_from_probe(
            access_token, f"for account {number}", self.probe_model
        )
        usage = outcome.usage
        if usage is not None and outcome.error is None:
            usage = self._with_kept_windows(number, usage)
        return self._record(
            number,
            Reading(usage, error=outcome.error, retry_after_s=outcome.retry_after_s),
        )

    def _with_kept_windows(self, number: str, usage: dict) -> dict:
        """The probe reading plus the per-model windows only the endpoint
        reports, taken from the account's stored reading until they reset:
        usage only grows until then, so the older figure is a lower bound."""
        identities = self.switcher.usage_identities([number])
        entry = self.switcher.usage_store().entries(identities).get(number)
        stored = entry.last_good if entry is not None else None
        if not isinstance(stored, dict):
            return usage
        have = {
            w["name"].lower()
            for w in usage.get("scoped") or ()
            if isinstance(w, dict) and isinstance(w.get("name"), str)
        }
        now = self.clock()
        kept = [
            w
            for w in stored.get("scoped") or ()
            if isinstance(w, dict)
            and isinstance(w.get("name"), str)
            and w["name"].lower() not in have
            and not _reset_passed(w.get("resets_at"), now)
        ]
        if not kept:
            return usage
        return {**usage, "scoped": [*(usage.get("scoped") or ()), *kept]}

    def _record(self, number: str, reading: Reading) -> Reading:
        """Store the reading for every surface. ``no-credentials`` and
        ``token-expired`` describe this machine's credential, not the
        account's usage, so they are not recorded as fetch failures."""
        if reading.error in ("no-credentials", "token-expired"):
            return reading
        identities = self.switcher.usage_identities([number])
        if number not in identities:
            return reading
        store = self.switcher.usage_store()
        if reading.ok:
            store.record({number: FetchRecord(usage=reading.usage)}, identities)
            store.adopt(
                {number: (reading.usage, 0.0)}, identities,
                hold_s=self.hold_s(number),
            )
        else:
            store.record(
                {
                    number: FetchRecord(
                        error=reading.error, retry_after_s=reading.retry_after_s
                    )
                },
                identities,
            )
        return reading
