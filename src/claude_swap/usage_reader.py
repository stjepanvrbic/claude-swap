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
    reader.read_account("4")   # endpoint read of an inactive account
    reader.probe_account("4", token)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from claude_swap import oauth
from claude_swap.usage_store import FetchRecord

# How long the live access token must still be valid to be used for a probe.
# Claude Code refreshes it on its next request once it is inside its own
# five-minute buffer; a token that close to expiry is still accepted.
_LIVE_TOKEN_MARGIN_MS = 30 * 1000


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
    struck_fp: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.usage is not None


class UsageReader:
    """Reads and records usage for one switcher's accounts.

    ``hold_s(number)`` says how long a successful reading keeps other
    surfaces from fetching that account themselves.
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

    def read_account(self, number: str) -> Reading:
        """Read an inactive account from the usage endpoint, refreshing its
        stored token through the switcher's consume gate when it has lapsed."""
        email = self.switcher.account_email(number)
        creds = self.switcher.read_account_credentials(number, email) if email else ""
        if not creds:
            return self._record(number, Reading(None, error="no-credentials"))
        outcome = oauth.try_fetch_usage_for_account(
            number,
            email,
            creds,
            is_active=False,
            refresh_via=self.switcher.consume_backup_grant,
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
                struck_fp=outcome.struck_fp,
            ),
        )

    def probe_account(self, number: str, access_token: str) -> Reading:
        """Probe one account with the given access token and record it."""
        outcome = oauth._usage_from_probe(
            access_token, f"for account {number}", self.probe_model
        )
        return self._record(
            number,
            Reading(
                outcome.usage,
                error=outcome.error,
                retry_after_s=outcome.retry_after_s,
            ),
        )

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
                        error=reading.error,
                        retry_after_s=reading.retry_after_s,
                        struck_fp=reading.struck_fp,
                    )
                },
                identities,
            )
        return reading
