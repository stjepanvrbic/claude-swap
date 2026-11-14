"""Auto-switch engine: keep Claude Code on one account until it is used up,
then move it to the best account left.

The policy:

- Stay. A switch starts the prompt cache cold on every running session, so
  the active account is kept until one of its windows reaches its switch line
  (``threshold`` for the 5-hour window, ``weekly_threshold`` for the 7-day and
  per-model windows), the server refuses it, or its login is about to end.
- Then move to an account with at least ``min_headroom_pct`` left under each
  of its lines, chosen in this order: higher priority tier; Ultra (no 5-hour
  window, so nothing stops it before its weekly quota) before the rest; the
  weekly window that resets soonest, since quota left at a reset is lost;
  most headroom.
- Confirm the target with a fresh probe just before switching, so stale
  numbers never land Claude Code on a spent account.
- When no account has that much room, keep the active account until the
  server actually refuses it, then take any account with room left. When
  none has any, wait for the earliest reset.
- Unknown usage is never a reason to switch: a failed read keeps the active
  account (only a dead login moves it).

Usage comes from :class:`~claude_swap.usage_reader.UsageReader`: the active
account is probed every ``active_interval_seconds`` (every
``NEAR_LINE_PROBE_S`` once it is within ``NEAR_LINE_PCT`` of a line), and the
other accounts are read from the usage endpoint every
``account_interval_seconds``, one per tick.

Cooldown, quarantine and the last switch persist in
``<backup_root>/autoswitch_state.json``; a live engine holds
``<backup_root>/autoswitch.lock`` so only one runs per data directory.

    engine = AutoSwitchEngine(switcher, settings, on_event=print)
    engine.tick()        # one evaluation, returns a TickOutcome
    engine.run_loop()    # until stop()
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

from claude_swap import oauth
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.json_output import SCHEMA_VERSION, USAGE_RELOGIN_REQUIRED
from claude_swap.lines import binding_window
from claude_swap.locking import FileLock, InstanceLock
from claude_swap.settings import (
    AutoSwitchSettings,
    atomic_write_json,
    parse_model_names,
)
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_reader import UsageReader

STATE_FILENAME = "autoswitch_state.json"
STATE_SCHEMA_VERSION = 1
LOCK_FILENAME = "autoswitch.lock"

_logger = logging.getLogger("claude-swap")

WEEK_S = 7 * 24 * 3600.0

# Within this many points of a line the active account is probed every
# NEAR_LINE_PROBE_S: a busy fleet burns a weekly window by about half a
# point a minute, so the switch lands within a fraction of a point.
NEAR_LINE_PCT = 5.0
NEAR_LINE_PROBE_S = 20.0

# Targets confirmed with a probe per tick, at most; the rest wait a tick.
MAX_VERIFY_PER_TICK = 3
# A target whose probe failed is not probed again for this long.
VERIFY_RETRY_S = 300.0

# A target's stored token is refreshed before activation when it expires
# within this, twice Claude Code's own refresh buffer, so a running Claude
# Code finds a fresh token and does not race a refresh of its own.
FRESHEN_BUFFER_MS = 10 * 60 * 1000

# The live token lapses when nothing uses the account. Once it has been
# lapsed this long, the engine asks the switcher to refresh it (under Claude
# Code's own lock); a refresh refused as invalid_grant means the login is
# dead and Claude Code cannot work on it, so the engine moves on.
DEAD_LOGIN_CHECK_S = 600.0

# Longest sleep between ticks while waiting for a reset.
MAX_SLEEP_S = 600.0

# How long a starting engine waits for the lock: another process's holder
# only briefly (a status check takes it for an instant), an earlier engine of
# this same process until its last tick is over.
LOCK_WAIT_S = 3.0
SAME_PROCESS_LOCK_WAIT_S = 180.0

# A login ends at a fixed time (only a fresh /login replaces it). Leave the
# active account this long before, and never land on one ending sooner than
# LOGIN_LANDING_MIN_S unless nothing else has room.
LOGIN_HANDOFF_S = 15 * 60
LOGIN_LANDING_MIN_S = 2 * 3600

# Errors that mean an account's stored login is dead.
_DEAD_ERRORS = ("invalid_grant", "no_refresh_token")

# Triggers that do not wait out the cooldown: the active account cannot
# serve, or is about to stop serving, requests.
_HARD_TRIGGERS = ("at-limit", "failover", "login-expiring")

# How often the background worker re-reads settings while it sleeps, so
# stop/pause/resume from another process apply within seconds.
_CONTROL_POLL_S = 5.0

_SYSTEMIC_MESSAGES = {
    "store-unmirrored": "CLAUDE_SECURESTORAGE_CONFIG_DIR is set — unset it or "
                        "run cswap from a normal shell",
    "invalid_client": "cswap's OAuth client was rejected — systemic, not this "
                      "account",
    "stash-unreadable": "a stashed successor is unreadable — `cswap unclaimed` "
                        "inspects it",
    "consume-busy": "another cswap surface holds the slot — retries next pass",
}


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def pct_label(value: float) -> str:
    """A percentage as configured: 85.555555 stays itself and 99.9 never
    renders as a misleading "100"."""
    return f"{value:.10g}"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoSwitchEvent:
    """Base event. ``to_json()`` payloads are additive: consumers must ignore
    unknown ``event`` kinds and unknown fields."""

    kind: ClassVar[str] = "event"
    ts: str = field(default_factory=_now_iso, kw_only=True)

    def _fields(self) -> dict:
        return {}

    def to_json(self) -> dict:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "event": self.kind,
            "ts": self.ts,
            **self._fields(),
        }

    def human(self) -> str:  # pragma: no cover - overridden
        return self.kind


@dataclass(frozen=True)
class PollEvent(AutoSwitchEvent):
    """What the engine knows after a read: per account, the raw headroom
    (``100 - max pct``, None when unknown), the window percentages, and the
    last read error of accounts with no usable reading."""

    kind: ClassVar[str] = "poll"
    active: dict | None
    headroom: dict[str, float | None]
    threshold: float
    fetch_errors: dict[str, str] = field(default_factory=dict)
    windows: dict[str, dict[str, float]] = field(default_factory=dict)
    weekly_threshold: float | None = None

    def _fields(self) -> dict:
        fields = {
            "active": self.active,
            "headroomPct": self.headroom,
            "threshold": self.threshold,
        }
        if self.fetch_errors:
            fields["fetchErrors"] = self.fetch_errors
        if self.windows:
            fields["windowsPct"] = self.windows
        if self.weekly_threshold is not None:
            fields["weeklyThreshold"] = self.weekly_threshold
        return fields

    def _describe(self, num: str) -> str:
        wins = self.windows.get(num)
        if wins:
            return " · ".join(f"{name} {pct:.0f}%" for name, pct in wins.items())
        h = self.headroom.get(num)
        if h is not None:
            return f"{100 - h:.0f}%"
        err = self.fetch_errors.get(num)
        return f"? ({err})" if err else "?"

    def human(self) -> str:
        if self.active is None:
            return "poll: no active account"
        num = str(self.active.get("number"))
        line = self.threshold
        wins = self.windows.get(num)
        if wins:
            binding = binding_window(
                ((label, pct, None) for label, pct in wins.items()),
                self.threshold,
                self.weekly_threshold,
            )
            used = f"{binding.label} {binding.pct:.0f}% used"
            line = binding.line
        else:
            err = self.fetch_errors.get(num)
            used = f"usage unknown ({err})" if err else "usage unknown"
        others = ", ".join(
            f"#{n}: {self._describe(n)}" for n in self.headroom if n != num
        )
        tail = f" | others: {others}" if others else ""
        return (
            f"Account-{num} ({self.active.get('email')}): {used} "
            f"(switch at {pct_label(line)}%){tail}"
        )


@dataclass(frozen=True)
class SwitchEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "switch"
    # "threshold" | "at-limit" | "failover" | "login-expiring"
    trigger: str
    from_ref: dict | None
    to_ref: dict | None
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False

    def _fields(self) -> dict:
        return {
            "trigger": self.trigger,
            "from": self.from_ref,
            "to": self.to_ref,
            "warnings": self.warnings,
            "dryRun": self.dry_run,
        }

    def human(self) -> str:
        src = (
            f"Account-{self.from_ref.get('number')}" if self.from_ref else "(none)"
        )
        dst = (
            f"Account-{self.to_ref.get('number')} ({self.to_ref.get('email')})"
            if self.to_ref
            else "?"
        )
        prefix = "[dry-run] would switch" if self.dry_run else "Switched"
        return f"{prefix} {src} -> {dst} ({self.trigger})"


@dataclass(frozen=True)
class NoSwitchEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "no-switch"
    reason: str
    detail: str = ""

    def _fields(self) -> dict:
        return {"reason": self.reason, "detail": self.detail}

    def human(self) -> str:
        return f"no switch: {self.reason}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class QuarantineEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "account-quarantined"
    number: str
    email: str
    reason: str

    def _fields(self) -> dict:
        return {"number": self.number, "email": self.email, "reason": self.reason}

    def human(self) -> str:
        return (
            f"Account-{self.number} ({self.email}) quarantined: {self.reason}. "
            f"Log in with it and run 'cswap add --slot {self.number}' to recover."
        )


@dataclass(frozen=True)
class UnquarantineEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "account-unquarantined"
    number: str
    email: str
    reason: str = "credentials-replaced"

    def _fields(self) -> dict:
        return {"number": self.number, "email": self.email, "reason": self.reason}

    def human(self) -> str:
        return f"Account-{self.number} ({self.email}) back in rotation ({self.reason})"


@dataclass(frozen=True)
class AllExhaustedEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "all-exhausted"
    earliest_reset_at: str | None

    def _fields(self) -> dict:
        return {"earliestResetAt": self.earliest_reset_at}

    def human(self) -> str:
        if self.earliest_reset_at:
            return f"all accounts exhausted; earliest reset {self.earliest_reset_at}"
        return "all accounts exhausted; no reset time known"


@dataclass(frozen=True)
class SleepEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "sleep"
    seconds: float
    until: str

    def _fields(self) -> dict:
        return {"seconds": round(self.seconds, 1), "until": self.until}

    def human(self) -> str:
        return f"sleeping {self.seconds / 60:.0f}m (until {self.until})"


@dataclass(frozen=True)
class ErrorEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "error"
    message: str
    transient: bool = True

    def _fields(self) -> dict:
        return {"message": self.message, "transient": self.transient}

    def human(self) -> str:
        return f"error: {self.message}" + (" (will retry)" if self.transient else "")


class TickOutcome(enum.Enum):
    """Outcome of one evaluation tick; values double as --once exit codes."""

    SWITCHED = 0
    ERROR = 1
    NO_ACTION = 2
    BLOCKED = 3  # wanted to switch but no account can take over


class EngineBusy(ClaudeSwitchError):
    """Another live auto-switch engine already runs on this data directory."""

    def __init__(self, pid: int | None):
        self.pid = pid
        where = f" (pid {pid})" if pid else ""
        super().__init__(f"another auto-switch is already running{where}")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One usage window as the engine judges it: its percentage now (0 once
    its reset has passed since the reading), its switch line, and its reset
    time (epoch seconds, None when unknown)."""

    label: str
    pct: float
    line: float
    resets_at: float | None

    @property
    def weekly(self) -> bool:
        return self.label not in (oauth.FIVE_HOUR_LABEL, oauth.REJECTED_LABEL)


def _epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _raw_windows(
    usage: dict | None, models: tuple[str, ...]
) -> list[tuple[str, float, float | None]]:
    """``(label, pct, resets_at epoch)`` for every window that gates the
    account: 5h and 7d when reported, each scoped window named in ``models``
    (``all`` matches every one), and a refusal as a 100% ``rejected``
    window. A reading missing a window simply contributes fewer."""
    if not isinstance(usage, dict):
        return []
    out: list[tuple[str, float, float | None]] = []
    for key, label in (("five_hour", oauth.FIVE_HOUR_LABEL), ("seven_day", "7d")):
        window = usage.get(key)
        if isinstance(window, dict) and isinstance(window.get("pct"), (int, float)):
            out.append((label, float(window["pct"]), _epoch(window.get("resets_at"))))
    wanted = {m.lower() for m in models}
    for scoped in usage.get("scoped") or ():
        if (
            wanted
            and isinstance(scoped, dict)
            and isinstance(scoped.get("pct"), (int, float))
            and isinstance(scoped.get("name"), str)
            and ("all" in wanted or scoped["name"].lower() in wanted)
        ):
            out.append(
                (scoped["name"], float(scoped["pct"]), _epoch(scoped.get("resets_at")))
            )
    rejected = usage.get("rejected")
    if isinstance(rejected, dict):
        out.append((oauth.REJECTED_LABEL, 100.0, _epoch(rejected.get("resets_at"))))
    return [(label, pct, ts) for label, pct, ts in out if math.isfinite(pct)]


def account_windows(
    usage: dict | None,
    models: tuple[str, ...],
    threshold: float,
    weekly_threshold: float | None,
    now: float,
) -> list[Window]:
    """The account's windows judged at ``now``. Usage only grows until a
    window resets, so a reading stays a valid lower bound until then, and a
    window whose reset has passed reads 0% (a refusal past its end, gone)."""
    weekly_line = threshold if weekly_threshold is None else weekly_threshold
    windows: list[Window] = []
    for label, pct, resets_at in _raw_windows(usage, models):
        if resets_at is not None and resets_at <= now:
            if label == oauth.REJECTED_LABEL:
                continue
            pct = 0.0
        line = (
            threshold
            if label in (oauth.FIVE_HOUR_LABEL, oauth.REJECTED_LABEL)
            else weekly_line
        )
        windows.append(Window(label, pct, line, resets_at))
    return windows


def headroom(windows: list[Window]) -> float | None:
    """Points left under the tightest line; None when nothing is known."""
    return min((w.line - w.pct for w in windows), default=None)


def is_over(windows: list[Window]) -> bool:
    return any(w.pct >= w.line for w in windows)


def is_refused(windows: list[Window]) -> bool:
    """At 100% on some window, or refused outright by the server."""
    return any(w.pct >= 100.0 for w in windows)


def weekly_reset_at(windows: list[Window], now: float) -> float | None:
    """When the account's weekly budget next resets. A reset already past
    rolls forward a week (the reading predates it)."""
    resets = [w.resets_at for w in windows if w.weekly and w.resets_at is not None]
    if not resets:
        return None
    soonest = min(resets)
    while soonest <= now:
        soonest += WEEK_S
    return soonest


def usable_again_at(windows: list[Window]) -> float | None:
    """When every window at 100% has reset; None when one has no reset time."""
    times = []
    for w in windows:
        if w.pct >= 100.0:
            if w.resets_at is None:
                return None
            times.append(w.resets_at)
    return max(times, default=None)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountView:
    """One managed account as a tick sees it."""

    number: str
    email: str
    usage: dict | None
    read_at: float | None
    attempted_at: float | None
    error: str | None
    backoff_until: float | None
    dead: bool
    in_rotation: bool
    kind: str
    priority: int
    threshold: float

    def ref(self) -> dict:
        return {"number": int(self.number), "email": self.email}


def probe_model_for(settings: AutoSwitchSettings) -> str:
    """The model the usage probe asks: the configured one, else Fable when a
    Fable window gates switching (only a Fable response reports it), else
    the cheap default."""
    if settings.probe_model:
        return settings.probe_model
    models = {m.lower() for m in parse_model_names(settings.model)}
    if "fable" in models or "all" in models:
        return oauth.FABLE_PROBE_MODEL
    return oauth.USAGE_PROBE_MODEL


class AutoSwitchEngine:
    """The auto-switch policy over a :class:`ClaudeAccountSwitcher`.

    ``on_event`` receives every :class:`AutoSwitchEvent`. ``clock`` is wall
    time (cooldown stamps are persisted across processes). ``reader`` reads
    usage; by default a :class:`UsageReader` on the same switcher.
    """

    def __init__(
        self,
        switcher: ClaudeAccountSwitcher,
        settings: AutoSwitchSettings,
        on_event: Callable[[AutoSwitchEvent], None],
        *,
        dry_run: bool = False,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        reader: UsageReader | None = None,
    ):
        self.switcher = switcher
        self.settings = settings
        self.on_event = on_event
        self.dry_run = dry_run
        self.state_path = state_path or (switcher.backup_dir / STATE_FILENAME)
        self.clock = clock
        self.reader = reader or UsageReader(
            switcher,
            probe_model=probe_model_for(settings),
            hold_s=self._hold_s,
            clock=clock,
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active_number: str | None = None
        self._last_active_attempt: float | None = None
        self._last_account_attempt: dict[str, float] = {}
        self._token_expired_since: float | None = None
        self._last_login_check = 0.0
        self._active_login_dead = False
        self._sleep_until: float | None = None
        self._last_decision: tuple[str, str] | None = None
        self._verify_failed_at: dict[str, float] = {}
        self._last_active_endpoint: float | None = None
        self._model_check_done = False

    @property
    def _models(self) -> tuple[str, ...]:
        return parse_model_names(self.settings.model)

    def _hold_s(self, number: str) -> float:
        """How long a reading keeps other surfaces from fetching the account.
        An inactive account's hold ends just before the engine's own next
        read, which goes through the same collector and would be refused."""
        if number == self._active_number:
            return 2 * self.settings.active_interval_seconds
        return max(self.settings.account_interval_seconds - 60.0, 60.0)

    # -- state file ---------------------------------------------------------

    def _state_lock(self) -> FileLock:
        return FileLock(self.state_path.parent / ".autoswitch_state.lock")

    def _read_state(self) -> dict:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _mutate_state(self, mutator: Callable[[dict], None]) -> dict:
        with self._state_lock():
            state = self._read_state()
            state["schemaVersion"] = STATE_SCHEMA_VERSION
            mutator(state)
            atomic_write_json(self.state_path, state)
            return state

    # -- quarantine ---------------------------------------------------------

    def _quarantine(self, number: str, email: str, reason: str) -> None:
        if self.dry_run:
            return
        creds = self.switcher.read_account_credentials(number, email)
        fingerprint = oauth.credential_fingerprint(creds) if creds else None

        def add(state: dict) -> None:
            state.setdefault("quarantine", {})[number] = {
                "email": email,
                "reason": reason,
                "at": _now_iso(),
                "refreshTokenFingerprint": fingerprint,
            }

        self._mutate_state(add)
        self._emit(QuarantineEvent(number=number, email=email, reason=reason))

    def _release_recovered_quarantines(self, state: dict) -> dict:
        """Drop quarantine entries whose slot was re-added or whose stored
        credential changed since (the user logged in again)."""
        quarantine = state.get("quarantine")
        if not isinstance(quarantine, dict) or not quarantine:
            return state
        released: list[tuple[str, str, str]] = []
        for number, entry in quarantine.items():
            email_now = self.switcher.account_email(number)
            if not email_now or email_now != entry.get("email"):
                released.append((number, entry.get("email", ""), "account-replaced"))
                continue
            creds = self.switcher.read_account_credentials(number, email_now)
            fingerprint = oauth.credential_fingerprint(creds) if creds else None
            if fingerprint != entry.get("refreshTokenFingerprint"):
                released.append((number, email_now, "credentials-replaced"))
        if not released:
            return state

        def drop(s: dict) -> None:
            q = s.get("quarantine")
            if isinstance(q, dict):
                for number, _, _ in released:
                    q.pop(number, None)

        state = self._mutate_state(drop)
        for number, email, reason in released:
            self._emit(UnquarantineEvent(number=number, email=email, reason=reason))
        return state

    # -- snapshot -------------------------------------------------------------

    def _views(self, quarantined: set[str]) -> dict[str, AccountView]:
        identities = self.switcher.usage_identities()
        entries = self.switcher.usage_store().entries(identities, self._models)
        rotating = set(self.switcher.switchable_account_numbers())
        policies = self.switcher.account_policies()
        views: dict[str, AccountView] = {}
        for number, (email, _org) in identities.items():
            entry = entries.get(number)
            dead = number in quarantined
            if entry is not None and entry.auth_dead_strikes and not dead:
                creds = self.switcher.read_account_credentials(number, email)
                dead = entry.token_dead(
                    stored_fp=oauth.credential_fingerprint(creds) if creds else None
                )
            policy = policies.get(number)
            views[number] = AccountView(
                number=number,
                email=email,
                usage=entry.last_good if entry else None,
                read_at=entry.fetched_at if entry else None,
                attempted_at=entry.last_attempt_at if entry else None,
                error=entry.last_error if entry else None,
                backoff_until=entry.backoff_until if entry else None,
                dead=dead,
                in_rotation=number in rotating,
                kind=self.switcher.account_kind_for(number),
                priority=(policy.priority if policy and policy.priority else 0),
                threshold=(
                    policy.threshold
                    if policy and policy.threshold is not None
                    else self.settings.threshold
                ),
            )
        return views

    def _windows(self, view: AccountView, now: float, usage: dict | None = None) -> list[Window]:
        return account_windows(
            view.usage if usage is None else usage,
            self._models,
            view.threshold,
            self.settings.weekly_threshold,
            now,
        )

    # -- reading ----------------------------------------------------------------

    def _read_active(self, view: AccountView, now: float) -> bool:
        """Probe the active account when its reading is due. True when a
        read was attempted."""
        if view.number != self._active_number:
            self._active_number = view.number
            self._last_active_attempt = None
            self._token_expired_since = None
            self._active_login_dead = False
        interval = self.settings.active_interval_seconds
        if any(w.pct >= w.line - NEAR_LINE_PCT for w in self._windows(view, now)):
            interval = min(interval, NEAR_LINE_PROBE_S)
        last = max(
            (t for t in (self._last_active_attempt, view.read_at) if t is not None),
            default=None,
        )
        if last is not None and now - last < interval:
            return False
        self._last_active_attempt = now
        reading = self.reader.read_active(view.number)
        if reading.ok and self._needs_endpoint() and (
            self._last_active_endpoint is None
            or now - self._last_active_endpoint >= self.settings.account_interval_seconds
        ) and not (view.backoff_until is not None and now < view.backoff_until):
            self._last_active_endpoint = now
            self.reader.read_active_endpoint(view.number)
        if reading.error != "token-expired":
            self._token_expired_since = None
            if reading.ok:
                self._active_login_dead = False
            return True
        if self._token_expired_since is None:
            self._token_expired_since = now
        if (
            now - self._token_expired_since >= DEAD_LOGIN_CHECK_S
            and now - self._last_login_check >= DEAD_LOGIN_CHECK_S
        ):
            self._last_login_check = now
            self._active_login_dead = self._active_login_is_dead(view.number)
        return True

    def _needs_endpoint(self) -> bool:
        """Whether a per-model window gates switching that only the usage
        endpoint reports (the probe reports Fable's alone)."""
        return any(m.lower() != "fable" for m in self._models)

    def _check_models(self, views: dict[str, AccountView]) -> None:
        """Once every account has a reading, warn about a configured model
        name that no account reports: it would gate nothing, silently."""
        names = [m for m in self._models if m.lower() != "all"]
        if self._model_check_done or not names:
            return
        readings = [
            v.usage for v in views.values()
            if v.in_rotation and not v.dead and v.kind != "api_key"
        ]
        if not readings or any(u is None for u in readings):
            return
        self._model_check_done = True
        reported = {
            w["name"].lower()
            for u in readings
            for w in u.get("scoped") or ()
            if isinstance(w, dict) and isinstance(w.get("name"), str)
        }
        missing = [m for m in names if m.lower() not in reported]
        if missing:
            self._emit(ErrorEvent(
                message=(
                    f"autoswitch.model {', '.join(missing)} matches no per-model "
                    "window any account reports, so it gates nothing"
                ),
                transient=False,
            ))

    def _active_login_is_dead(self, number: str) -> bool:
        """Let the switcher refresh the lapsed live token under Claude Code's
        lock; True only when the refresh proves the login dead."""
        try:
            entry = self.switcher.usage_entries_by_account(fetch={number}).get(number)
        except ClaudeSwitchError as e:
            _logger.debug("active login check failed: %r", e)
            return False
        if entry is None:
            return False
        return entry.sentinel == USAGE_RELOGIN_REQUIRED or entry.token_dead()

    def _read_one_account(
        self, views: dict[str, AccountView], current: str, now: float
    ) -> bool:
        """Read the inactive account whose reading is most overdue, if any.
        One per tick keeps the endpoint traffic flat however many accounts
        there are. A window reset since the last reading makes it due."""
        interval = self.settings.account_interval_seconds
        due: list[tuple[float, str]] = []
        for view in views.values():
            if view.number == current or view.dead or view.kind == "api_key":
                continue
            if view.backoff_until is not None and now < view.backoff_until:
                continue
            last = max(
                (
                    t
                    for t in (
                        view.read_at,
                        view.attempted_at,
                        self._last_account_attempt.get(view.number),
                    )
                    if t is not None
                ),
                default=None,
            )
            reset_since = view.read_at is not None and any(
                ts is not None and view.read_at < ts <= now
                for _, _, ts in _raw_windows(view.usage, self._models)
            )
            if last is None or now - last >= interval or (reset_since and now - last >= 60):
                due.append((last or 0.0, view.number))
        if not due:
            return False
        _, number = min(due)
        self._last_account_attempt[number] = now
        self.reader.read_account(number)
        return True

    # -- tick -----------------------------------------------------------------

    def tick(self) -> TickOutcome:
        """Evaluate once: read what is due, maybe switch. Never raises."""
        try:
            return self._tick_inner()
        except ClaudeSwitchError as e:
            self._emit(ErrorEvent(message=str(e), transient=True))
            return TickOutcome.ERROR
        except Exception as e:  # pragma: no cover - safety net
            _logger.debug("auto-switch tick failed", exc_info=True)
            self._emit(ErrorEvent(message=f"{type(e).__name__}: {e}", transient=True))
            return TickOutcome.ERROR

    def _tick_inner(self) -> TickOutcome:
        self._sleep_until = None
        settings = self.settings
        if settings.paused:
            self._decide_hold("paused", "resume with 'cswap auto resume'", False)
            return TickOutcome.NO_ACTION

        state = self._read_state()
        if not self.dry_run:
            state = self._release_recovered_quarantines(state)
        quarantine = state.get("quarantine")
        quarantined = set(quarantine) if isinstance(quarantine, dict) else set()

        current = self.switcher.current_account_number()
        if current is None:
            if self.switcher.has_live_login():
                self._decide_hold(
                    "unmanaged-active-account",
                    "run 'cswap add' to include it in rotation",
                    False,
                )
            else:
                self._decide_hold(
                    "no-active-account", "log in and run 'cswap add' first", False
                )
            return TickOutcome.NO_ACTION

        now = self.clock()
        views = self._views(quarantined)
        active = views.get(current)
        if active is None:
            self._decide_hold("no-active-account", "", False)
            return TickOutcome.NO_ACTION
        read = self._read_active(active, now)
        read = self._read_one_account(views, current, now) or read
        if read:
            views = self._views(quarantined)
            active = views[current]
            self._emit_poll(views, active, now)
            self._check_models(views)

        if active.kind == "api_key" and not settings.include_api_key_accounts:
            self._decide_hold("active-api-key", "API-key accounts have no quota to watch", read)
            return TickOutcome.NO_ACTION

        trigger = self._trigger(active, now, read)
        if trigger is None:
            return TickOutcome.NO_ACTION
        return self._switch_away(active, trigger, views, now, quarantined, read)

    def _trigger(self, active: AccountView, now: float, read: bool) -> str | None:
        """Why the active account has to go, or None (and the hold reported)."""
        login_left = self._login_left(active.number, active=True)
        if login_left is not None and login_left <= LOGIN_HANDOFF_S:
            return "login-expiring"
        windows = self._windows(active, now)
        if windows and is_refused(windows):
            return "at-limit"
        if windows and is_over(windows):
            return "threshold"
        fresh = (
            active.read_at is not None
            and now - active.read_at <= self._active_trust_s()
        )
        if windows and fresh:
            tightest = min(windows, key=lambda w: w.line - w.pct)
            self._decide_hold(
                "below-threshold",
                f"{tightest.label} {pct_label(tightest.pct)}% < "
                f"{pct_label(tightest.line)}%",
                read,
            )
            return None
        if self._active_login_dead:
            return "failover"
        if self._token_expired_since is not None:
            self._decide_hold(
                "active-idle",
                "Claude Code has not used this login lately; checking again on its next use",
                read,
            )
            return None
        detail = active.error or "no reading yet"
        self._decide_hold("active-usage-unknown", f"{detail}; keeping the account", read)
        return None

    def _active_trust_s(self) -> float:
        """How old the active reading may be for "below its line" to stand."""
        return max(3 * self.settings.active_interval_seconds, 180.0)

    def _login_left(self, number: str, *, active: bool) -> float | None:
        try:
            until = self.switcher.login_usable_until(number, active=active)
        except (ClaudeSwitchError, OSError) as e:
            _logger.debug("login expiry unreadable for account %s: %r", number, e)
            return None
        return None if until is None else until - self.clock()

    # -- choosing a target ----------------------------------------------------

    def _rank_key(self, view: AccountView, now: float) -> tuple:
        windows = self._windows(view, now)
        room = headroom(windows)
        ultra = (
            isinstance(view.usage, dict)
            and isinstance(view.usage.get("seven_day"), dict)
            and not oauth.has_five_hour_window(view.usage)
        )
        reset = weekly_reset_at(windows, now)
        return (
            -view.priority,
            0 if ultra else 1,
            reset if reset is not None else math.inf,
            -(room if room is not None else 0.0),
            int(view.number) if view.number.isdigit() else 0,
        )

    def _switch_away(
        self,
        active: AccountView,
        trigger: str,
        views: dict[str, AccountView],
        now: float,
        quarantined: set[str],
        read: bool,
    ) -> TickOutcome:
        settings = self.settings
        hard = trigger in _HARD_TRIGGERS
        if not hard and self._in_cooldown(self._read_state()):
            self._decide_hold("cooldown", "", read)
            return TickOutcome.NO_ACTION

        candidates = [
            v
            for v in views.values()
            if v.number != active.number
            and v.in_rotation
            and not v.dead
            and v.number not in quarantined
            and v.kind != "api_key"
        ]
        candidates.sort(key=lambda v: self._rank_key(v, now))
        if not candidates and not settings.include_api_key_accounts:
            self._decide_hold("no-candidates", "no other account is in rotation", read)
            return TickOutcome.BLOCKED

        budget = [MAX_VERIFY_PER_TICK]
        min_room = settings.min_headroom_pct
        login_short = {v.number for v in candidates if self._login_short(v.number)}
        roomy = [
            v
            for v in candidates
            if v.number not in login_short
            and ((room := headroom(self._windows(v, now))) is None or room >= min_room)
        ]
        outcome = self._land(roomy, active, trigger, now, budget, min_room)
        if outcome is not None:
            return outcome

        active_refused = is_refused(self._windows(active, now))
        if not (hard or active_refused):
            self._decide_hold(
                "no-roomy-target",
                f"no account has {pct_label(min_room)}% left; staying on "
                f"Account-{active.number} until it is refused",
                read,
            )
            return TickOutcome.BLOCKED

        # The active account cannot serve, or soon cannot: any account with
        # room beats none, the one with the most room first, a login that
        # lasts before one that is ending too. Leaving an ending login for
        # another ending one gains nothing and would bounce straight back.
        fallback = sorted(
            (
                v
                for v in candidates
                if not is_refused(self._windows(v, now))
                and not (trigger == "login-expiring" and v.number in login_short)
            ),
            key=lambda v: (
                v.number in login_short,
                max((w.pct for w in self._windows(v, now)), default=0.0),
            ),
        )
        outcome = self._land(fallback, active, trigger, now, budget, None)
        if outcome is not None:
            return outcome
        if budget[0] <= 0:
            self._decide_hold("verifying", "more targets to confirm next tick", True)
            return TickOutcome.BLOCKED
        if settings.include_api_key_accounts:
            for v in views.values():
                if v.kind == "api_key" and v.in_rotation and v.number != active.number:
                    return self._perform(v, active, trigger)
        if trigger == "login-expiring":
            self._decide_hold(
                "no-target",
                "no account with a login that lasts has room; staying until "
                "this login ends",
                read,
            )
            return TickOutcome.BLOCKED

        earliest = min(
            (
                t
                for v in candidates
                if (t := usable_again_at(self._windows(v, now))) is not None
            ),
            default=None,
        )
        if earliest is not None:
            self._sleep_until = earliest
        self._emit(
            AllExhaustedEvent(
                earliest_reset_at=_iso(earliest) if earliest is not None else None
            )
        )
        self._last_decision = ("all-exhausted", "")
        return TickOutcome.BLOCKED

    def _login_short(self, number: str) -> bool:
        left = self._login_left(number, active=False)
        return left is not None and left <= LOGIN_LANDING_MIN_S

    def _land(
        self,
        targets: list[AccountView],
        active: AccountView,
        trigger: str,
        now: float,
        budget: list[int],
        min_room: float | None,
    ) -> TickOutcome | None:
        """Confirm each target with a fresh probe, best first, and switch to
        the first that has the room. ``min_room`` None means any room below
        100% on every window. None when no target qualified."""
        for view in targets:
            if budget[0] <= 0:
                return None
            if self.dry_run:
                # A dry run never refreshes a stored login or spends a
                # request on an account it is not on: it decides on what it
                # last read.
                if self._qualifies(self._windows(view, now), min_room):
                    return self._perform(view, active, trigger)
                continue
            failed_at = self._verify_failed_at.get(view.number)
            if failed_at is not None and now - failed_at < VERIFY_RETRY_S:
                continue
            token = self._freshen(view)
            if token is None:
                self._verify_failed_at[view.number] = now
                continue
            budget[0] -= 1
            reading = self.reader.probe_account(view.number, token)
            if reading.error in _DEAD_ERRORS:
                self._quarantine(view.number, view.email, reading.error)
                continue
            if not reading.ok:
                self._verify_failed_at[view.number] = now
                continue
            self._verify_failed_at.pop(view.number, None)
            if self._qualifies(self._windows(view, self.clock(), reading.usage), min_room):
                return self._perform(view, active, trigger)
        return None

    @staticmethod
    def _qualifies(windows: list[Window], min_room: float | None) -> bool:
        """Room enough to land: ``min_room`` under every line, or with
        ``min_room`` None, no window at 100%."""
        room = headroom(windows)
        if room is None:
            return False
        if min_room is None:
            return not is_refused(windows)
        return room >= min_room

    def _freshen(self, view: AccountView) -> str | None:
        """The target's access token, refreshed first when it expires soon.
        None when it cannot be used now; a dead or foreign login is
        quarantined on the way."""
        if self.switcher.live_session_pids_for(view.number, view.email):
            # A live `cswap run` session owns this account's token in its own
            # profile; activating it here too would put one rotating refresh
            # token in two places.
            return None
        creds = self.switcher.read_account_credentials(view.number, view.email)
        data = oauth.extract_oauth_data(creds) if creds else None
        if not data:
            self._quarantine(view.number, view.email, "no-credentials")
            return None
        expires_at = data.get("expiresAt")
        if not (
            isinstance(expires_at, (int, float))
            and self.clock() * 1000 + FRESHEN_BUFFER_MS >= expires_at
        ):
            return data.get("accessToken") or None
        outcome = self.switcher.consume_backup_grant(view.number, view.email, creds)
        if outcome.error is None and outcome.credentials:
            if self._identity_conflict(view.number, outcome.token_account):
                self._quarantine(view.number, view.email, "identity-conflict")
                return None
            fresh = oauth.extract_oauth_data(outcome.credentials) or {}
            return fresh.get("accessToken") or None
        if outcome.error in _DEAD_ERRORS:
            self._quarantine(view.number, view.email, outcome.error)
        elif outcome.error in _SYSTEMIC_MESSAGES:
            self._emit(ErrorEvent(message=_SYSTEMIC_MESSAGES[outcome.error]))
        return None

    def _identity_conflict(self, number: str, token_account: dict | None) -> bool:
        """Whether a refreshed credential authenticates as another account
        than the slot records (org first, then account uuid). A slot with
        no uuid yet gets this one."""
        if not isinstance(token_account, dict):
            return False
        uuid = token_account.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            return False
        uuid = uuid.strip()
        identity = self.switcher.account_identity(number)
        org = token_account.get("organizationUuid")
        slot_org = identity.get("organizationUuid") or ""
        if isinstance(org, str) and org and slot_org and org != slot_org:
            return True
        if not identity.get("uuid"):
            try:
                self.switcher.backfill_account_uuid(number, uuid)
            except Exception as e:  # bookkeeping never blocks a switch
                _logger.debug("uuid backfill failed for account %s: %r", number, e)
            return False
        return identity["uuid"] != uuid

    def _perform(
        self, target: AccountView, active: AccountView, trigger: str
    ) -> TickOutcome:
        if self.dry_run:
            if self._last_decision != ("switch", target.number):
                self._emit(
                    SwitchEvent(
                        trigger=trigger,
                        from_ref=active.ref(),
                        to_ref=target.ref(),
                        dry_run=True,
                    )
                )
            self._last_decision = ("switch", target.number)
            return TickOutcome.SWITCHED
        with self._state_lock():
            state = self._read_state()
            if trigger not in _HARD_TRIGGERS and self._in_cooldown(state):
                self._decide_hold("cooldown", "", True)
                return TickOutcome.NO_ACTION
            result = self.switcher.switch_to(target.number, json_output=True)
            if not result or not result.get("switched"):
                self._decide_hold(
                    "already-active", (result or {}).get("reason", ""), True
                )
                return TickOutcome.NO_ACTION
            state["schemaVersion"] = STATE_SCHEMA_VERSION
            state["lastSwitchAt"] = self.clock()
            state["lastSwitchTo"] = target.number
            state["lastSwitchFrom"] = (result.get("from") or {}).get("number")
            state["leftTrigger"] = trigger
            atomic_write_json(self.state_path, state)
        self._emit(
            SwitchEvent(
                trigger=trigger,
                from_ref=result.get("from"),
                to_ref=result.get("to"),
                warnings=result.get("warnings", []),
            )
        )
        self._last_decision = ("switch", target.number)
        return TickOutcome.SWITCHED

    def _in_cooldown(self, state: dict) -> bool:
        last = state.get("lastSwitchAt")
        if not isinstance(last, (int, float)) or isinstance(last, bool):
            return False
        return (self.clock() - last) < self.settings.cooldown_seconds

    # -- events -----------------------------------------------------------------

    def _emit(self, event: AutoSwitchEvent) -> None:
        self.on_event(event)

    def _decide_hold(self, reason: str, detail: str, read: bool) -> None:
        """Report a hold when something was read this tick or the decision
        changed; an unchanged decision on an idle tick is not news."""
        if read or self._last_decision != (reason, detail):
            self._emit(NoSwitchEvent(reason=reason, detail=detail))
        self._last_decision = (reason, detail)

    def _emit_poll(
        self, views: dict[str, AccountView], active: AccountView, now: float
    ) -> None:
        windows: dict[str, dict[str, float]] = {}
        room: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for number, view in views.items():
            if view.dead:
                room[number] = None
                errors[number] = "re-login needed"
                continue
            wins = self._windows(view, now)
            room[number] = (100.0 - max(w.pct for w in wins)) if wins else None
            if wins:
                windows[number] = {w.label: w.pct for w in wins}
            elif view.error:
                errors[number] = view.error
        self._emit(
            PollEvent(
                active=active.ref(),
                headroom=room,
                threshold=active.threshold,
                weekly_threshold=self.settings.weekly_threshold,
                fetch_errors=errors,
                windows=windows,
            )
        )

    # -- loop -------------------------------------------------------------------

    def stop(self) -> None:
        """Ask ``run_loop`` to exit; wakes it from any sleep. Engines are
        single-use: a stop is never cleared."""
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Cut the current sleep short and tick now."""
        self._wake.set()

    def apply_threshold(self, threshold: float) -> None:
        """Session override from the TUI."""
        self.settings = replace(self.settings, threshold=threshold)

    def apply_settings(self, settings: AutoSwitchSettings) -> None:
        """Adopt reloaded settings mid-run, the probe model included."""
        if settings != self.settings:
            if settings.model != self.settings.model:
                self._model_check_done = False
            self.settings = settings
            self.reader.probe_model = probe_model_for(settings)

    def _next_delay(self, outcome: TickOutcome) -> float:
        interval = self.settings.interval_seconds
        if outcome is TickOutcome.BLOCKED and self._sleep_until is not None:
            return min(max(self._sleep_until - self.clock(), interval), MAX_SLEEP_S)
        return interval

    def run_loop(
        self,
        *,
        settings_loader: Callable[[], AutoSwitchSettings | None] | None = None,
        should_continue: Callable[[AutoSwitchSettings], bool] | None = None,
    ) -> int:
        """Tick until :meth:`stop`; a failing tick never ends it.

        A live engine holds the data directory's auto-switch lock for the
        whole loop and raises :class:`EngineBusy` when another one has it.
        The background worker passes ``settings_loader`` and
        ``should_continue`` so stop/pause/resume, which are settings writes
        from other processes, apply between ticks and during sleeps; a loader
        returning None (file mid-write) keeps the settings in force.
        """
        lock = None
        if not self.dry_run:
            lock = self._take_lock()
            if lock is None:
                return 0
        try:
            while True:
                self._wake.clear()
                if self._stop.is_set():
                    return 0
                if not self._reload_settings(settings_loader, should_continue):
                    return 0
                outcome = self.tick()
                delay = self._next_delay(outcome)
                if delay > self.settings.interval_seconds * 1.5:
                    until = datetime.now(timezone.utc) + timedelta(seconds=delay)
                    self._emit(
                        SleepEvent(
                            seconds=delay,
                            until=until.isoformat(timespec="seconds").replace(
                                "+00:00", "Z"
                            ),
                        )
                    )
                self._sleep(delay, settings_loader, should_continue)
        finally:
            if lock is not None:
                lock.release()

    def _take_lock(self) -> InstanceLock | None:
        """The data directory's engine lock, or None when this engine was
        stopped while waiting. A holder in another process gets a few seconds
        (a status check holds it for an instant); an earlier engine of this
        process (a TUI or menu bar restart) gets until its tick ends."""
        lock = InstanceLock(self.switcher.backup_dir / LOCK_FILENAME)
        started = time.monotonic()
        while not lock.try_acquire(timeout=0.2):
            if self._stop.is_set():
                return None
            holder = lock.holder_pid()
            limit = SAME_PROCESS_LOCK_WAIT_S if holder == os.getpid() else LOCK_WAIT_S
            if time.monotonic() - started >= limit:
                busy = EngineBusy(holder)
                self._emit(ErrorEvent(message=str(busy), transient=False))
                raise busy
        return lock

    def _reload_settings(
        self,
        settings_loader: Callable[[], AutoSwitchSettings | None] | None,
        should_continue: Callable[[AutoSwitchSettings], bool] | None,
    ) -> bool:
        if settings_loader is not None:
            loaded = settings_loader()
            if loaded is not None:
                self.apply_settings(loaded)
        return should_continue is None or should_continue(self.settings)

    def _sleep(
        self,
        delay: float,
        settings_loader: Callable[[], AutoSwitchSettings | None] | None,
        should_continue: Callable[[AutoSwitchSettings], bool] | None,
    ) -> None:
        if settings_loader is None and should_continue is None:
            self._wake.wait(delay)
            return
        deadline = self.clock() + delay
        while not self._stop.is_set():
            remaining = deadline - self.clock()
            if remaining <= 0:
                return
            if self._wake.wait(min(remaining, _CONTROL_POLL_S)):
                return
            if not self._reload_settings(settings_loader, should_continue):
                self.stop()
                return
