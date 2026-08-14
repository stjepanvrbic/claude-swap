"""Auto-switch engine: poll usage, switch accounts before they hit rate limits.

``AutoSwitchEngine`` is UI-agnostic — no printing, no argparse, no TUI
imports. It composes a :class:`ClaudeAccountSwitcher`, evaluates a threshold
policy each :meth:`~AutoSwitchEngine.tick`, and reports everything through
typed events handed to an ``on_event`` callback; the CLI renders them as
human lines or JSONL, and any future frontend (TUI dashboard, menubar) can
consume the same stream.

Policy in one paragraph: a 5h or 7d limit always triggers a handoff. A Fable
limit triggers only when another account can land below its own 5h, 7d, and
Fable limits; otherwise the engine waits for a general limit instead of
cycling an exhausted Fable fleet. There is no load-spreading trigger.

Target selection is Fable-first. Eligible Fable accounts are ranked by their
remaining Fable quota, then general 5h/7d headroom, then slot order. When none
can accept Fable, the account with the most general 5h/7d headroom wins.
Every landing stays below its 5h and 7d limits, so it cannot immediately
re-trigger on a general window.

The switch is proactive: the old account is still valid while a running Claude
Code picks the new one up (this is what makes the macOS ~30s Keychain cache
latency harmless). Before activation the target's token is *freshened*
(refreshed if it expires within 10 minutes — twice Claude Code's refresh
buffer, so a running Claude Code's under-lock re-read sees a fresh token and
aborts its own refresh); a target whose refresh token is dead gets quarantined
instead of activated. When the active account's own usage becomes unreadable
for ``unhealthy_ticks`` consecutive ticks, the engine fails over to any
account that will take traffic.

Request cadence, 429 backoff and escalation are NOT policy and live in
``poll_policy`` + ``usage_store``, shared with every other surface; the engine
only tells the planner which windows and which limit it decides on.

Quarantine state persists in ``<backup_root>/autoswitch_state.json`` and is
mutated read-modify-write under a dedicated file lock.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

from claude_swap import oauth, poll_policy
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.json_output import SCHEMA_VERSION, USAGE_TOKEN_EXPIRED
from claude_swap.locking import FileLock
from claude_swap.poll_policy import (
    ESCALATION_MARGIN_PCT,
    RESET_SLACK_S,
    binding_pct,
)
from claude_swap.settings import AutoSwitchSettings, atomic_write_json, parse_model_names
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import due_candidate, plan_oversleeps_interval

STATE_FILENAME = "autoswitch_state.json"
STATE_SCHEMA_VERSION = 1

_logger = logging.getLogger("claude-swap")

# Systemic freshen refusals, MOST ACTIONABLE FIRST. Deterministic conditions
# that every candidate hits identically, so the tick reports one of them —
# and the order decides which, because reporting the wrong one is how a cause
# needing a human hides behind one that clears itself. store-unmirrored and
# invalid_client stay until somebody unsets an env var or fixes a client
# registration, and stash-unreadable until they unlock a keychain, fix a mode,
# or purge the row; consume-busy is gone by the next pass. stash-unreadable is
# the one that is per-SLOT rather than global, which costs nothing here: this
# message is only ever emitted when NO candidate freshened, so naming the real
# cause of the only slot that had one beats "(network?)".
_SYSTEMIC_MESSAGES = {
    "store-unmirrored": "CLAUDE_SECURESTORAGE_CONFIG_DIR is set — unset it or "
                        "run cswap from a normal shell",
    "invalid_client": "cswap's OAuth client was rejected — systemic, not this "
                      "account",
    "stash-unreadable": "a stashed successor is unreadable — unlock the "
                        "keychain or fix the file, then retry; "
                        "`cswap unclaimed` inspects it",
    "consume-busy": "another cswap surface holds the slot — retries next pass",
}
# Insertion order IS the precedence order, so the remedy and its rank cannot
# drift apart.
_SYSTEMIC_STATUSES = tuple(_SYSTEMIC_MESSAGES)

# Freshen targets whose access token expires within this window: twice Claude
# Code's own 5-minute refresh buffer, so its post-lock "abort refresh if not
# expired" re-read holds with margin after our swap.
FRESHEN_BUFFER_MS = 10 * 60 * 1000

# Sleep caps around a known quota reset (RESET_SLACK_S lives in poll_policy
# with the rest of the cadence numbers). Recheck at the exhausted-account poll
# cadence: providers can grant quota before the previously reported reset, and
# a long engine sleep must not suppress the fetch that discovers it.
MAX_SLEEP_S = poll_policy.EXHAUSTED_INTERVAL_S
NO_RESET_FALLBACK_S = 300.0

# Idle-hold cap (elapsed, not ticks — the hold itself slows the cadence to
# NO_RESET_FALLBACK_S): an owned-and-expired token normally means Claude Code
# is idle and will self-heal on next use, but a *dead* refresh token with an
# active user would look identical forever, so after this long the engine
# falls back to normal unhealthy counting.
IDLE_HOLD_MAX_S = 30 * 60.0

# ---------------------------------------------------------------------------
# Policy: one Fable-first rule over three per-window limits
# ---------------------------------------------------------------------------
#
# Upstream's anti-flap machinery (percentage-point hysteresis, recovery-time
# ranking, the no-return bar) is deliberately absent: all of it existed to
# stop a PROACTIVE load-spreading engine from ping-ponging, and this engine
# has no proactive trigger. The landing gate below is the whole anti-flap
# argument — a handoff only ever moves to an account below its own 5h and 7d
# thresholds, and the account just left is by definition at one of its
# limits, so it cannot win the next tick's ranking until a window rolls over.

# Fable is a BUILT-IN axis of the policy, not an ``autoswitch.model`` opt-in:
# ``fable_threshold`` governs it whether or not the user named the model.
# ``autoswitch.model`` stays what upstream made it — the extra model windows
# folded into manual switch strategies and poll cadence.
FABLE_WINDOW = "Fable"


def policy_models(models: Sequence[str] = ()) -> tuple[str, ...]:
    """The scoped windows the policy reads: ``autoswitch.model``'s, plus Fable.

    Fable is appended because it is a built-in axis, not an opt-in. The ``all``
    sentinel already covers every scoped window an account reports, so nothing
    is added to it.
    """
    lowered = {m.lower() for m in models}
    if "all" in lowered:
        return tuple(models)
    return (*models, *(w for w in (FABLE_WINDOW,) if w.lower() not in lowered))


def policy_limits(settings: AutoSwitchSettings) -> dict[str, float]:
    """Lower-cased window label → utilization pct at which the ACTIVE account
    hands off. The three limits are independent on purpose: the 5h window
    recycles within a session, while 7d and Fable are worth burning nearly dry
    because no switch gets them back.

    Any OTHER per-model weekly window pulled in by ``autoswitch.model`` is
    governed by the 7-day limit — it is a weekly quota, and the weekly limit is
    the one the user set for weekly quotas. Fable keeps its own knob because it
    is the model this fleet actually rations."""
    return {
        "5h": float(settings.five_hour_threshold),
        "7d": float(settings.seven_day_threshold),
        FABLE_WINDOW.lower(): float(settings.fable_threshold),
    }


def window_headrooms(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> dict[str, float]:
    """Window label → points remaining before THAT window reaches ITS limit.

    Keyed by the label as reported, so display keeps the API's spelling.
    Windows the policy does not govern are omitted rather than defaulted:
    an ungoverned window must not be able to trigger a handoff.
    """
    limits = policy_limits(settings)
    weekly = float(settings.seven_day_threshold)
    return {
        label: limits.get(label.lower(), weekly) - pct
        for label, pct, _ in oauth.relevant_windows(usage, policy_models(models))
    }


def policy_headroom(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> float | None:
    """Points before the NEAREST configured limit; ``<= 0`` means hand off now.

    ``None`` when no governed window is readable — the failover signal, and
    the same "unknown, never auto-skipped" contract as
    :func:`oauth.account_headroom`.
    """
    headrooms = window_headrooms(usage, settings, models)
    return min(headrooms.values()) if headrooms else None


def policy_threshold_reached(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> bool:
    """Has the account reached a configured limit on any governed window?"""
    headroom = policy_headroom(usage, settings, models)
    return headroom is not None and headroom <= 0


def general_headroom(usage: dict | None) -> float | None:
    """Account-wide 5h/7d headroom to 100% — the fallback ranking axis, and
    the tie-break on the Fable axis. Deliberately model-free: "general" in the
    policy means the account's own two windows, whatever ``autoswitch.model``
    adds for cadence and manual switching."""
    return oauth.account_headroom(usage)


def fable_headroom(usage: dict | None) -> float | None:
    """Points of Fable quota left (``100 - pct``), or ``None`` when the account
    reports no Fable window at all (a plan without it, or usage unreadable).
    ``<= 0`` means Fable cannot be used on this account right now."""
    for label, pct, _ in oauth.relevant_windows(usage, (FABLE_WINDOW,)):
        if label.lower() == FABLE_WINDOW.lower():
            return 100.0 - pct
    return None


def accepts_traffic(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> bool:
    """Is this account a valid place to land?

    Both account-wide windows must be readable and below their thresholds.
    Fable eligibility is a separate tier because the general fallback remains
    available when every account has exhausted Fable.
    """
    headrooms = window_headrooms(usage, settings, models)
    if "5h" not in headrooms or "7d" not in headrooms:
        return False
    return headrooms["5h"] > 0 and headrooms["7d"] > 0


def accepts_fable(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> bool:
    """Can this account land without immediately switching again?"""
    if not accepts_traffic(usage, settings, models):
        return False
    return window_headrooms(usage, settings, models).get(FABLE_WINDOW, 0.0) > 0


def _policy_candidate_key(
    usage: dict | None,
    settings: AutoSwitchSettings,
    _now: float,
    number: str,
) -> tuple[int, float, float, int] | None:
    """Sort key shared by the engine and its live view."""
    models = parse_model_names(settings.model)
    general = general_headroom(usage)
    if general is None or not accepts_traffic(usage, settings, models):
        return None
    fable = fable_headroom(usage)
    if accepts_fable(usage, settings, models) and fable is not None:
        return (0, -fable, -general, int(number))
    return (1, 0.0, -general, int(number))


def limits_detail(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> str:
    """``5h 42/95%, 7d 61/98%, Fable 12/98%`` — every governed window against
    its limit, for the no-switch/switch reason lines."""
    return ", ".join(
        f"{label} {pct_label(pct)}/{pct_label(limit)}%"
        for label, pct, limit in _governed(usage, settings, models)
    )


def _governed(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str]
) -> list[tuple[str, float, float]]:
    """``(label, pct, limit)`` per governed window, in report order."""
    limits = policy_limits(settings)
    weekly = float(settings.seven_day_threshold)
    return [
        (label, pct, limits.get(label.lower(), weekly))
        for label, pct, _ in oauth.relevant_windows(usage, policy_models(models))
    ]


def reached_windows(
    usage: dict | None, settings: AutoSwitchSettings, models: Sequence[str] = ()
) -> list[str]:
    """Labels of the windows that have reached their limit (drives the switch
    reason: "which limit made us move")."""
    return [
        label
        for label, left in window_headrooms(usage, settings, models).items()
        if left <= 0
    ]


# Adaptive scheduling: the baseline request volume is O(1) per tick — the
# active account plus ONE due candidate (stalest data first) — instead of
# every account in parallel, and the per-account cadence itself (movement,
# threshold distance, urgent mode, 429 recovery) lives in poll_policy, is
# persisted in the usage store by whichever collector fetched, and is shared
# by every surface. The engine escalates to a full candidate refresh only
# when a switch could actually be near: the active account within
# ESCALATION_MARGIN_PCT of its nearest configured limit, or active usage
# unknown (failover needs fresh candidate data). Both triggers imply that
# band, so a decision never runs on the pre-escalation snapshot.


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def pct_label(value: float) -> str:
    """A percentage for display, as configured: 85.555555 stays itself
    (never a rounded "85.5556") and 99.9 never becomes a lying "100" the
    way ``.0f`` renders it. Ten significant digits still absorb IEEE float
    noise (~15th digit) in computed utilizations (100.0 - headroom).
    Displayed comparisons must format BOTH sides with this helper — mixing
    formatters can render an impossible "85.5556% < 85.555555%"."""
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
    kind: ClassVar[str] = "poll"
    active: dict | None  # account_ref shape, or None
    headroom: dict[str, float | None]  # account number → headroom pct (None=unknown)
    threshold: float  # the 5h limit; `limits` carries all three
    # Window label → configured limit pct. Additive: the policy has three
    # independent limits, so a single `threshold` cannot describe it.
    limits: dict[str, float] = field(default_factory=dict)
    # account number → last fetch-error cause ("http-429", "timeout", ...) for
    # accounts whose usage is unknown this tick. Additive field.
    fetch_errors: dict[str, str] = field(default_factory=dict)
    # account number → ordered window label → utilization pct ("5h", "7d",
    # then scoped model display names). Additive field: the binding pct alone
    # (e.g. "89%") hides which window binds — #115 was reported off that
    # ambiguity.
    windows: dict[str, dict[str, float]] = field(default_factory=dict)

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
        num = self.active.get("number")
        h = self.headroom.get(str(num))
        if h is not None:
            used = f"{100 - h:.0f}% used"
        else:
            err = self.fetch_errors.get(str(num))
            used = f"usage unknown ({err})" if err else "usage unknown"
        others = ", ".join(
            f"#{n}: {self._describe(n)}"
            for n in self.headroom
            if n != str(num)
        )
        tail = f" | others: {others}" if others else ""
        return (
            f"Account-{num} ({self.active.get('email')}): {used} "
            f"(switch at {pct_label(self.threshold)}%){tail}"
        )


@dataclass(frozen=True)
class SwitchEvent(AutoSwitchEvent):
    kind: ClassVar[str] = "switch"
    trigger: str  # "at-limit" (a configured limit was reached) | "failover"
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
            f"Log in with it and run 'cswap --add-account --slot {self.number}' "
            "to recover."
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


@dataclass(frozen=True)
class ConfigWarningEvent(AutoSwitchEvent):
    """A configuration value is syntactically fine but provably inert (e.g.
    an ``autoswitch.model`` name no account reports). Not an error: the
    engine keeps running on the axes that do exist."""

    kind: ClassVar[str] = "config-warning"
    message: str

    def _fields(self) -> dict:
        return {"message": self.message}

    def human(self) -> str:
        return f"warning: {self.message}"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TickOutcome(enum.Enum):
    """Outcome of one evaluation tick; values double as --once exit codes."""

    SWITCHED = 0
    ERROR = 1
    NO_ACTION = 2
    BLOCKED = 3  # wanted to switch but no viable target / all exhausted


# Quarantine state persisted fingerprints from a local refresh-token-only
# helper; oauth.credential_fingerprint is identical for refresh-token creds.
# Setup-token quarantines stored None where the shared helper now yields a
# full-content hash — those release once on first recheck and re-quarantine on
# the next dead freshen (one harmless extra cycle, migration only).
_refresh_fingerprint = oauth.credential_fingerprint


def _window_pcts(
    usage: dict | None, models: tuple[str, ...] = ()
) -> dict[str, float]:
    """Ordered window label → pct: "5h", "7d", then configured scoped names.

    Deliberately restricted to the windows the *decision* reads (same
    ``models`` filter): showing an unconfigured scoped window at 100% next
    to a switch onto that account would look like a bug, when the engine
    correctly ignored it. Full per-model usage lives in ``cswap list``.
    """
    return {
        name: pct for name, pct, _ in oauth.relevant_windows(usage, models)
    }


# Reset math moved to poll_policy with the cadence numbers; aliased for the
# engine's sleep scheduling and the test suite.
_limiting_reset_ts = poll_policy.limiting_reset_ts
_earliest_future_reset_ts = poll_policy.earliest_future_reset_ts
_parse_reset_ts = poll_policy.parse_reset_ts


def _poll_threshold(settings: AutoSwitchSettings) -> float:
    """The single threshold handed to the poll planner: the TIGHTEST of the
    three limits.

    The planner compares one threshold against the BINDING (worst) window, so
    only the tightest limit is guaranteed never to plan a poll later than a
    real handoff. Looser choices would let the engine coast past a limit it
    was configured to act on; tighter-than-needed only costs a little extra
    freshness near the band, which the store's own 429 backoff bounds anyway.
    """
    return min(policy_limits(settings).values())


# How often the background worker re-reads settings while sleeping. Control
# commands (stop/pause/resume) are settings writes from another process, so
# this is the worst-case latency for one — not a polling cost that reaches the
# network.
_CONTROL_POLL_S = 5.0


def _ref(number: str, email: str) -> dict:
    return {"number": int(number), "email": email}


def _headroom_by_account(
    usage: dict[str, dict | str | None], models: tuple[str, ...]
) -> dict[str, float | None]:
    """Per-account headroom derived from decision values."""
    return {
        num: oauth.account_headroom(
            value if isinstance(value, dict) else None, models
        )
        for num, value in usage.items()
    }


class AutoSwitchEngine:
    """Threshold-policy auto-switcher over a :class:`ClaudeAccountSwitcher`.

    ``on_event`` receives every :class:`AutoSwitchEvent`; exceptions it raises
    are not caught (a broken frontend should fail loudly in tests). ``clock``
    is wall time (persisted cooldown timestamps must survive processes).
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
    ):
        self.switcher = switcher
        self.settings = settings
        # Model(s) whose per-model weekly limit also binds the switch decision
        # (empty = account-wide 5h/7d only). ``settings.model`` is a comma-
        # separated list ("Fable", "Opus,Sonnet", "all"); parse once here and
        # pass everywhere usage windows are read — decisions, cadence, and
        # reset scheduling must all see the same axes.
        self._models = parse_model_names(settings.model)
        # Cadence must see every window the POLICY can act on — Fable
        # included, whether or not the user named it — so a poll plan can
        # never be slower than the decision it feeds.
        self._poll_models = policy_models(self._models)
        # Poll plans written by the collector must key on the same limits/
        # models the engine decides with (CLI overrides included), not on
        # whatever the settings file happens to say.
        switcher.set_poll_policy_inputs(_poll_threshold(settings), self._poll_models)
        self.on_event = on_event
        self.dry_run = dry_run
        self.state_path = state_path or (switcher.backup_dir / STATE_FILENAME)
        self.clock = clock
        self._stop = threading.Event()
        # Cuts the current inter-tick sleep short (a session threshold change
        # from the TUI should show a fresh decision now, not next interval).
        self._wake = threading.Event()
        self._unhealthy_ticks = 0
        # Both set per tick: a known-reset sleep target, and whether a BLOCKED
        # outcome is static enough (truly exhausted / no candidates) to wait
        # longer than the normal interval.
        self._sleep_until_ts: float | None = None
        self._blocked_wait_long = False
        # Idle-hold: when the active token expired while Claude Code owns it
        # (and is therefore idle), crawl instead of counting unhealthy ticks.
        # ``_idle_hold_since`` survives across ticks (elapsed-time cap);
        # ``_idle_hold_slow`` is per-tick like ``_blocked_wait_long``.
        self._idle_hold_since: float | None = None
        self._idle_hold_slow = False
        # One-shot typo guard for ``autoswitch.model``: resolved (and possibly
        # warned) on the first tick where every relevant account has readable
        # usage — adaptive polling legitimately leaves gaps before that.
        self._model_check_done = not self._models

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
        """Read-modify-write the state file under its lock; returns new state.

        The lock prevents two concurrent engines (loop + cron ``--once``) from
        overwriting each other's quarantine/cooldown updates. Never called
        while any other lock is held.
        """
        with self._state_lock():
            state = self._read_state()
            state["schemaVersion"] = STATE_SCHEMA_VERSION
            mutator(state)
            atomic_write_json(self.state_path, state)
            return state

    # -- quarantine -----------------------------------------------------------

    def _quarantine(self, number: str, email: str, reason: str) -> None:
        creds = self.switcher.read_account_credentials(number, email)
        fingerprint = _refresh_fingerprint(creds) if creds else None

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
        """Drop quarantine entries whose credential was replaced since.

        A changed refresh-token fingerprint (or a removed/re-added slot) means
        the user re-logged in and re-captured the account — the dead lineage
        is gone, so it re-enters rotation.
        """
        quarantine = state.get("quarantine")
        if not isinstance(quarantine, dict) or not quarantine:
            return state
        to_release: list[tuple[str, str, str]] = []
        for number, entry in quarantine.items():
            email_now = self.switcher.account_email(number)
            if not email_now or email_now != entry.get("email"):
                to_release.append(
                    (number, entry.get("email", ""), "account-replaced")
                )
                continue
            creds = self.switcher.read_account_credentials(number, email_now)
            fingerprint = _refresh_fingerprint(creds) if creds else None
            if fingerprint != entry.get("refreshTokenFingerprint"):
                to_release.append((number, email_now, "credentials-replaced"))
        if not to_release:
            return state

        def drop(s: dict) -> None:
            q = s.get("quarantine")
            if isinstance(q, dict):
                for number, _, _ in to_release:
                    q.pop(number, None)

        state = self._mutate_state(drop)
        for number, email, reason in to_release:
            self._emit(UnquarantineEvent(number=number, email=email, reason=reason))
        return state

    # -- freshening -----------------------------------------------------------

    def _freshen_target(self, number: str, email: str) -> str:
        """Ensure a candidate's stored token outlives Claude Code's 5-min
        refresh buffer before it gets activated.

        Returns ``"ok"``, ``"invalid_grant"`` (dead lineage — quarantine),
        ``"identity-conflict"`` (alive but authenticates as a different
        account — quarantine, do not activate), ``"transient"`` (network
        trouble — try again next tick) or ``"skip-live-session"``. Only ever
        touches the slot's *backup* store; the active credential belongs to
        Claude Code.
        """
        if self.switcher.account_kind_for(number) == "api_key":
            return "ok"  # API keys don't expire/refresh
        if self.switcher.live_session_pids_for(number, email):
            # A live `cswap run` session owns this account's token in its own
            # profile. Auto-activating it as the default login too would put
            # one rotating refresh token in two config dirs (the stale-copy
            # failure class) with nobody reading the warning — and its quota
            # is already being consumed by that session anyway. Manual
            # switch_to keeps its warn-and-proceed behavior; auto skips.
            return "skip-live-session"
        creds = self.switcher.read_account_credentials(number, email)
        if not creds:
            return "transient"
        data = oauth.extract_oauth_data(creds)
        if not data:
            return "invalid_grant"
        expires_at = data.get("expiresAt")
        now_ms = self.clock() * 1000
        near_expiry = (
            isinstance(expires_at, (int, float))
            and now_ms + FRESHEN_BUFFER_MS >= expires_at
        )
        if not near_expiry:
            return "ok"
        # The consume gate serializes every backup-rt POST (the recovery
        # branch in `_fetch_active_usage` is a second call site, under the
        # same per-slot consume lock):
        # it re-reads under the slot lock (our snapshot may be superseded),
        # consults the session profile for a newer generation, and persists
        # via fingerprint CAS — so a freshen racing the collector (or a
        # sibling surface) can no longer double-consume one grant.
        outcome = self.switcher.consume_backup_grant(number, email, creds)
        if outcome.error is None and outcome.credentials:
            # The gate already persisted the successor (or adopted a racing
            # writer's newer lineage) under its own lock.
            if self._note_token_identity(number, outcome.token_account):
                # The slot's stored credential authenticates as a *different*
                # account — activating it would put the user on the wrong
                # account with every gauge reading normal. Not a viable
                # target; the caller quarantines it (released automatically
                # once the credential is replaced by a re-add).
                return "identity-conflict"
            return "ok"
        if outcome.error in ("invalid_grant", "no_refresh_token"):
            return "invalid_grant"
        if outcome.error in _SYSTEMIC_STATUSES:
            # Deterministic conditions, not network trouble: every candidate
            # refuses identically and keeps refusing until something outside
            # this process changes — the shell for store-unmirrored (an
            # inherited CLAUDE_SECURESTORAGE_CONFIG_DIR), our OAuth client
            # registration for invalid_client. Reported distinctly so the tick
            # error names the real cause instead of "(network?)", which would
            # send the user to check a connection that is fine.
            return outcome.error
        return "transient"

    def _note_token_identity(
        self, number: str, token_account: dict | None
    ) -> bool:
        """Use the token endpoint's free identity to verify/backfill a slot.

        The refresh grant just ran against the slot's own stored credential,
        so ``token_account`` (when the server includes it) names who that
        credential really is. Returns True on a *conflict*: the credential
        authenticates under a different organization than the slot records
        (org compared first, whenever both sides record one), or as a
        different account uuid. An empty slot uuid (blank-uuid records from
        older versions, add-token placeholders) is backfilled — but only
        when no org conflict exists: a wrong-org credential is evidence the
        slot holds the wrong account, and backfilling *its* uuid would
        poison the slot's identity record (backfill never rewrites a
        non-empty uuid, so that corruption would be sticky).

        ``_parse_token_account`` already enforces a strict boundary, but this
        identity is opportunistic — re-check types here so malformed data can
        never break the freshen that carried it (the successor credential is
        already persisted by the time this runs).
        """
        if not isinstance(token_account, dict):
            return False
        ta_uuid = token_account.get("uuid")
        if not isinstance(ta_uuid, str) or not ta_uuid.strip():
            return False
        ta_uuid = ta_uuid.strip()
        slot_identity = self.switcher.account_identity(number)
        ta_org = token_account.get("organizationUuid")
        slot_org = slot_identity.get("organizationUuid") or ""
        if isinstance(ta_org, str) and ta_org and slot_org and ta_org != slot_org:
            return True
        if not slot_identity.get("uuid"):
            try:
                self.switcher.backfill_account_uuid(number, ta_uuid)
            except Exception as e:  # never let bookkeeping break a freshen
                _logger.debug("uuid backfill failed for account %s: %r", number, e)
            return False
        return slot_identity["uuid"] != ta_uuid

    # -- tick -----------------------------------------------------------------

    def tick(self) -> TickOutcome:
        """Evaluate once: poll usage, maybe switch. Never raises."""
        try:
            return self._tick_inner()
        except ClaudeSwitchError as e:
            self._emit(ErrorEvent(message=str(e), transient=True))
            return TickOutcome.ERROR
        except Exception as e:  # pragma: no cover - safety net
            self._emit(
                ErrorEvent(message=f"{type(e).__name__}: {e}", transient=True)
            )
            return TickOutcome.ERROR

    def _tick_inner(self) -> TickOutcome:
        self._sleep_until_ts = None
        self._blocked_wait_long = False
        self._idle_hold_slow = False
        settings = self.settings
        if settings.paused:
            self._emit(
                NoSwitchEvent(
                    reason="paused",
                    detail="resume with 'cswap auto resume'",
                )
            )
            return TickOutcome.NO_ACTION
        state = self._read_state()
        if not self.dry_run:
            # Dry-run must not write anything, so recovered quarantines are
            # only released (state mutation) on real ticks.
            state = self._release_recovered_quarantines(state)
        quarantined = set(
            state.get("quarantine", {})
            if isinstance(state.get("quarantine"), dict)
            else {}
        )

        current = self.switcher.current_account_number()
        if current is None:
            self._emit(
                PollEvent(
                    active=None,
                    headroom={},
                    threshold=settings.five_hour_threshold,
                    limits=policy_limits(settings),
                )
            )
            if self.switcher.has_live_login():
                # Live login exists but cswap doesn't manage it: never act —
                # a switch would overwrite it without a backup.
                self._emit(
                    NoSwitchEvent(
                        reason="unmanaged-active-account",
                        detail="run 'cswap --add-account' to include it in rotation",
                    )
                )
            else:
                self._emit(
                    NoSwitchEvent(
                        reason="no-active-account",
                        detail="log in and run 'cswap --add-account' first",
                    )
                )
            return TickOutcome.NO_ACTION

        current_email = self.switcher.account_email(current)
        active_ref = _ref(current, current_email) if current_email else {
            "number": int(current),
            "email": "",
        }

        entries, usage, headroom = self._collect_scheduled_usage(
            current, quarantined, settings=settings
        )
        self._emit(
            PollEvent(
                active=active_ref,
                headroom=headroom,
                threshold=settings.five_hour_threshold,
                limits=policy_limits(settings),
                fetch_errors={
                    num: entry.last_error
                    for num, entry in entries.items()
                    if usage.get(num) is None and entry.last_error
                },
                windows={
                    num: pcts
                    for num, value in usage.items()
                    if (pcts := _window_pcts(
                        value if isinstance(value, dict) else None,
                        self._poll_models,
                    ))
                },
            )
        )

        if not self._model_check_done:
            self._check_model_names(quarantined, usage)

        if (
            self.switcher.account_kind_for(current) == "api_key"
            and not settings.include_api_key_accounts
        ):
            self._emit(
                NoSwitchEvent(
                    reason="active-api-key",
                    detail="API-key accounts have no quota to watch",
                )
            )
            return TickOutcome.NO_ACTION

        active_value = usage.get(current)
        active_usage = active_value if isinstance(active_value, dict) else None
        active_headroom = headroom.get(current)
        # THE trigger, and the only one: distance to the nearest of this
        # account's three configured limits. Not `headroom` — that is
        # distance to 100% and is what the ranking and the display use.
        active_distance = policy_headroom(active_usage, settings, self._models)
        if active_distance is not None:
            self._unhealthy_ticks = 0
            self._idle_hold_since = None
            if active_distance > 0:
                self._emit(
                    NoSwitchEvent(
                        reason="below-threshold",
                        detail=limits_detail(active_usage, settings, self._models),
                    )
                )
                return TickOutcome.NO_ACTION
            reached = reached_windows(active_usage, settings, self._models)
            if reached and all(
                label.lower() == FABLE_WINDOW.lower() for label in reached
            ):
                fable_target_exists = any(
                    num != current
                    and num not in quarantined
                    and self.switcher.account_kind_for(num) != "api_key"
                    and accepts_fable(
                        usage.get(num) if isinstance(usage.get(num), dict) else None,
                        settings,
                        self._models,
                    )
                    for num in self.switcher.switchable_account_numbers()
                )
                if not fable_target_exists:
                    self._emit(
                        NoSwitchEvent(
                            reason="fable-unavailable",
                            detail="waiting for a 5h or 7d limit",
                        )
                    )
                    return TickOutcome.NO_ACTION
            # A limit was reached. Label it by how hard: "at-limit" once the
            # account is actually spent (headroom to 100% gone), "proactive"
            # while a configured threshold has been crossed but real quota
            # remains — which is the normal, intended case and the whole point
            # of switching before the hard wall. Labelling only; both take the
            # identical path from here.
            trigger = (
                "at-limit"
                if active_headroom is not None and active_headroom <= 0
                else "proactive"
            )
        else:
            if usage.get(current) == USAGE_TOKEN_EXPIRED:
                # Expired and the refresh could not complete this pass (lock
                # contention, unattributable lineage, failed persist, or the
                # row's failure backoff gating the fetch). The locked-refresh
                # path retries on later passes — no quota burn, nothing to
                # switch for yet; crawl slowly instead of burning failover
                # ticks (Finding 2 of the usage-lapse investigation).
                now = self.clock()
                if self._idle_hold_since is None:
                    self._idle_hold_since = now
                if now - self._idle_hold_since <= IDLE_HOLD_MAX_S:
                    self._unhealthy_ticks = 0
                    self._idle_hold_slow = True
                    self._emit(
                        NoSwitchEvent(
                            reason="active-idle",
                            detail=(
                                "token expired while Claude Code is idle; "
                                "resumes on next use"
                            ),
                        )
                    )
                    return TickOutcome.NO_ACTION
                # Held far longer than any idle nap should need — likely a
                # dead refresh token with an *active* user. Fall through to
                # normal unhealthy counting so failover can still happen.
                _logger.warning(
                    "Active token expired and owned for over %.0f minutes; "
                    "resuming unhealthy counting (dead refresh token?)",
                    IDLE_HOLD_MAX_S / 60,
                )
            else:
                self._idle_hold_since = None
            self._unhealthy_ticks += 1
            if self._unhealthy_ticks < settings.unhealthy_ticks:
                self._emit(
                    NoSwitchEvent(
                        reason="active-usage-unknown",
                        detail=(
                            f"{self._unhealthy_ticks}/{settings.unhealthy_ticks} "
                            "before failover"
                        ),
                    )
                )
                return TickOutcome.NO_ACTION
            trigger = "failover"

        # No cooldown gate: the only trigger is "the active account reached a
        # limit", and delaying that handoff strands the user at a hard limit.
        # `cooldown_seconds` and `hysteresis_pct` are retained as inert
        # settings keys (see AutoSwitchSettings) and read nowhere.

        # -- candidate selection ------------------------------------------
        candidates = [
            num
            for num in self.switcher.switchable_account_numbers()
            if num != current and num not in quarantined
        ]
        oauth_candidates = [
            n for n in candidates if self.switcher.account_kind_for(n) != "api_key"
        ]
        api_key_candidates = (
            [n for n in candidates if self.switcher.account_kind_for(n) == "api_key"]
            if settings.include_api_key_accounts
            else []
        )
        if not oauth_candidates and not api_key_candidates:
            # Won't change until the user adds/recovers an account — no point
            # re-polling at full cadence.
            self._blocked_wait_long = True
            self._emit(NoSwitchEvent(reason="no-candidates"))
            return TickOutcome.BLOCKED

        ordered, any_known = self._rank_candidates(
            oauth_candidates=oauth_candidates, usage=usage, settings=settings
        )
        if not ordered and api_key_candidates:
            # Last resort when we must move: metered API-key accounts, whose
            # headroom is unmeasurable (they bill per token instead).
            ordered = api_key_candidates

        if not ordered:
            if not any_known:
                self._emit(
                    NoSwitchEvent(
                        reason="no-comparison",
                        detail="no candidate has readable usage",
                    )
                )
                return TickOutcome.BLOCKED
            # "All exhausted" (and its bounded reset-aware sleep) only when
            # it is literally true: every candidate is known and at a HARD
            # limit. A candidate merely over a configured threshold is back
            # in play the moment a window rolls over, and the active account
            # can still hit 100% — so those keep the normal cadence.
            candidate_headrooms = [headroom.get(n) for n in oauth_candidates]
            truly_exhausted = all(
                h is not None and h <= 0 for h in candidate_headrooms
            )
            if not truly_exhausted:
                self._emit(
                    NoSwitchEvent(
                        reason="no-qualifying-candidate",
                        detail=(
                            "no candidate is below both its 5h and 7d "
                            "thresholds, or usage is unreadable this tick"
                        ),
                    )
                )
                return TickOutcome.BLOCKED
            self._blocked_wait_long = True
            earliest = self._earliest_recovery(usage)
            if earliest is not None:
                self._sleep_until_ts = earliest.timestamp() + RESET_SLACK_S
            self._emit(
                AllExhaustedEvent(
                    earliest_reset_at=(
                        earliest.isoformat().replace("+00:00", "Z")
                        if earliest
                        else None
                    )
                )
            )
            return TickOutcome.BLOCKED

        # -- freshen + switch ----------------------------------------------
        transient_failure = False
        systemic = ""
        for num in ordered:
            email = self.switcher.account_email(num)
            if self.dry_run:
                # Dry-run stops at the decision: no token refresh, no
                # quarantine writes — freshening is a mutation.
                return self._perform(num, email, trigger)
            status = self._freshen_target(num, email)
            if status == "identity-conflict":
                # The slot's credential is alive but belongs to a different
                # account — switching onto it would silently run the wrong
                # account. Quarantine (auto-released once a re-add replaces
                # the credential).
                self._quarantine(num, email, "identity-conflict")
                continue
            if status == "invalid_grant":
                self._quarantine(num, email, "invalid_grant")
                continue
            if status == "transient":
                transient_failure = True
                continue
            if status in _SYSTEMIC_STATUSES:
                # ONE cause is reported, so it must be the one worth acting
                # on. Assigning unconditionally made it the LAST candidate's,
                # and `consume-busy` clears itself on the next pass while the
                # other two need a human — unset an env var, chase a rejected
                # client_id. So a busy slot sorting after an unmirrored one
                # named the harmless cause and hid the real one: exactly the
                # "reads as intermittent, nothing names it" trap these kinds
                # were split out of "transient" to escape.
                if not systemic or _SYSTEMIC_STATUSES.index(
                    status
                ) < _SYSTEMIC_STATUSES.index(systemic):
                    systemic = status
                continue
            if status == "skip-live-session":
                continue
            return self._perform(num, email, trigger)

        if systemic or transient_failure:
            self._emit(
                ErrorEvent(
                    message="could not freshen: " + _SYSTEMIC_MESSAGES[systemic]
                    if systemic
                    else "could not freshen any candidate (network?)",
                    transient=True,
                )
            )
            return TickOutcome.ERROR
        self._emit(NoSwitchEvent(reason="no-viable-target"))
        return TickOutcome.BLOCKED

    def _rank_candidates(
        self,
        *,
        oauth_candidates: list[str],
        usage: dict[str, dict | str | None],
        settings: AutoSwitchSettings,
    ) -> tuple[list[str], bool]:
        """Order the accounts we could hand off to, best first.

        Returns ``(ordered, any_known)``; ``any_known`` distinguishes "nothing
        qualifies" from "nothing is readable", which are different outcomes.
        Pure — no emits, no state writes.

        The Fable tier wins whenever an account can land below all three
        configured limits. If no candidate can do that, Fable drops out and
        general 5h/7d headroom decides. This prevents an exhausted Fable fleet
        from cycling accounts while still allowing a 5h/7d limit handoff.
        """
        fable_tier: list[tuple[tuple, str]] = []
        general_tier: list[tuple[tuple, str]] = []
        any_known = False
        # Slot order is the final tie-break, carried in the key so the sort
        # is total and two equal candidates never trade places between ticks.
        for slot, num in enumerate(oauth_candidates):
            value = usage.get(num)
            value = value if isinstance(value, dict) else None
            general = general_headroom(value)
            if general is None:
                continue  # unreadable: not evidence of an empty account
            any_known = True  # it EXISTS and is readable either way
            if not accepts_traffic(value, settings, self._models):
                continue
            general_tier.append(((-general, slot), num))
            if accepts_fable(value, settings, self._models):
                fable = fable_headroom(value)
                assert fable is not None
                fable_tier.append(((-fable, -general, slot), num))
        tier = fable_tier or general_tier
        tier.sort(key=lambda t: t[0])
        return [num for _, num in tier], any_known

    # -- adaptive usage scheduling ---------------------------------------------

    def _collect_scheduled_usage(
        self,
        current: str,
        quarantined: set[str] = frozenset(),
        *,
        settings: AutoSwitchSettings | None = None,
    ) -> tuple[dict, dict[str, dict | str | None], dict[str, float | None]]:
        """Two-phase usage collection with an O(1) baseline.

        Phase A fetches the active account (when its persisted poll plan says
        it is due — poll_policy's urgent mode is what tightens that cadence
        near the band) plus ONE due candidate (the one with the stalest data
        — never-fetched first, then oldest fetch); everyone else is served
        from the usage store. Phase B refetches ALL candidates and recomputes
        before any switch decision when a switch could be near: active
        utilization within ``ESCALATION_MARGIN_PCT`` of the threshold, or
        active usage unknown (failover must not run on stale candidate data).
        Selection never runs on the pre-escalation snapshot: the only
        triggers are at-limit and failover, and both imply the escalation
        condition (the deliberate exception: an owned-and-expired active is
        excluded above, so a post-idle-hold failover can run without
        escalating). There is no below-threshold trigger to decide
        provisionally for, so no two-phase commit is needed.

        Stalest-first needs no rotation cursor: it reads the persisted store,
        so the loop and cron-driven ``--once`` runs schedule identically.
        Backoff (``backoffUntil``) is enforced by the collector even for the
        active account — a Retry-After must never be defeated — and during an
        idle-hold no candidate is polled at all (slow crawl for everything).
        Adapted cadences are persisted by the collector itself after each
        fetch (shared with every other surface), not by the engine.

        Returns ``(entries, usage, headroom)`` where ``usage`` carries
        decision values and ``headroom`` the derived headroom per account.
        """
        now = self.clock()
        # Quarantined accounts can never be switch targets, so spending the
        # single alternate poll slot (or an escalation fetch) on one is wasted.
        candidates = [
            n
            for n in self.switcher.switchable_account_numbers()
            if n != current and n not in quarantined
        ]

        pre = self.switcher.usage_entries_by_account(fetch=set())
        plan: set[str] = set()
        active_pre = pre.get(current)
        # The active account is nominated when never fetched, poll-due per its
        # persisted plan, or (no plan yet) past the normal cadence floor. The
        # collector's reserve() honors due-ness even inside the serve TTL, so
        # an urgent plan (60s while burning near the band) actually fetches.
        # A candidate-style plan (slower than any active plan can be) left
        # over from a role change the switcher never saw (e.g. a manual
        # login) is overridden past the active age cap. Exhausted accounts
        # carry their own bounded plan and become due normally.
        stale_candidate_plan = (
            active_pre is not None
            and active_pre.age_s is not None
            and active_pre.age_s >= poll_policy.ACTIVE_MAX_INTERVAL_S
            and (active_pre.poll_interval_s or 0.0)
            > poll_policy.ACTIVE_MAX_INTERVAL_S
            and (binding_pct(active_pre.last_good, self._poll_models) or 0.0) < 100.0
        )
        overslept_plan = (
            active_pre is not None
            and plan_oversleeps_interval(active_pre, now)
        )
        if (
            active_pre is None
            or active_pre.age_s is None
            or stale_candidate_plan
            or overslept_plan
            or (
                active_pre.next_poll_at is not None
                and now >= active_pre.next_poll_at
            )
            or (
                active_pre.next_poll_at is None
                and active_pre.age_s >= poll_policy.MIN_INTERVAL_S
            )
        ):
            plan.add(current)
        if self._idle_hold_since is None:
            pick = due_candidate(candidates, pre, now)
            if pick is not None:
                plan.add(pick)
        entries = self.switcher.usage_entries_by_account(
            fetch=plan,
            # A candidate-style plan on the active slot is deliberately
            # overridden after the active age cap; every other baseline
            # nomination preserves a valid future plan under the store lock.
            scheduled=not stale_candidate_plan,
        )
        usage = {num: entry.decision_value() for num, entry in entries.items()}

        active_value = usage.get(current)
        active_usage = active_value if isinstance(active_value, dict) else None
        # The caller's tick-snapshotted settings, so one tick fetches and
        # decides on the same limits even if apply_threshold() lands mid-tick.
        if settings is None:
            settings = self.settings
        # Distance to the nearest CONFIGURED limit, the same quantity the
        # trigger uses — so the escalation band cannot sit on a different
        # axis from the decision it exists to feed.
        active_distance = policy_headroom(active_usage, settings, self._models)
        escalate = bool(candidates) and (
            (active_distance is None and active_value != USAGE_TOKEN_EXPIRED)
            or (
                active_distance is not None
                and active_distance <= ESCALATION_MARGIN_PCT
            )
        )
        if escalate:
            escalation_fetch = {current, *candidates}
            # Escalation may beat ordinary candidate plans to obtain a fresh
            # switch decision, but a decision-trusted exhausted row cannot be
            # a target. Preserve any wider post-429 plan instead of refetching
            # that token at the bounded all-exhausted wake cadence.
            for num in tuple(escalation_fetch):
                entry = entries.get(num)
                value = usage.get(num)
                planned_headroom = oauth.account_headroom(
                    value if isinstance(value, dict) else None, self._poll_models
                )
                if (
                    entry is not None
                    and entry.next_poll_at is not None
                    and now < entry.next_poll_at
                    and (entry.poll_interval_s or 0.0)
                    > poll_policy.EXHAUSTED_INTERVAL_S
                    and planned_headroom is not None
                    and planned_headroom <= 0
                ):
                    escalation_fetch.remove(num)
            entries = self.switcher.usage_entries_by_account(
                fetch=escalation_fetch
            )
            usage = {num: entry.decision_value() for num, entry in entries.items()}

        headroom = _headroom_by_account(usage, self._models)
        return entries, usage, headroom

    def _perform(
        self,
        number: str,
        email: str,
        trigger: str,
    ) -> TickOutcome:
        if self.dry_run:
            current = self.switcher.current_account_number()
            current_email = self.switcher.account_email(current) if current else ""
            self._emit(
                SwitchEvent(
                    trigger=trigger,
                    from_ref=_ref(current, current_email) if current else None,
                    to_ref=_ref(number, email),
                    dry_run=True,
                )
            )
            return TickOutcome.SWITCHED

        # Hold the state lock across the whole recheck -> switch -> record
        # sequence so two concurrent engines (loop + cron --once) make one
        # serialized decision: the loser re-reads the winner's lastSwitchAt
        # and backs off instead of double-switching. No deadlock cycle: the
        # switch path (cswap FileLock + Claude Code locks) never takes the
        # state lock.
        with self._state_lock():
            state = self._read_state()
            result = self.switcher.switch_to(number, json_output=True)
            if not result or not result.get("switched"):
                self._emit(
                    NoSwitchEvent(
                        reason="already-active",
                        detail=(result or {}).get("reason", ""),
                    )
                )
                return TickOutcome.NO_ACTION

            state["schemaVersion"] = STATE_SCHEMA_VERSION
            state["lastSwitchAt"] = self.clock()
            state["lastSwitchTo"] = number
            atomic_write_json(self.state_path, state)

        self._emit(
            SwitchEvent(
                trigger=trigger,
                from_ref=result.get("from"),
                to_ref=result.get("to"),
                warnings=result.get("warnings", []),
            )
        )
        return TickOutcome.SWITCHED

    # -- helpers --------------------------------------------------------------

    def _check_model_names(
        self, quarantined: set[str], usage: dict[str, dict | str | None]
    ) -> None:
        """One-shot ``autoswitch.model`` typo guard.

        A configured name that no account reports means the filter looks
        active while gating nothing. That's only provable once every
        relevant oauth account has readable usage this tick — adaptive
        polling legitimately leaves gaps before that — and never worth a
        forced refresh of its own.
        """
        wanted = {m.lower(): m for m in self._models if m.lower() != "all"}
        if not wanted:
            self._model_check_done = True  # bare "all" needs no name match
            return
        relevant = [
            n
            for n in self.switcher.switchable_account_numbers()
            if n not in quarantined
            and self.switcher.account_kind_for(n) != "api_key"
        ]
        values = [usage.get(n) for n in relevant]
        readable = [v for v in values if isinstance(v, dict)]
        if not readable or len(readable) != len(values):
            return  # not every account observed yet — re-check next tick
        seen = {
            s["name"].lower()
            for v in readable
            for s in (v.get("scoped") or [])
            if isinstance(s, dict) and isinstance(s.get("name"), str)
        }
        self._model_check_done = True
        missing = [name for low, name in wanted.items() if low not in seen]
        if missing:
            self._emit(
                ConfigWarningEvent(
                    message=(
                        f"autoswitch.model: {', '.join(missing)} matches no "
                        "account's usage windows — only the 5h/7d limits are "
                        "being watched for it (typo?)"
                    )
                )
            )

    def _earliest_recovery(
        self, usage: dict[str, dict | str | None]
    ) -> datetime | None:
        """Earliest moment any account becomes usable again (UTC), or None
        when that moment can't be proven.

        Per account that's the *latest* reset among its ≥100% relevant
        windows — an account blocked on both 5h and a scoped weekly limit
        isn't usable when the 5h rolls over — then the minimum across
        accounts, the active one included (its recovery also ends the
        blocked state). A blocked account whose exhausted windows carry no
        reset time at all could recover at any moment, so it makes the whole
        answer unprovable: return None and let the bounded blocked-cadence
        fallback re-check, rather than sleeping toward another account's
        later known reset."""
        earliest: float | None = None
        now = self.clock()
        for value in usage.values():
            if not isinstance(value, dict):
                continue
            blocked = [
                resets_at
                for _, pct, resets_at in oauth.relevant_windows(
                    value, self._poll_models
                )
                if pct >= 100.0
            ]
            if not blocked:
                continue  # not exhausted — doesn't gate the blocked state
            usable_at = _limiting_reset_ts(value, self._poll_models)
            if usable_at is None or usable_at <= now:
                return None  # blocked with unprovable recovery — don't oversleep
            if earliest is None or usable_at < earliest:
                earliest = usable_at
        if earliest is None:
            return None
        return datetime.fromtimestamp(earliest, tz=timezone.utc)

    def _emit(self, event: AutoSwitchEvent) -> None:
        self.on_event(event)

    # -- loop -------------------------------------------------------------------

    def stop(self) -> None:
        """Ask ``run_loop`` to exit; wakes it from any sleep. Safe to call
        before the loop starts — the stop is never cleared, so the loop
        exits immediately (engines are single-use)."""
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Cut the current inter-tick sleep short and tick now."""
        self._wake.set()

    def apply_threshold(self, threshold: float) -> None:
        """Session override from the TUI: retarget the trigger and poll
        cadence mid-run. The 5h limit (and its deprecated ``threshold``
        alias) only — the 7d/Fable limits and the model axes are fixed at
        construction. The frozen-settings swap is atomic and each tick
        snapshots ``self.settings`` once, so no locking."""
        self.settings = replace(
            self.settings, threshold=threshold, five_hour_threshold=threshold
        )
        self.switcher.set_poll_policy_inputs(
            _poll_threshold(self.settings), self._poll_models
        )

    def _next_delay(self, outcome: TickOutcome) -> float:
        interval = self.settings.interval_seconds
        if outcome is TickOutcome.BLOCKED:
            if self._sleep_until_ts is not None:
                delay = self._sleep_until_ts - self.clock()
                return min(max(delay, interval), MAX_SLEEP_S)
            if self._blocked_wait_long:
                # Truly exhausted with no reset time known / no candidates.
                return max(interval, NO_RESET_FALLBACK_S)
            # Blocked on something that can resolve any tick (hysteresis,
            # unreadable usage) — keep the normal cadence so the at-limit
            # escape isn't missed.
        elif outcome is TickOutcome.NO_ACTION and self._idle_hold_slow:
            # Idle-hold: Claude is idle on an expired token — nothing changes
            # until the user comes back, so crawl. Worst case protection
            # resumes one slow tick after they do.
            return max(interval, NO_RESET_FALLBACK_S)
        # ±10% jitter so multiple machines don't synchronize their API hits.
        return self._respect_poll_plan(interval * (0.9 + 0.2 * random.random()))

    def _respect_poll_plan(self, delay: float) -> float:
        """Shorten a normal-cadence sleep to the store's own next-poll time.

        The planner tightens the active row to URGENT_INTERVAL_S while it
        burns toward the threshold, but the loop always slept
        ``interval_seconds`` — so the plan ran late. Measured mid-episode: the
        row was due 112s ago while the engine still had minutes of sleep left.

        Only ever shortens, never below the planner's floor: the 429 budget
        lives in the plan, and this makes the loop obey it rather than
        override it. Best-effort — the unshortened delay is always safe.
        """
        try:
            current = self.switcher.current_account_number()
            if current is None:
                return delay
            entry = self.switcher.usage_entries_by_account(fetch=set()).get(current)
            if entry is None or entry.next_poll_at is None:
                return delay
            due_in = entry.next_poll_at - self.clock()
            # Clamp the DEADLINE, not the result. max(min(delay, due_in), U)
            # raises a delay that was ALREADY below U: at the configurable
            # floor of 15s it turns a 13.5s jittered sleep into 60s, and at
            # the 60s default it flattens the entire lower jitter half.
            # Bounding due_in instead keeps "only ever shortens" true at every
            # configured interval, and still refuses to poll faster than the
            # planner's own floor when the row is overdue.
            return min(delay, max(due_in, poll_policy.URGENT_INTERVAL_S))
        except Exception:
            return delay

    def run_loop(
        self,
        *,
        settings_loader: Callable[[], AutoSwitchSettings] | None = None,
        should_continue: Callable[[AutoSwitchSettings], bool] | None = None,
    ) -> int:
        """Tick forever (until :meth:`stop`); a failing tick never kills it.

        The detached background worker passes ``settings_loader`` and
        ``should_continue``: ``cswap auto stop|pause|resume`` are settings
        writes made by ANOTHER process, so the loop re-reads settings before
        every tick and mid-sleep, and exits when the worker should no longer
        be running. In-process callers (``cswap auto``, the TUI, the menu bar)
        pass neither and keep the plain loop.
        """
        while True:
            # Clear at the top, not after the wait: a wake() racing a wait
            # timeout is then never lost — the tick right after this clear
            # already sees whatever settings that wake announced.
            self._wake.clear()
            if self._stop.is_set():
                return 0
            if settings_loader is not None:
                self.apply_settings(settings_loader())
            if should_continue is not None and not should_continue(self.settings):
                return 0
            try:
                outcome = self.tick()
            except Exception as e:  # pragma: no cover - tick() already guards
                self._emit(
                    ErrorEvent(message=f"{type(e).__name__}: {e}", transient=True)
                )
                outcome = TickOutcome.ERROR
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
        return 0

    def _sleep(
        self,
        delay: float,
        settings_loader: Callable[[], AutoSwitchSettings] | None,
        should_continue: Callable[[AutoSwitchSettings], bool] | None,
    ) -> None:
        """Wait out the inter-tick delay, wakeable.

        With no background hooks this is exactly ``self._wake.wait(delay)``.
        With them it wakes every ``_CONTROL_POLL_S`` to re-read settings, so
        ``cswap auto stop`` from another process is honored within seconds
        instead of after a possibly ten-minute exhausted-fleet sleep.
        """
        if settings_loader is None and should_continue is None:
            self._wake.wait(delay)
            return
        deadline = self.clock() + delay
        while not self._stop.is_set():
            remaining = deadline - self.clock()
            if remaining <= 0:
                return
            if self._wake.wait(min(remaining, _CONTROL_POLL_S)):
                return  # a real wake: tick now
            if settings_loader is not None:
                self.apply_settings(settings_loader())
            if should_continue is not None and not should_continue(self.settings):
                self.stop()
                return

    def apply_settings(self, settings: AutoSwitchSettings) -> None:
        """Adopt a reloaded settings object mid-run, keeping the poll planner
        in step with the limits the engine now decides on. The model axes are
        fixed at construction (they size derived state), so only the limits
        and the control flags move."""
        if settings == self.settings:
            return
        self.settings = settings
        self.switcher.set_poll_policy_inputs(
            _poll_threshold(settings), self._poll_models
        )
