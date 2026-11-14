# Automatic switching: the exact rules

Reference for `cswap auto`, and for the TUI and menu bar auto-switch, which run the same engine. Everyday use is in the [README](../README.md#automatic-switching).

## When it leaves the active account

Every switch starts the prompt cache cold on every running Claude Code session, so the engine keeps the active account until it has to go:

- one of its windows reaches its switch line — `threshold` for the 5-hour window (or the account's own `cswap threshold`), `weeklyThreshold` for the 7-day window and each window `model` names (log reason `threshold`);
- the server refuses it, or a window is at 100% (`at-limit`);
- its login ends within 15 minutes (`login-expiring`);
- its login is dead: Claude Code's token has lapsed for 10 minutes and refreshing it is refused as `invalid_grant` (`failover`).

A failed or missing usage read never moves the account: it holds and reads again (`active-usage-unknown`). A reading that is already over a line stays valid until that window resets, since usage only grows until then. A lapsed live token means nothing is using the account right now, so it holds (`active-idle`).

A `cooldown` (default 5 min) spaces out switches at a line. `at-limit`, `login-expiring` and `failover` never wait for it.

## Where it lands

A candidate is any other account in rotation (not `cswap disable`d, not quarantined, not an API-key account unless `--include-api-key-accounts`, no live `cswap run` session on it). Candidates are ranked by:

1. priority tier (`cswap priority`), higher first;
2. Ultra (no 5-hour window) before accounts with one;
3. the soonest weekly reset, since quota left at a reset is lost;
4. the most room left under its lines.

It takes the first candidate with at least `minHeadroomPct` (default 20) left under each of its lines and a login that lasts at least another 2 hours. A window whose reset has passed since its last reading counts as empty. Before switching, the candidate's token is refreshed if it expires within 10 minutes and the account is read again with a probe; a candidate that no longer qualifies is skipped (at most 3 probes per check, and a candidate whose probe failed is not tried again for 5 minutes).

When no candidate has that much room, it stays on the active account until it is refused, then takes the candidate with the most room that is not at 100% (`no-roomy-target` until then). When every candidate is at 100%, it reports `all-exhausted` with the earliest reset and sleeps until then (at most 10 minutes at a time).

## How it reads usage

- **The active account**: every `activeIntervalSeconds` (default 60), and every 20 seconds once a window is within 5 points of its line, from the rate-limit headers of a one-token request with Claude Code's live token. The usage endpoint only answers a few requests per account before it refuses with 429, and Claude Code spends that allowance on the active account itself; the headers have no such limit. A request to Fable is the only one whose headers carry Fable's weekly window, so the probe asks Fable when `model` names it (`probeModel` overrides the choice).
- **Every other account**: from the usage endpoint every `accountIntervalSeconds` (default 15 min), one account per check, the most overdue first, and right after a window of it resets, through the same collector `cswap list` uses (it refreshes a lapsed stored token through the consume gate, and only reads, never refreshes, an account a live `cswap run` session owns). A 429 backs off that account only.
- **Per-model windows other than Fable's** (`model` set to `Opus`, `all`, …) are not in the headers. For the active account they are read from the endpoint with Claude Code's token, never refreshed, every `accountIntervalSeconds`; between reads, and on a probed target, the last endpoint figure is kept until its reset, as a lower bound. A `model` name that no account reports is named once in the log.

Each reading is stored in the usage cache that `cswap list` and the TUI read, and held there until the engine's next read, so those surfaces do not spend the endpoint allowance on the same account.

## One engine

A live engine holds `autoswitch.lock` in the backup directory while it runs. A second live engine in another process (another worker, a foreground `cswap auto`, the TUI) refuses to start after a few seconds; an engine restarted inside the same process (the TUI or menu bar applying a change) waits for the previous one to finish its tick. `cswap auto status` reports the process that holds the lock.

`cswap auto stop` clears `autoswitch.enabled` and waits up to 30 seconds: the worker exits after its current tick, so a token refresh in flight always gets stored. It never kills anything. A foreground `cswap auto` or the TUI is stopped where it runs. A worker from a build before the lock is asked to stop the same way when `cswap auto start` or `stop` runs.

A dry run (`--dry-run`, the TUI's watch mode) does not take the lock, and never refreshes a stored login or probes a target: it decides on the last readings and names a would-be switch once.

## Logins that end

A stored login ends at a fixed time, and only a fresh `/login` replaces it (`cswap list` shows `re-login in …` during its last week). An account whose refresh token has died is quarantined and reported until you log in with it and re-run `cswap add --slot N`, or replace its stored credentials from a known-good export.

## Cron and systemd

For cron/systemd timers, `--once` reports the outcome in its exit code (`0` switched, `1` error, `2` nothing to do, `3` blocked — no viable target), and `--json` emits one JSON event per line:

```bash
*/5 * * * * cswap auto --once --json >> ~/.cswap-auto.log 2>&1
```

A systemd user service runs the worker in the foreground:

```ini
[Service]
ExecStart=%h/.local/bin/cswap auto _worker
Restart=on-failure
```
