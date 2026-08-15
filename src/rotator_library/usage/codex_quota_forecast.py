"""Pure Codex weekly-quota forecast calculation.

The engine works in normalized weekly-quota percentage points (``pp``).  It has
no persistence or clock dependencies: callers supply current account state,
observation history, and an aware ``now`` value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import math
from typing import Any, Mapping, Sequence


QUOTA_DAY_START_HOUR = 6
WEEK_SECONDS = 7 * 24 * 60 * 60
STALE_AFTER_SECONDS = 15 * 60
_EPSILON = 1e-9
_FEASIBILITY_TOLERANCE = 1e-7
# Risk is an operator-facing state, not a solver convergence signal.  Keep its
# tolerances dimensionally separate: quota residue is measured in percentage
# points, while a dry interval is measured in seconds.
_MATERIAL_DRY_QUOTA_PP = 1e-6
_MATERIAL_DRY_SECONDS = 1.0


@dataclass(frozen=True)
class _ResetEvent:
    at: float
    account_id: str
    kind: str
    credit_id: str | None = None


@dataclass
class _Account:
    account_id: str
    email: str | None
    remaining: float
    natural_reset_at: float
    credit_events: list[_ResetEvent] = field(default_factory=list)
    known_credit_events: list[_ResetEvent] = field(default_factory=list)
    untimed_credit_count: int = 0


@dataclass
class _State:
    balances: dict[str, float]
    events: list[_ResetEvent]
    cursor: float

    def copy(self) -> "_State":
        return _State(dict(self.balances), list(self.events), self.cursor)


def build_codex_quota_forecast(
    *,
    accounts: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]] = (),
    now: datetime,
    source_timestamp: float | None = None,
    stale_after_seconds: float = STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    """Build a deterministic seven-day view of a rolling 168-hour forecast.

    The seven returned quota days are only the display window.  Every daily
    target is solved over the following 168 hours, and reset events are loaded
    through the end of that final rolling window.  This prevents the visible
    plan from consuming capacity needed immediately after day seven.

    Account input fields are ``stable_id`` (or ``account_id``), optional
    ``email``, ``remaining_percent``, ``reset_at``, and optional
    ``reset_credits``.  Each credit may contain ``auto_redeem_at`` or
    ``expires_at``; a timed credit is modeled as an overwrite-to-100 event.

    Observation fields are ``stable_id`` (or ``account_id``), ``observed_at``,
    and ``remaining_percent``.  A baseline at or before the current 06:00 local
    boundary is required for every included account before current-day actual
    usage is reported.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not math.isfinite(stale_after_seconds) or stale_after_seconds < 0:
        raise ValueError("stale_after_seconds must be a finite non-negative number")

    now_ts = now.timestamp()
    quota_day_start = _quota_day_start(now)
    # Return seven display days, but simulate fourteen.  The second week is
    # needed to expose the first complete quota day after every account's next
    # reset and to prove that the visible plan does not strand that cycle.
    boundaries = [_civil_boundary(quota_day_start, offset) for offset in range(15)]
    normalized, unknown = _normalize_accounts(accounts, now_ts, boundaries[-1].timestamp())
    actual, actual_status = _derive_actual(
        normalized, observations, quota_day_start.timestamp(), now_ts
    )

    baseline_accounts = _accounts_at_day_start(
        normalized, observations, quota_day_start.timestamp()
    )
    baseline_plan: dict[str, Any] | None = None
    if baseline_accounts is not None:
        baseline_events = _forecast_events(
            baseline_accounts,
            quota_day_start.timestamp(),
            boundaries[-1].timestamp() + WEEK_SECONDS,
        )
        baseline_state = _State(
            balances={account.account_id: account.remaining for account in baseline_accounts},
            events=baseline_events,
            cursor=quota_day_start.timestamp(),
        )
        baseline_rate_state = baseline_state.copy()
        _apply_events_at(baseline_rate_state, quota_day_start.timestamp())
        baseline_rate = _maximum_sustainable_rate(
            baseline_rate_state,
            quota_day_start.timestamp(),
            quota_day_start.timestamp() + WEEK_SECONDS,
        )
        baseline_plan = _consume_baseline_plus_expiry(
            baseline_state,
            quota_day_start.timestamp(),
            boundaries[1].timestamp(),
            baseline_rate,
        )

    events = _forecast_events(normalized, now_ts, boundaries[-1].timestamp() + WEEK_SECONDS)
    state = _State(
        balances={account.account_id: account.remaining for account in normalized},
        events=events,
        cursor=now_ts,
    )
    aggregate_exhaustion = False

    days: list[dict[str, Any]] = []
    for index in range(14):
        day_start = boundaries[index].timestamp()
        day_end = boundaries[index + 1].timestamp()
        interval_start = max(state.cursor, day_start)
        rate_state = state.copy()
        _apply_events_at(rate_state, interval_start)
        daily_rate = _maximum_sustainable_rate(
            rate_state, interval_start, interval_start + WEEK_SECONDS
        ) * 24.0 * 60.0 * 60.0
        live_sustainable_remaining = daily_rate * max(0.0, day_end - interval_start) / (
            24.0 * 60.0 * 60.0
        )

        if index == 0 and baseline_plan is not None and actual is not None:
            target = baseline_plan["target"]
            target_remaining = max(0.0, target - actual)
            baseline_allocation = baseline_plan["baseline_allocation"]
            expiry_bonus = baseline_plan["expiry_bonus"]
            expected_used_by_now = _expected_used_by(
                baseline_plan["segments"], interval_start
            )
            if normalized and _has_positive_dry_interval(
                state, interval_start, day_end, target_remaining
            ):
                aggregate_exhaustion = True
            contributions, reset_events, consumed = _consume_fixed_allocation(
                state, interval_start, day_end, target_remaining
            )
            target_remaining = max(0.0, target - actual)
        elif interval_start >= day_end - _EPSILON:
            target = 0.0
            target_remaining = 0.0
            baseline_allocation = 0.0
            expiry_bonus = 0.0
            expected_used_by_now = 0.0
            contributions: list[dict[str, Any]] = []
            reset_events: list[dict[str, Any]] = []
        else:
            plan = _consume_baseline_plus_expiry(
                state,
                interval_start,
                day_end,
                daily_rate / (24.0 * 60.0 * 60.0),
            )
            target = plan["target"]
            target_remaining = target
            baseline_allocation = plan["baseline_allocation"]
            expiry_bonus = plan["expiry_bonus"]
            expected_used_by_now = 0.0
            contributions = plan["contributions"]
            reset_events = plan["reset_events"]

        day = {
                "index": index,
                "start_at": day_start,
                "end_at": day_end,
                "local_date": boundaries[index].date().isoformat(),
                "target": target,
                "remaining_target": target_remaining,
                "baseline_allocation": baseline_allocation,
                "expiry_bonus": expiry_bonus,
                "sustainable_daily_rate": daily_rate,
                "live_sustainable_remaining": live_sustainable_remaining,
                "expected_used_by_now": expected_used_by_now,
                "contributions": contributions,
                "reset_events": reset_events,
            }
        days.append(day)

    generated_at = now_ts
    age_seconds = (
        max(0.0, generated_at - float(source_timestamp))
        if _finite_number(source_timestamp)
        else None
    )
    stale = age_seconds is None or age_seconds > stale_after_seconds
    status = (
        "ready"
        if normalized and baseline_plan is not None and actual is not None
        else "unavailable"
    )

    account_result = []
    for account in normalized:
        credits = [
            {
                "at": event.at,
                "kind": event.kind,
                "credit_id": event.credit_id,
                "restores_to": 100.0,
            }
            for event in account.credit_events
        ]
        account_result.append(
            {
                "account_id": account.account_id,
                "email": account.email,
                "remaining_percent": account.remaining,
                "natural_reset_at": account.natural_reset_at,
                "timed_credit_resets": credits,
                "untimed_credit_count": account.untimed_credit_count,
            }
        )

    # Find the first complete quota day after every account has crossed its
    # next overwrite. A timed credit can overwrite an account before its
    # reset and starts a fresh weekly cadence, so it participates in
    # the same selection. Export the current-state constant-rate advisory,
    # not a target that may include a one-off expiry bonus.
    next_overwrites: list[float] = []
    for account in normalized:
        natural = account.natural_reset_at
        while natural <= now_ts + _EPSILON:
            natural += WEEK_SECONDS
        candidates = [natural]
        candidates.extend(
            event.at for event in account.credit_events if event.at > now_ts + _EPSILON
        )
        next_overwrites.append(min(candidates))
    final_next_overwrite = max(next_overwrites) if next_overwrites else None
    post_reset_day = next(
        (
            day
            for day in days[1:]
            if final_next_overwrite is not None
            and day["start_at"] >= final_next_overwrite - _EPSILON
        ),
        None,
    )
    post_reset = {
        "after_at": final_next_overwrite,
        "day_start_at": post_reset_day["start_at"] if post_reset_day else None,
        "daily_sustainable_pace": (
            post_reset_day["sustainable_daily_rate"] if post_reset_day else None
        ),
    }

    return {
        "schema_version": 2,
        "status": status,
        "unit": "weekly_quota_percentage_points",
        "generated_at": generated_at,
        "source_timestamp": float(source_timestamp) if _finite_number(source_timestamp) else None,
        "age_seconds": age_seconds,
        "stale": stale,
        "timezone": str(now.tzinfo),
        "quota_day_start_hour": QUOTA_DAY_START_HOUR,
        "horizon": {
            "start_at": boundaries[0].timestamp(),
            "end_at": boundaries[7].timestamp(),
            "day_count": 7,
        },
        "planning_horizon": {
            "start_at": boundaries[0].timestamp(),
            "end_at": boundaries[14].timestamp(),
            "day_count": 14,
            "rolling_window_seconds": WEEK_SECONDS,
        },
        "actual": {
            "status": actual_status,
            "used_since_day_start": actual,
        },
        "risk": {
            # This evaluates only the engine's own capacity-derived targets.
            # Learning, stale, and unavailable source data are data-quality
            # states; none is evidence that the aggregate pool is exhausted.
            "aggregate_exhaustion": aggregate_exhaustion,
            "basis": "forecast_targets",
        },
        "today": days[0],
        "days": days[:7],
        "planning_days": days,
        "post_reset": post_reset,
        "accounts": account_result,
        "unknown_accounts": unknown,
    }


def _quota_day_start(now: datetime) -> datetime:
    candidate = datetime.combine(now.date(), time(QUOTA_DAY_START_HOUR), now.tzinfo)
    if now < candidate:
        candidate = datetime.combine(
            now.date() - timedelta(days=1), time(QUOTA_DAY_START_HOUR), now.tzinfo
        )
    return candidate


def _civil_boundary(start: datetime, offset_days: int) -> datetime:
    return datetime.combine(
        start.date() + timedelta(days=offset_days),
        time(QUOTA_DAY_START_HOUR),
        start.tzinfo,
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _timestamp(value: Any) -> float | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        value = value.timestamp()
    return float(value) if _finite_number(value) else None


def _normalize_accounts(
    accounts: Sequence[Mapping[str, Any]], now_ts: float, horizon_end: float
) -> tuple[list[_Account], list[dict[str, Any]]]:
    normalized: list[_Account] = []
    unknown: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(accounts):
        account_id_value = raw.get("stable_id", raw.get("account_id"))
        account_id = str(account_id_value) if account_id_value not in (None, "") else ""
        reasons: list[str] = []
        if not account_id:
            reasons.append("missing_account_id")
        elif account_id in seen:
            reasons.append("duplicate_account_id")
        remaining_value = raw.get("remaining_percent")
        if not _finite_number(remaining_value) or not 0.0 <= float(remaining_value) <= 100.0:
            reasons.append("missing_or_invalid_remaining_percent")
        reset_at = _timestamp(raw.get("reset_at"))
        if reset_at is None:
            reasons.append("missing_or_invalid_reset_at")
        if reasons:
            unknown.append(
                {
                    "account_id": account_id or None,
                    "email": raw.get("email"),
                    "input_index": position,
                    "reasons": reasons,
                }
            )
            continue
        seen.add(account_id)
        credit_events: list[_ResetEvent] = []
        known_credit_events: list[_ResetEvent] = []
        untimed = 0
        credit_seen: set[tuple[float, str | None]] = set()
        for credit in raw.get("reset_credits") or ():
            if not isinstance(credit, Mapping):
                untimed += 1
                continue
            event_at = _timestamp(credit.get("auto_redeem_at"))
            kind = "credit_auto_redeem"
            if event_at is None:
                event_at = _timestamp(credit.get("expires_at"))
                kind = "credit_expiry"
            if event_at is None:
                untimed += 1
                continue
            credit_id_value = credit.get("id", credit.get("credit_id"))
            credit_id = str(credit_id_value) if credit_id_value is not None else None
            key = (event_at, credit_id)
            if key in credit_seen:
                continue
            credit_seen.add(key)
            known_credit_events.append(_ResetEvent(event_at, account_id, kind, credit_id))
            if now_ts < event_at <= horizon_end + WEEK_SECONDS:
                credit_events.append(_ResetEvent(event_at, account_id, kind, credit_id))
        known_credit_events.sort(key=lambda event: (event.at, event.credit_id or ""))
        credit_events.sort(key=lambda event: (event.at, event.credit_id or ""))
        normalized.append(
            _Account(
                account_id=account_id,
                email=str(raw["email"]) if raw.get("email") is not None else None,
                remaining=float(remaining_value),
                natural_reset_at=float(reset_at),
                credit_events=credit_events,
                known_credit_events=known_credit_events,
                untimed_credit_count=untimed,
            )
        )
    normalized.sort(key=lambda account: (account.natural_reset_at, account.account_id))
    unknown.sort(key=lambda item: (item["account_id"] or "", item["input_index"]))
    return normalized, unknown


def _forecast_events(accounts: Sequence[_Account], start: float, end: float) -> list[_ResetEvent]:
    events: list[_ResetEvent] = []
    for account in accounts:
        timed = sorted(account.credit_events, key=lambda event: (event.at, event.kind, event.credit_id or ""))
        natural = account.natural_reset_at
        # ``reset_at`` normally names the next reset. For a reconstructed
        # start state it can be one week ahead of an overwrite later that same
        # quota day, so locate the first cadence event strictly after start.
        while natural - WEEK_SECONDS > start + _EPSILON:
            natural -= WEEK_SECONDS
        while natural <= start + _EPSILON:
            natural += WEEK_SECONDS
        for credit in timed:
            if credit.at < start - _EPSILON:
                continue
            while natural < credit.at - _EPSILON and natural <= end + _EPSILON:
                events.append(_ResetEvent(natural, account.account_id, "natural"))
                natural += WEEK_SECONDS
            if credit.at <= end + _EPSILON:
                events.append(credit)
            # A timed overwrite starts a fresh weekly window. The old natural
            # reset inside that window is no longer real capacity.
            natural = credit.at + WEEK_SECONDS
        while natural <= end + _EPSILON:
            events.append(_ResetEvent(natural, account.account_id, "natural"))
            natural += WEEK_SECONDS
    # More than one reset at the same instant is still explanatory, but applying
    # multiple overwrite events has the same deterministic result.
    events.sort(key=lambda event: (event.at, event.account_id, event.kind, event.credit_id or ""))
    return events


def _derive_actual(
    accounts: Sequence[_Account],
    observations: Sequence[Mapping[str, Any]],
    day_start: float,
    now_ts: float,
) -> tuple[float | None, str]:
    if not accounts:
        return None, "baseline_unavailable"
    by_account: dict[str, list[tuple[float, float]]] = {account.account_id: [] for account in accounts}
    for raw in observations:
        account_id_value = raw.get("stable_id", raw.get("account_id"))
        account_id = str(account_id_value) if account_id_value is not None else ""
        if account_id not in by_account:
            continue
        observed_at = _timestamp(raw.get("observed_at"))
        remaining = raw.get("remaining_percent")
        if (
            observed_at is None
            or observed_at > now_ts + _EPSILON
            or not _finite_number(remaining)
            or not 0.0 <= float(remaining) <= 100.0
        ):
            continue
        by_account[account_id].append((observed_at, float(remaining)))

    total = 0.0
    for account in accounts:
        points = sorted(by_account[account.account_id], key=lambda item: (item[0], item[1]))
        baselines = [point for point in points if point[0] <= day_start + _EPSILON]
        if not baselines:
            return None, "baseline_unavailable"
        baseline = baselines[-1]
        # A reset restores the window to 100 and overwrites the preceding
        # remainder.  Insert it even when the proxy did not sample exactly at
        # that moment, otherwise a pre-reset baseline and post-reset balance
        # loses all post-reset consumption.
        transitions: list[tuple[float, int, float]] = []
        for reset_at in _actual_reset_times(account, day_start, now_ts):
            transitions.append((reset_at, 0, 100.0))
        transitions.extend(
            (observed_at, 1, remaining)
            for observed_at, remaining in points
            if day_start < observed_at <= now_ts + _EPSILON
        )
        if not transitions or transitions[-1][0] < now_ts - _EPSILON:
            transitions.append((now_ts, 1, account.remaining))
        previous = baseline[1]
        for _, _, current in sorted(transitions):
            if current < previous:
                total += previous - current
            previous = current
    return total, "observed"


def _actual_reset_times(account: _Account, start: float, end: float) -> list[float]:
    """Return overwrites in the half-open observation span ``[start, end)``."""
    natural = account.natural_reset_at
    while natural - WEEK_SECONDS >= start - _EPSILON:
        natural -= WEEK_SECONDS
    while natural < start - _EPSILON:
        natural += WEEK_SECONDS
    result: list[float] = []
    while natural < end - _EPSILON:
        if natural >= start - _EPSILON:
            result.append(natural)
        natural += WEEK_SECONDS
    result.extend(
        event.at
        for event in account.known_credit_events
        if start - _EPSILON <= event.at < end - _EPSILON
    )
    return sorted(set(result))


def _accounts_at_day_start(
    accounts: Sequence[_Account],
    observations: Sequence[Mapping[str, Any]],
    day_start: float,
) -> list[_Account] | None:
    """Reconstruct account balances at the current quota-day boundary."""
    baselines: dict[str, tuple[float, float]] = {}
    for raw in observations:
        account_id_value = raw.get("stable_id", raw.get("account_id"))
        account_id = str(account_id_value) if account_id_value is not None else ""
        observed_at = _timestamp(raw.get("observed_at"))
        remaining = raw.get("remaining_percent")
        if (
            observed_at is None
            or observed_at > day_start + _EPSILON
            or not _finite_number(remaining)
            or not 0.0 <= float(remaining) <= 100.0
        ):
            continue
        candidate = (observed_at, float(remaining))
        if account_id not in baselines or candidate[0] > baselines[account_id][0]:
            baselines[account_id] = candidate
    if any(account.account_id not in baselines for account in accounts):
        return None
    reconstructed: list[_Account] = []
    for account in accounts:
        observed_at, remaining = baselines[account.account_id]
        if observed_at < day_start - _EPSILON and _actual_reset_times(
            account, day_start, day_start + 1.0
        ):
            remaining = 100.0
        natural_reset_at = account.natural_reset_at
        while natural_reset_at - WEEK_SECONDS >= day_start - _EPSILON:
            natural_reset_at -= WEEK_SECONDS
        reconstructed.append(
            _Account(
                account_id=account.account_id,
                email=account.email,
                remaining=remaining,
                natural_reset_at=natural_reset_at,
                credit_events=list(account.known_credit_events),
                known_credit_events=list(account.known_credit_events),
                untimed_credit_count=account.untimed_credit_count,
            )
        )
    return reconstructed


def _next_event_for_account(state: _State, account_id: str, at: float) -> float:
    return min(
        (event.at for event in state.events if event.account_id == account_id and event.at > at + _EPSILON),
        default=math.inf,
    )


def _maximum_sustainable_rate(state: _State, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    upper_capacity = sum(state.balances.values()) + 100.0 * sum(
        1 for event in state.events if start < event.at <= end + _EPSILON
    )
    high = upper_capacity / (end - start)
    low = 0.0
    for _ in range(70):
        middle = (low + high) / 2.0
        probe = state.copy()
        _, _, consumed = _consume_interval(probe, start, end, middle * (end - start))
        if consumed + _FEASIBILITY_TOLERANCE >= middle * (end - start):
            low = middle
        else:
            high = middle
    return low


def _has_positive_dry_interval(
    state: _State, start: float, end: float, requested: float
) -> bool:
    """Return whether paced planned use makes the pool dry before replenishment.

    A reset at the start of an interval is applied before checking it, so an
    exact depletion/reset boundary is not a dry interval. An already-dead pool
    is a dry interval even when its capacity-derived target is zero.
    """
    if end <= start + _EPSILON:
        return False
    probe = state.copy()
    rate = max(0.0, requested) / (end - start)
    cursor = start
    while cursor < end - _EPSILON:
        _apply_events_at(probe, cursor)
        next_at = min(
            (event.at for event in probe.events if event.at > cursor + _EPSILON),
            default=end,
        )
        segment_end = min(end, next_at)
        duration = segment_end - cursor
        available = sum(max(0.0, balance) for balance in probe.balances.values())
        if duration >= _MATERIAL_DRY_SECONDS:
            if available <= _MATERIAL_DRY_QUOTA_PP:
                return True
            if rate > _EPSILON:
                quota_deficit = rate * duration - available
                if quota_deficit > _MATERIAL_DRY_QUOTA_PP:
                    dry_seconds = quota_deficit / rate
                    if dry_seconds >= _MATERIAL_DRY_SECONDS:
                        return True
        _consume_amount(probe, cursor, rate * duration, {})
        cursor = segment_end
        if abs(cursor - next_at) <= _EPSILON and cursor < end - _EPSILON:
            _apply_events_at(probe, cursor)
    return False


def _consume_baseline_plus_expiry(
    state: _State, start: float, end: float, baseline_rate: float
) -> dict[str, Any]:
    """Consume a constant baseline and only the quota it would overwrite.

    Each reset boundary closes one planning segment. The constant baseline is
    consumed earliest-deadline-first within that segment, then any balance
    still held by *all* accounts resetting at that instant is the segment's
    expiry bonus. This aggregates simultaneous resets and never carries a
    pre-reset drain rate into the following segment.
    """
    contributions: dict[str, float] = {}
    reset_result = _apply_events_at(state, start)
    segments: list[dict[str, float]] = []
    baseline_allocation = 0.0
    expiry_bonus = 0.0
    cursor = start

    while cursor < end - _EPSILON:
        next_at = min(
            (
                event.at
                for event in state.events
                if cursor < event.at <= end + _EPSILON
            ),
            default=end,
        )
        baseline_requested = max(0.0, baseline_rate) * (next_at - cursor)
        baseline_consumed = _consume_amount(
            state, cursor, baseline_requested, contributions
        )
        baseline_allocation += baseline_consumed

        segment_bonus = 0.0
        if next_at <= end + _EPSILON:
            resetting_accounts = sorted(
                {
                    event.account_id
                    for event in state.events
                    if abs(event.at - next_at) <= _EPSILON
                }
            )
            for account_id in resetting_accounts:
                amount = max(0.0, state.balances.get(account_id, 0.0))
                if amount <= _EPSILON:
                    continue
                state.balances[account_id] = 0.0
                contributions[account_id] = (
                    contributions.get(account_id, 0.0) + amount
                )
                segment_bonus += amount
            expiry_bonus += segment_bonus

        segments.append(
            {
                "start_at": cursor,
                "end_at": next_at,
                "baseline_allocation": baseline_consumed,
                "expiry_bonus": segment_bonus,
                "target": baseline_consumed + segment_bonus,
            }
        )
        cursor = next_at
        if cursor < end - _EPSILON:
            reset_result.extend(_apply_events_at(state, cursor))

    state.cursor = end
    return {
        "target": baseline_allocation + expiry_bonus,
        "baseline_allocation": baseline_allocation,
        "expiry_bonus": expiry_bonus,
        "segments": segments,
        "contributions": _contribution_list(contributions),
        "reset_events": reset_result,
    }


def _expected_used_by(segments: Sequence[Mapping[str, Any]], at: float) -> float:
    """Integrate a fixed piecewise allocation schedule through ``at``."""
    expected = 0.0
    for segment in segments:
        start = float(segment["start_at"])
        end = float(segment["end_at"])
        target = float(segment["target"])
        if at <= start + _EPSILON:
            break
        if at >= end - _EPSILON:
            expected += target
            continue
        if end > start:
            expected += target * (at - start) / (end - start)
        break
    return expected


def _consume_interval(
    state: _State, start: float, end: float, requested: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Consume a total amount uniformly, applying reset events on the timeline."""
    if end <= start or requested <= _EPSILON:
        state.cursor = max(state.cursor, end)
        applied = _apply_events_through(state, start, end)
        return [], applied, 0.0
    rate = requested / (end - start)
    contributions: dict[str, float] = {}
    reset_result: list[dict[str, Any]] = []
    cursor = start
    while cursor < end - _EPSILON:
        next_at = min(
            (event.at for event in state.events if event.at > cursor + _EPSILON),
            default=end,
        )
        segment_end = min(end, next_at)
        wanted = rate * (segment_end - cursor)
        consumed = _consume_amount(state, cursor, wanted, contributions)
        if consumed + _EPSILON < wanted:
            # Do not stop at a depleted account pool. A later reset may restore
            # capacity in this same day and must remain visible in both the
            # simulated balance and explanatory event list.
            cursor = segment_end
            if abs(cursor - next_at) <= _EPSILON:
                reset_result.extend(_apply_events_at(state, cursor))
            continue
        cursor = segment_end
        if abs(cursor - next_at) <= _EPSILON:
            reset_result.extend(_apply_events_at(state, cursor))
    state.cursor = end
    return _contribution_list(contributions), reset_result, sum(contributions.values())


def _consume_fixed_allocation(
    state: _State, start: float, end: float, requested: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Allocate a frozen remainder earliest-deadline-first.

    The amount is fixed before this function runs. Spending currently banked
    quota first keeps the reconstructed day plan coherent without inventing a
    new drain rate or carrying one across a reset boundary.
    """
    contributions: dict[str, float] = {}
    reset_result = _apply_events_at(state, start)
    remaining = max(0.0, requested)
    cursor = start
    while cursor < end - _EPSILON:
        _consume_amount(state, cursor, remaining, contributions)
        remaining = max(0.0, requested - sum(contributions.values()))
        next_at = min(
            (
                event.at
                for event in state.events
                if cursor < event.at < end - _EPSILON
            ),
            default=end,
        )
        cursor = next_at
        if cursor < end - _EPSILON:
            reset_result.extend(_apply_events_at(state, cursor))
        if remaining <= _EPSILON:
            break
    reset_result.extend(_apply_events_through(state, cursor, end))
    state.cursor = end
    return (
        _contribution_list(contributions),
        reset_result,
        requested - max(0.0, remaining),
    )


def _consume_amount(
    state: _State,
    at: float,
    amount: float,
    contributions: dict[str, float],
) -> float:
    remaining = amount
    ordered = sorted(
        state.balances,
        key=lambda account_id: (_next_event_for_account(state, account_id, at), account_id),
    )
    for account_id in ordered:
        available = state.balances[account_id]
        taken = min(available, remaining)
        if taken > _EPSILON:
            state.balances[account_id] = available - taken
            contributions[account_id] = contributions.get(account_id, 0.0) + taken
            remaining -= taken
        if remaining <= _EPSILON:
            break
    return amount - max(0.0, remaining)


def _apply_events_through(state: _State, start: float, end: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for at in sorted({event.at for event in state.events if start < event.at < end - _EPSILON}):
        result.extend(_apply_events_at(state, at))
    return result


def _apply_events_at(state: _State, at: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    matching = [event for event in state.events if abs(event.at - at) <= _EPSILON]
    for event in matching:
        before = state.balances.get(event.account_id, 0.0)
        state.balances[event.account_id] = 100.0
        result.append(
            {
                "at": event.at,
                "account_id": event.account_id,
                "kind": event.kind,
                "credit_id": event.credit_id,
                "balance_before": before,
                "balance_after": 100.0,
                "capacity_added": 100.0 - before,
            }
        )
    state.events = [event for event in state.events if abs(event.at - at) > _EPSILON]
    return result


def _contribution_list(contributions: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {"account_id": account_id, "amount": amount}
        for account_id, amount in contributions.items()
        if amount > _EPSILON
    ]
