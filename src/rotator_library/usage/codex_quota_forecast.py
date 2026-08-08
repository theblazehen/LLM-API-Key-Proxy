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
    """Build a deterministic seven-quota-day Codex forecast.

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
    boundaries = [_civil_boundary(quota_day_start, offset) for offset in range(8)]
    normalized, unknown = _normalize_accounts(accounts, now_ts, boundaries[-1].timestamp())
    actual, actual_status = _derive_actual(
        normalized, observations, quota_day_start.timestamp(), now_ts
    )

    baseline_accounts = _accounts_at_day_start(
        normalized, observations, quota_day_start.timestamp()
    )
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
        baseline_day_seconds = boundaries[1].timestamp() - quota_day_start.timestamp()
        baseline_drain = min(
            _drain_rate(baseline_state, quota_day_start.timestamp()) * baseline_day_seconds,
            _maximum_consumable(
                baseline_state, quota_day_start.timestamp(), boundaries[1].timestamp()
            ),
        )
        baseline_sustainable = _maximum_sustainable_rate(
            baseline_state,
            quota_day_start.timestamp(),
            quota_day_start.timestamp() + WEEK_SECONDS,
        ) * baseline_day_seconds
    else:
        baseline_drain = None
        baseline_sustainable = None

    events = _forecast_events(normalized, now_ts, boundaries[-1].timestamp() + WEEK_SECONDS)
    state = _State(
        balances={account.account_id: account.remaining for account in normalized},
        events=events,
        cursor=now_ts,
    )
    aggregate_exhaustion = False

    days: list[dict[str, Any]] = []
    actual_for_target = actual if actual is not None else 0.0
    for index in range(7):
        day_start = boundaries[index].timestamp()
        day_end = boundaries[index + 1].timestamp()
        interval_start = max(state.cursor, day_start)
        if interval_start >= day_end - _EPSILON:
            target_remaining = 0.0
            drain_remaining = 0.0
            sustainable_remaining = 0.0
            selected_reason = "rolling_168h_sustainable"
            contributions: list[dict[str, Any]] = []
            reset_events: list[dict[str, Any]] = []
        else:
            # Quota-day intervals are half-open. A reset at exactly 06:00
            # restores capacity for this new day, never for the prior one.
            reset_events = _apply_events_at(state, interval_start)
            drain_rate = _drain_rate(state, interval_start)
            sustainable_rate = _maximum_sustainable_rate(
                state, interval_start, interval_start + WEEK_SECONDS
            )
            remaining_seconds = day_end - interval_start
            day_capacity = _maximum_consumable(state, interval_start, day_end)
            drain_remaining = min(drain_rate * remaining_seconds, day_capacity)
            sustainable_remaining = min(
                sustainable_rate * remaining_seconds, day_capacity
            )
            if drain_remaining > sustainable_remaining + _EPSILON:
                selected_reason = "next_reset_drain"
                target_remaining = drain_remaining
            else:
                selected_reason = "rolling_168h_sustainable"
                target_remaining = sustainable_remaining

            if normalized and _has_positive_dry_interval(
                state, interval_start, day_end, target_remaining
            ):
                aggregate_exhaustion = True
            consumption = max(0.0, target_remaining)
            contributions, later_reset_events, consumed = _consume_target(
                state, interval_start, day_end, consumption
            )
            reset_events.extend(later_reset_events)
            target_remaining = consumed

        target = (
            actual_for_target + target_remaining if index == 0 else target_remaining
        )
        drain_total = actual_for_target + drain_remaining if index == 0 else drain_remaining
        sustainable_total = (
            actual_for_target + sustainable_remaining
            if index == 0
            else sustainable_remaining
        )
        day = {
                "index": index,
                "start_at": day_start,
                "end_at": day_end,
                "local_date": boundaries[index].date().isoformat(),
                "target": target,
                "remaining_target": target_remaining,
                "drain_candidate": drain_total,
                "sustainable_candidate": sustainable_total,
                "selected_reason": selected_reason,
                "contributions": contributions,
                "reset_events": reset_events,
            }
        if index == 0 and baseline_drain is not None and baseline_sustainable is not None:
            day["planned_at_day_start"] = max(baseline_drain, baseline_sustainable)
        days.append(day)

    generated_at = now_ts
    age_seconds = (
        max(0.0, generated_at - float(source_timestamp))
        if _finite_number(source_timestamp)
        else None
    )
    stale = age_seconds is None or age_seconds > stale_after_seconds
    status = "unavailable" if not normalized else "ready"

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

    return {
        "schema_version": 1,
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
        "days": days,
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
        reconstructed.append(
            _Account(
                account_id=account.account_id,
                email=account.email,
                remaining=remaining,
                natural_reset_at=account.natural_reset_at,
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


def _drain_rate(state: _State, at: float) -> float:
    candidates = [event for event in state.events if event.at > at + _EPSILON]
    if not candidates:
        return 0.0
    event = candidates[0]
    seconds = event.at - at
    return state.balances.get(event.account_id, 0.0) / seconds if seconds > 0 else 0.0


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


def _maximum_consumable(state: _State, start: float, end: float) -> float:
    """Return capacity that can be spent in an interval, including overwrites."""
    probe = state.copy()
    _, _, consumed = _consume_interval(probe, start, end, 1_000_000_000.0)
    return consumed


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


def _consume_target(
    state: _State, start: float, end: float, requested: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Allocate a selected daily target before the earliest overwrite events.

    This is deliberately not the uniform-rate simulation used to solve the
    sustainable candidate. The displayed target is a use-it-or-lose-it target:
    it drains the account that will reset first, then carries any remaining
    target through subsequent reset events.
    """
    contributions: dict[str, float] = {}
    reset_result: list[dict[str, Any]] = []
    remaining = requested
    cursor = start
    while cursor < end - _EPSILON and remaining > _EPSILON:
        next_at = min(
            (event.at for event in state.events if event.at > cursor + _EPSILON),
            default=end,
        )
        _consume_amount(state, cursor, remaining, contributions)
        remaining = requested - sum(contributions.values())
        cursor = min(end, next_at)
        if abs(cursor - next_at) <= _EPSILON and cursor < end - _EPSILON:
            reset_result.extend(_apply_events_at(state, cursor))
    if cursor < end - _EPSILON:
        _consume_amount(state, cursor, remaining, contributions)
    # Preserve every overwrite later in this quota day even when the target
    # was already satisfied. The half-open interval leaves an event exactly at
    # `end` for the quota day beginning at that boundary.
    reset_result.extend(_apply_events_through(state, cursor, end))
    state.cursor = end
    return _contribution_list(contributions), reset_result, sum(contributions.values())


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
