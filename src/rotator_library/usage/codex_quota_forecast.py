"""Pure, forward-only weekly quota plans in normalized account percentage points.

Natural resets are projected full overwrites with a common 168-hour cadence.
The plan is anchored at the current civil day's cross-over, so every civil day
keeps one budget for its whole length: consumption inside a day is measured
against that budget instead of re-planning the remaining week downward.
Credit options and retrospective measurement are separate from this baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import math
from typing import Any, Mapping, Sequence

from .codex_quota_actual import derive_usage_observation
from .codex_quota_scenarios import build_credit_scenarios


QUOTA_DAY_START_HOUR = 6
WEEK_SECONDS = 7 * 24 * 60 * 60
STALE_AFTER_SECONDS = 15 * 60
_EPSILON = 1e-9
_DAY_SECONDS = 24 * 60 * 60
_USAGE_BASIS = "sampled_counter_drop_since_crossover"
_PARTIAL_USAGE_BASIS = "partial_pool_usage"
_NO_CROSSOVER_BASIS = "no_crossover_snapshot"
_GENERATION_CHANGED_BASIS = "pool_reset_after_crossover"
_RISE_USAGE_BASIS = "counter_rise_since_crossover"
# Upstream weekly anchors jitter by about a second; a real generation change
# moves the anchor by days. Anything in between is treated as not comparable.
_ANCHOR_TOLERANCE_SECONDS = 60.0
_ASSUMPTIONS = [
    "Current balances and reset anchors describe coherent, distinct weekly quota pools.",
    "Advertised natural resets are projected full overwrites repeating every 168 hours.",
    "Weekly percentage points are fungible for planning, not token-equivalent across accounts.",
    "Short-window, model-family and routing constraints are not covered by this weekly plan.",
    "Optional surplus must be used before its account generation expires; it is never carried forward.",
    "Credit expiry does not establish an automatic refill; credit scenarios require successful redemption.",
    "Each civil day's budget is fixed at its cross-over snapshot and is not replanned until the next cross-over.",
    "Reported day usage is a sampled weekly-counter drop from that cross-over; it is not precision-corrected.",
]


@dataclass(frozen=True)
class _ResetEvent:
    at: float
    account_id: str


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
    """Build seven civil quota days of forward allocation anchored at cross-over.

    Each civil day owns one budget for its whole length. When every pool has a
    cross-over sample for the current day, the plan is computed from those
    samples, so consumption earlier today consumes today's budget instead of
    shrinking every later day. Without cross-over samples the plan falls back to
    the live snapshot and reports no day usage. Past consumption is never
    subtracted from the plan, and expiring inventory may only be consumed before
    its own overwrite.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not _finite_number(stale_after_seconds) or stale_after_seconds < 0:
        raise ValueError("stale_after_seconds must be a finite non-negative number")
    now_ts = now.timestamp()
    boundary = _quota_day_start(now)
    boundaries = [_civil_boundary(boundary, offset).timestamp() for offset in range(15)]
    normalized, unknown = _normalize_accounts(accounts, now_ts)
    samples = _crossover_samples(observations, boundaries[0])
    timestamps = [_source_time(account) for account in normalized]
    source = _timestamp(source_timestamp)
    if timestamps and all(value is not None for value in timestamps):
        source = min(timestamps)
    elif any(value is not None for value in timestamps):
        source = None
    age = now_ts - source if source is not None else None
    stale = age is None or age < 0 or age > stale_after_seconds
    measurement_accounts = normalized + [accounts[item["input_index"]] for item in unknown]
    actual = derive_usage_observation(measurement_accounts, observations, boundaries[0], now_ts)
    result: dict[str, Any] = {
        "schema_version": 3,
        "status": "unavailable",
        "unit": "weekly_quota_percentage_points",
        "generated_at": now_ts,
        "source_timestamp": source,
        "age_seconds": max(0.0, age) if age is not None else None,
        "stale": stale,
        "timezone": str(now.tzinfo),
        "quota_day_start_hour": QUOTA_DAY_START_HOUR,
        "horizon": {"start_at": boundaries[0], "end_at": boundaries[7], "day_count": 7},
        "planning_horizon": {
            "start_at": boundaries[0], "end_at": boundaries[14],
            "day_count": 14, "rolling_window_seconds": WEEK_SECONDS,
        },
        "actual": actual,
        "risk": {"aggregate_exhaustion": None, "basis": "weekly_natural_resets"},
        "today": None,
        "days": [],
        "planning_days": [],
        "post_reset": {"after_at": None, "day_start_at": None, "daily_sustainable_pace": None},
        "accounts": [_account_result(account) for account in normalized],
        "unknown_accounts": unknown,
        "assumptions": list(_ASSUMPTIONS),
        "blocked_until": None,
        "credit_scenarios": [],
        "credit_scenarios_reason": None,
        "plan_basis": None,
        "crossover": {
            "at": boundaries[0], "anchored_pools": 0, "pool_count": len(normalized),
        },
        "usage": _day_usage(normalized, samples),
    }
    if not normalized:
        result["reason"] = "no_usable_weekly_accounts"
        result["credit_scenarios_reason"] = "no_usable_weekly_accounts"
        return result

    frozen = _frozen_balances(normalized, samples)
    if frozen is not None:
        state = _State(frozen, _forecast_events(normalized, boundaries[-1] + WEEK_SECONDS), boundaries[0])
        result["plan_basis"] = "crossover_frozen"
        result["crossover"]["anchored_pools"] = len(frozen)
    else:
        state = _State(
            {account["stable_id"]: account["remaining_percent"] for account in normalized},
            _forecast_events(normalized, boundaries[-1] + WEEK_SECONDS), now_ts,
        )
        result["plan_basis"] = "live_snapshot"
    # Exhaustion describes the live counter, not the frozen plan that continues
    # to show the day's full budget.
    empty = sum(account["remaining_percent"] for account in normalized) <= _EPSILON
    result["status"] = "partial" if unknown else "blocked" if empty else "ready"
    result["reason"] = "unknown_weekly_accounts" if unknown else "awaiting_natural_reset" if empty else None
    result["risk"]["aggregate_exhaustion"] = None if unknown else empty
    result["blocked_until"] = min(event.at for event in state.events) if empty and not unknown else None
    days = []
    for index in range(14):
        start = max(state.cursor, boundaries[index])
        plan = _consume_day(state, start, boundaries[index + 1])
        span = boundaries[index + 1] - boundaries[index]
        elapsed = now_ts - boundaries[index]
        days.append({
            "index": index, "start_at": boundaries[index], "end_at": boundaries[index + 1],
            "local_date": _civil_boundary(boundary, index).date().isoformat(),
            # A future day always owns one budget for its whole length. Today's
            # is only known from a frozen plan, so `_attach_today_budget` sets it.
            "allocated_percent": plan["target"] if index > 0 else None,
            # Day progress is civil time, not planned consumption, so a chart can
            # show how much of the day is left independently of the budget.
            "day_fraction_elapsed": _clamp_fraction(elapsed, span),
            "day_fraction_remaining": _clamp_fraction(span - elapsed, span),
            **plan,
        })
    result["today"] = days[0]
    result["days"] = days[:7]
    result["planning_days"] = days
    _attach_today_budget(result["today"], result["usage"], result["plan_basis"])
    after = max(account["reset_at"] for account in normalized)
    post_day = next((day for day in days[1:] if day["start_at"] >= after), None)
    result["post_reset"] = {
        "after_at": after,
        "day_start_at": post_day["start_at"] if post_day else None,
        "daily_sustainable_pace": post_day["sustainable_daily_rate"] if post_day else None,
    }
    scenario_reason = _credit_scenario_reason(normalized, unknown, now_ts, stale_after_seconds)
    if scenario_reason is None:
        scenarios = build_credit_scenarios(normalized, now_ts)
        unavailable = next((item for item in scenarios if item.get("status") == "unavailable"), None)
        if unavailable is not None:
            scenario_reason = unavailable.get("reason") or "credit_scenario_inputs_unavailable"
        else:
            result["credit_scenarios"] = scenarios
    result["credit_scenarios_reason"] = scenario_reason
    return result


def _quota_day_start(now: datetime) -> datetime:
    candidate = datetime.combine(now.date(), time(QUOTA_DAY_START_HOUR), now.tzinfo)
    if now < candidate:
        candidate = datetime.combine(now.date() - timedelta(days=1), time(QUOTA_DAY_START_HOUR), now.tzinfo)
    return candidate


def _civil_boundary(start: datetime, offset_days: int) -> datetime:
    return datetime.combine(start.date() + timedelta(days=offset_days), time(QUOTA_DAY_START_HOUR), start.tzinfo)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bounded_percent(value: Any) -> float | None:
    return float(value) if _finite_number(value) and 0 <= value <= 100 else None


def _timestamp(value: Any) -> float | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        value = value.timestamp()
    return float(value) if _finite_number(value) else None


def _source_time(account: Mapping[str, Any]) -> float | None:
    return _timestamp(account.get("account_source_timestamp", account.get("source_timestamp")))


def _normalize_accounts(accounts: Sequence[Mapping[str, Any]], now_ts: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    unknown = []
    for index, raw in enumerate(accounts):
        identity = raw.get("stable_id", raw.get("account_id"))
        if identity is None or str(identity) == "":
            unknown.append({"account_id": None, "input_index": index, "reasons": ["missing_account_id"]})
        else:
            grouped.setdefault(str(identity), []).append((index, raw))
    normalized = []
    for identity, entries in sorted(grouped.items()):
        reasons: set[str] = set()
        states = set()
        for _, raw in entries:
            remaining = raw.get("remaining_percent")
            reset_at = _timestamp(raw.get("reset_at"))
            if not _finite_number(remaining) or not 0 <= remaining <= 100:
                reasons.add("missing_or_invalid_remaining_percent")
            if reset_at is None:
                reasons.add("missing_or_invalid_reset_at")
            elif reset_at <= now_ts:
                reasons.add("overdue_reset_requires_observation")
            elif reset_at > now_ts + WEEK_SECONDS:
                reasons.add("reset_outside_weekly_cadence")
            explicit_reason = raw.get("unknown_reason") or raw.get("reason")
            if explicit_reason:
                reasons.add(str(explicit_reason))
            if raw.get("ambiguous_pool") is True:
                reasons.add("ambiguous_quota_pool")
            if _finite_number(remaining) and reset_at is not None:
                states.add((float(remaining), reset_at))
        if len(states) > 1:
            reasons.add("conflicting_duplicate_pool")
        if reasons:
            unknown.append({"account_id": identity, "input_index": entries[0][0], "reasons": sorted(reasons)})
            continue
        # Identical credentials for a single pool contribute capacity only once.
        raw = min((raw for _, raw in entries), key=lambda item: _source_time(item) or -math.inf)
        item = dict(raw)
        item.update(stable_id=identity, remaining_percent=float(raw["remaining_percent"]), reset_at=_timestamp(raw["reset_at"]))
        item["reset_credits"] = _credit_options(raw)
        if len(raw.get("reset_credits") or ()) != len(item["reset_credits"]):
            item["reset_credits_complete"] = False
        if len(entries) > 1 and any(_credit_options(other) != item["reset_credits"] for _, other in entries):
            item["reset_credits_complete"] = False
        normalized.append(item)
    return normalized, unknown


def _credit_options(account: Mapping[str, Any]) -> list[dict[str, Any]]:
    options = {}
    for credit in account.get("reset_credits") or ():
        if not isinstance(credit, Mapping):
            continue
        status = credit.get("status")
        identity = credit.get("id")
        if identity is None:
            continue
        options[str(identity)] = {
            "id": str(identity), "expires_at": _timestamp(credit.get("expires_at")),
            "status": status if isinstance(status, str) and status else None,
            "reset_type": credit.get("reset_type"),
        }
    return sorted(options.values(), key=lambda item: (item["expires_at"] or math.inf, item["id"]))


def _attach_today_budget(today: dict[str, Any], usage: Mapping[str, Any], plan_basis: str | None) -> None:
    """Publish today's fixed allocation against measured day usage.

    The allocation is the whole civil day's budget and does not shrink as the
    day is consumed. A used-versus-allocated comparison is only published when
    the allocation is frozen and every known pool's day usage was measured;
    otherwise it is withheld rather than shown as a number it is not.
    """
    allocated = _bounded_number(today.get("target"))
    used = _bounded_number(usage.get("used_percent"))
    covered = usage.get("covered_pools")
    comparable = (
        plan_basis == "crossover_frozen"
        and covered == usage.get("pool_count")
        and used is not None
    )
    today["allocated_percent"] = allocated if comparable else None
    today["usage_percent"] = used if comparable else None
    # The basis is always reported so a consumer can explain a withheld number
    # instead of showing a silent blank.
    today["usage_basis"] = usage.get("basis")
    today["usage_covered_pools"] = covered
    today["usage_pool_count"] = usage.get("pool_count")
    today["remaining_percent"] = (
        allocated - used if comparable and allocated is not None else None
    )
    today["remaining_fraction"] = (
        _budget_fraction(allocated, used) if comparable else None
    )
    today["plan_basis"] = plan_basis


def _bounded_number(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _clamp_fraction(part: float, whole: float) -> float | None:
    return min(1.0, max(0.0, part / whole)) if whole > 0 else None


def _budget_fraction(allocated: float | None, used: float | None) -> float | None:
    if allocated is None or used is None or allocated <= _EPSILON:
        return None
    # Clamped for a bar: overspending is visible in remaining_percent turning
    # negative, not in a fraction above one.
    return min(1.0, max(0.0, (allocated - used) / allocated))


def _account_result(account: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "account_id": account["stable_id"], "email": account.get("email"),
        "remaining_percent": account["remaining_percent"], "natural_reset_at": account["reset_at"],
        "source_timestamp": _source_time(account), "credit_options": account["reset_credits"],
    }


def _crossover_samples(observations: Sequence[Mapping[str, Any]], boundary: float) -> dict[str, dict[str, Any]]:
    """Select each pool's cross-over sample and the day's later counter changes.

    The store records changed states, so the latest change at or before the
    boundary is the counter value at the boundary: a later change would have
    produced a row and been selected instead. Rows after the boundary are the
    day's evidence, not cross-over anchors. Input order is not relied on.
    """
    ordered: dict[str, list[tuple[tuple[float, int], float, float, float]]] = {}
    for row in observations:
        identity = row.get("stable_id", row.get("account_id"))
        if not isinstance(identity, str) or not identity:
            continue
        remaining = _bounded_percent(row.get("remaining_percent"))
        observed = _timestamp(row.get("observed_at"))
        reset_at = _timestamp(row.get("reset_at"))
        if remaining is None or observed is None or reset_at is None:
            continue
        sequence = row.get("id", 0)
        order = (observed, sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else 0)
        ordered.setdefault(identity, []).append((order, observed, remaining, reset_at))
    samples: dict[str, dict[str, Any]] = {}
    for identity, rows in ordered.items():
        rows.sort()
        anchors = [row for row in rows if row[1] <= boundary + _EPSILON]
        if not anchors:
            continue
        _, observed, remaining, reset_at = anchors[-1]
        samples[identity] = {
            "remaining_percent": remaining, "observed_at": observed, "reset_at": reset_at,
            "changes": [(row[0], row[2]) for row in rows if row[1] > boundary + _EPSILON],
        }
    return samples


def _pool_day_usage(
    account: Mapping[str, Any], samples: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    """Return one pool's measured day usage, or why its counter cannot measure it.

    A usable measurement needs a cross-over sample from the pool's current
    natural-reset generation, and a counter that only fell since. A rise is a
    refill, correction or replacement capacity: the counter then describes new
    capacity rather than one generation's consumption, so nothing is reported.

    Usage is measured from the cross-over baseline, never from the last
    intermediate change, which in production already is the live balance.
    """
    identity = account["stable_id"]
    sample = samples.get(identity)
    if sample is None:
        return None, _NO_CROSSOVER_BASIS
    baseline = sample["remaining_percent"]
    # A moved natural anchor means the sample was taken in an already-overwritten
    # generation. Upstream anchors jitter by about a second, far below one civil
    # day of reset drift; an intra-day reset moves them by whole days.
    if abs(sample["reset_at"] - account["reset_at"]) > _ANCHOR_TOLERANCE_SECONDS:
        return None, _GENERATION_CHANGED_BASIS
    live = float(account["remaining_percent"])
    previous = baseline
    for _, value in sorted(sample["changes"]):
        if value > previous + _EPSILON:
            return None, _RISE_USAGE_BASIS
        previous = value
    if live > previous + _EPSILON:
        return None, _RISE_USAGE_BASIS
    return {
        "used_percent": baseline - live, "anchor_percent": baseline,
        "remaining_percent": live, "anchor_at": sample["observed_at"],
    }, _USAGE_BASIS


def _frozen_balances(
    accounts: Sequence[Mapping[str, Any]], samples: Mapping[str, Mapping[str, Any]]
) -> dict[str, float] | None:
    """Return cross-over balances for every pool, or None if any pool is unanchored.

    Freezing a subset would mix a fixed budget with a balance that already
    reflects unmeasured consumption, so the whole plan stays on live balances
    until every pool can be anchored at the same cross-over.
    """
    balances: dict[str, float] = {}
    for account in accounts:
        usage, _ = _pool_day_usage(account, samples)
        if usage is None:
            return None
        balances[account["stable_id"]] = usage["anchor_percent"]
    return balances


def _day_usage(
    accounts: Sequence[Mapping[str, Any]], samples: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Measure this civil day's counter drop per anchored pool, pooled honestly.

    Unusable counters are withheld instead of contributing a guessed number, so
    partial coverage is never a smaller total presented as the whole day.
    """
    pools: list[dict[str, Any]] = []
    total = 0.0
    covered = 0
    failure: str | None = None
    for account in accounts:
        entry: dict[str, Any] = {"account_id": account["stable_id"]}
        usage, reason = _pool_day_usage(account, samples)
        if usage is None:
            pools.append(dict(entry, status="unavailable", reason=reason))
            failure = failure or reason
            continue
        covered += 1
        total += usage["used_percent"]
        pools.append(dict(entry, status="measured", **usage))
    if covered == 0:
        basis = failure or _NO_CROSSOVER_BASIS
    elif covered == len(accounts):
        basis = _USAGE_BASIS
    else:
        basis = _PARTIAL_USAGE_BASIS
    return {
        "used_percent": total if covered else None, "basis": basis,
        "covered_pools": covered, "pool_count": len(accounts), "pools": pools,
    }


def _credit_scenario_reason(accounts: Sequence[Mapping[str, Any]], unknown: Sequence[Any], now_ts: float, stale_after: float) -> str | None:
    if unknown:
        return "unknown_weekly_accounts"
    for account in accounts:
        source = _source_time(account)
        if source is None or not 0 <= now_ts - source <= stale_after:
            return "quota_snapshot_unavailable_or_stale"
        if account.get("reset_credits_status") != "success" or account.get("reset_credits_stale") is not False:
            return "credit_inventory_unavailable_or_stale"
        if account.get("reset_credits_complete") is not True:
            return "credit_inventory_incomplete"
        if any(credit.get("status") != "available" for credit in account["reset_credits"]):
            return "unknown_credit_status"
    return None


def _forecast_events(accounts: Sequence[Mapping[str, Any]], end: float) -> list[_ResetEvent]:
    events = []
    for account in accounts:
        at = account["reset_at"]
        while at <= end:
            events.append(_ResetEvent(at, account["stable_id"]))
            at += WEEK_SECONDS
    return sorted(events, key=lambda event: (event.at, event.account_id))


def _next_event_for_account(state: _State, account_id: str, at: float) -> float:
    return min((event.at for event in state.events if event.account_id == account_id and event.at > at), default=math.inf)


def _consume_amount(state: _State, at: float, amount: float, contributions: dict[str, float]) -> float:
    remaining = amount
    for account_id in sorted(state.balances, key=lambda key: (_next_event_for_account(state, key, at), key)):
        taken = min(state.balances[account_id], remaining)
        if taken > 0:
            state.balances[account_id] -= taken
            contributions[account_id] = contributions.get(account_id, 0.0) + taken
            remaining -= taken
        if remaining <= 0:
            break
    return amount - remaining


def _apply_events_at(state: _State, at: float) -> list[dict[str, Any]]:
    matching = [event for event in state.events if event.at == at]
    result = []
    for event in matching:
        before = state.balances[event.account_id]
        state.balances[event.account_id] = 100.0
        result.append({
            "at": at, "account_id": event.account_id, "kind": "natural",
            "balance_before": before, "balance_after": 100.0, "capacity_added": 100.0 - before,
        })
    if matching:
        state.events = [event for event in state.events if event.at != at]
    return result


def _feasible_rate(state: _State, start: float, end: float, rate: float) -> bool:
    probe = state.copy()
    cursor = start
    while cursor < end:
        _apply_events_at(probe, cursor)
        stop = min((event.at for event in probe.events if event.at > cursor), default=end)
        stop = min(stop, end)
        wanted = rate * (stop - cursor)
        # Do not make up a pre-reset deficit with quota arriving afterward.
        if wanted > sum(probe.balances.values()):
            return False
        _consume_amount(probe, cursor, wanted, {})
        cursor = stop
    return True


def _maximum_sustainable_rate(state: _State, start: float, end: float) -> float:
    if end <= start or sum(state.balances.values()) <= 0:
        return 0.0
    # The finite first-period constraint plus the periodic capacity cap is a
    # repeatability certificate for common-period full overwrite inventories.
    high = 100.0 * len(state.balances) / WEEK_SECONDS
    low = 0.0
    for _ in range(60):
        middle = (low + high) / 2
        if _feasible_rate(state, start, end, middle):
            low = middle
        else:
            high = middle
    return low


def _consume_day(state: _State, start: float, end: float) -> dict[str, Any]:
    contributions: dict[str, float] = {}
    reset_events = _apply_events_at(state, start)
    rate = _maximum_sustainable_rate(state, start, start + WEEK_SECONDS)
    initial_rate = rate
    baseline = 0.0
    bonus = 0.0
    deadlines = []
    segments = []
    cursor = start
    while cursor < end:
        # A blocked prefix does not erase usable allocation after its refill.
        if rate == 0 and sum(state.balances.values()) > 0:
            rate = _maximum_sustainable_rate(state, cursor, cursor + WEEK_SECONDS)
        stop = min((event.at for event in state.events if event.at > cursor), default=end)
        stop = min(stop, end)
        amount = _consume_amount(state, cursor, rate * (stop - cursor), contributions)
        baseline += amount
        segment_bonus = 0.0
        for account_id in sorted({event.account_id for event in state.events if event.at == stop}):
            extra = state.balances[account_id]
            if extra > 0:
                state.balances[account_id] = 0.0
                contributions[account_id] = contributions.get(account_id, 0.0) + extra
                segment_bonus += extra
                deadlines.append({"account_id": account_id, "at": stop, "amount": extra})
        bonus += segment_bonus
        segments.append({"start_at": cursor, "end_at": stop, "baseline_allocation": amount,
                         "expiry_bonus": segment_bonus, "sustainable_daily_rate": rate * _DAY_SECONDS})
        cursor = stop
        # Events at a civil boundary belong to the following day. Their old
        # generation's optional surplus still belongs to the day just ended.
        if cursor < end:
            reset_events.extend(_apply_events_at(state, cursor))
    state.cursor = end
    return {
        "target": baseline + bonus, "baseline_allocation": baseline, "expiry_bonus": bonus,
        "sustainable_daily_rate": initial_rate * _DAY_SECONDS,
        "contributions": [{"account_id": key, "amount": value} for key, value in sorted(contributions.items()) if value > 0],
        "reset_events": reset_events, "expiry_deadlines": deadlines, "segments": segments,
    }
