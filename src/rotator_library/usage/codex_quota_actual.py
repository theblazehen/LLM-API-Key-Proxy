"""Retrospective counter bounds, independent of future allocation.

Trusted ingestion establishes provenance, not complete upstream event history.
Finite usage bounds require explicit event completeness and measurement precision.
Within each reset generation a monotone chain telescopes: tighten all observation
intervals jointly, then bound the difference at the reporting cuts. No optimizer,
per-edge rounding accumulation, or balance-rise reset inference is needed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


_MODEL_ASSUMPTIONS = [
    "One coherent quota-pool counter; no refunds, corrections or family changes",
    "Complete explicitly supplied overwrite history across the evidence span",
    "Explicit measurement precision; normalized account-week percentage points",
    "Consumption only on positive-duration intervals, without a finite rate cap",
]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _identity(raw: Mapping[str, Any]) -> Any:
    return raw.get("stable_id", raw.get("account_id", raw.get("stable_account_id")))


def _chain_bounds(
    points: list[tuple[float, float, float, str | None]],
    resets: set[float], start: float, end: float,
) -> tuple[float, float]:
    """Solve independent monotone generations with shared boundary variables."""
    times = sorted({start, end, *resets, *(point[0] for point in points)})
    constraints: dict[tuple[float, str | None], tuple[float, float]] = {}
    for at, lower, upper, side in points:
        key = (at, side)
        previous = constraints.get(key, (0.0, 100.0))
        constraints[key] = (max(previous[0], lower), min(previous[1], upper))
    chains: list[list[tuple[float, float, float]]] = [[]]
    for at in times:
        if at in resets:
            lower, upper = constraints.get((at, "pre"), (0.0, 100.0))
            chains[-1].append((at, lower, upper))
            lower, upper = constraints.get((at, "post"), (100.0, 100.0))
            chains.append([(at, max(100.0, lower), min(100.0, upper))])
        else:
            lower, upper = constraints.get((at, None), (0.0, 100.0))
            chains[-1].append((at, lower, upper))
    minimum = maximum = 0.0
    for chain in chains:
        lower = [node[1] for node in chain]
        upper = [node[2] for node in chain]
        # Later lower bounds constrain earlier states; earlier upper bounds
        # constrain later states. All evidence shares the same latent balances.
        for i in range(len(chain) - 2, -1, -1):
            lower[i] = max(lower[i], lower[i + 1])
        for i in range(1, len(chain)):
            upper[i] = min(upper[i], upper[i - 1])
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("inconsistent_counter_history")
        included = [i for i, node in enumerate(chain) if start <= node[0] <= end]
        if len(included) < 2:
            continue
        left, right = included[0], included[-1]
        minimum += max(0.0, lower[left] - upper[right])
        maximum += upper[left] - lower[right]
    return minimum, maximum


def _account_bounds(
    account: Mapping[str, Any], rows: list[Mapping[str, Any]],
    start: float, end: float,
) -> dict[str, Any]:
    pool = _identity(account)
    result: dict[str, Any] = {
        "account_id": pool, "status": "unavailable", "reason": "no_observations",
        "lower_bound": None, "upper_bound": None, "estimate": None,
        "observed_decrease": 0.0, "baseline_at": None, "latest_at": None,
    }
    def unavailable(reason: str) -> dict[str, Any]:
        return dict(result, reason=reason)

    if account.get("history_provenance") != "trusted_ingestion":
        return unavailable("untrusted_account_provenance")
    ordered: list[Mapping[str, Any]] = []
    last_key: tuple[float, int] | None = None
    for row in rows:
        at = _number(row.get("observed_at"))
        if at is None:
            return unavailable("invalid_observation_timestamp")
        if at > end:
            continue
        sequence = row.get("id", 0)
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            return unavailable("invalid_observation_sequence")
        key = (at, sequence)
        if last_key is not None and key < last_key:
            return unavailable("out_of_order_observations")
        last_key = key
        ordered.append(row)
    baselines = [i for i, row in enumerate(ordered) if row["observed_at"] <= start]
    if not baselines:
        return unavailable("baseline_unavailable")
    # Retain all ties at the boundary baseline so conflicting simultaneous
    # evidence cannot be hidden by choosing the final insertion ID.
    baseline_index = baselines[-1]
    baseline_time = ordered[baseline_index]["observed_at"]
    while baseline_index and ordered[baseline_index - 1]["observed_at"] == baseline_time:
        baseline_index -= 1
    ordered = ordered[baseline_index:]
    result["baseline_at"] = float(baseline_time)
    points: list[tuple[float, float, float, str | None]] = []
    previous: tuple[float, float, str | None] | None = None
    precision = _number(account.get("measurement_error_pp"))
    if account.get("measurement_model") == "exact":
        precision = 0.0
    precision_known = precision is not None and 0 <= precision <= 100
    for row in ordered:
        if row.get("source") not in {"api", "headers"}:
            return unavailable("legacy_or_untrusted_history")
        if row.get("pool_id") != pool:
            return unavailable("mixed_pool_history")
        at = float(row["observed_at"])
        balance = _number(row.get("remaining_percent"))
        source_at = _number(row.get("source_timestamp"))
        confirmed = _number(row.get("confirmed_at"))
        if balance is None or not 0 <= balance <= 100:
            return unavailable("invalid_counter_balance")
        if source_at is None or source_at != at or confirmed is None or confirmed < at:
            return unavailable("incoherent_observation_timestamps")
        side = row.get("side")
        if side not in {None, "pre", "post"}:
            return unavailable("invalid_observation_side")
        if previous is not None:
            if at > previous[0] and at > start:
                result["observed_decrease"] += max(0.0, previous[1] - balance)
        previous = (at, balance, side)
        error = precision if precision_known else 0.0
        points.append((at, max(0.0, balance - error), min(100.0, balance + error), side))
        # Confirmation is an endpoint, not evidence that balance stayed flat
        # continuously. Never interpolate a fresh endpoint across the day cut.
        if at < confirmed <= end:
            points.append((confirmed, max(0.0, balance - error), min(100.0, balance + error), None))
        result["latest_at"] = max(result["latest_at"] or at, at, confirmed if confirmed <= end else at)

    current_at = _number(account.get("source_timestamp"))
    current_balance = _number(account.get("remaining_percent"))
    if current_at is not None and current_at <= end and current_at >= (result["latest_at"] or start):
        if current_balance is None or not 0 <= current_balance <= 100:
            return unavailable("invalid_current_balance")
        error = precision if precision_known else 0.0
        current_side = account.get("observation_side")
        if current_side not in {None, "pre", "post"}:
            return unavailable("invalid_observation_side")
        points.append((current_at, max(0.0, current_balance - error), min(100.0, current_balance + error), current_side))
        result["latest_at"] = current_at
    if account.get("event_history_complete") is not True:
        return unavailable("reset_history_incomplete")
    if not precision_known:
        return unavailable("measurement_precision_unestablished")
    # Different reported centers can describe the same latent balance. Apply
    # the measurement model before deciding whether simultaneous evidence
    # conflicts, including repeated confirmations and the current snapshot.
    simultaneous: dict[tuple[float, str | None], tuple[float, float]] = {}
    for at, lower, upper, side in points:
        key = (at, side)
        earlier_lower, earlier_upper = simultaneous.get(key, (0.0, 100.0))
        intersection = (max(lower, earlier_lower), min(upper, earlier_upper))
        if intersection[0] > intersection[1]:
            return unavailable("conflicting_same_time_observations")
        simultaneous[key] = intersection
    reset_values = account.get("confirmed_resets", ())
    if not isinstance(reset_values, (list, tuple)):
        return unavailable("invalid_reset_history")
    resets: set[float] = set()
    for raw in reset_values:
        at = _number(raw.get("at") if isinstance(raw, Mapping) else raw)
        if at is None:
            return unavailable("invalid_reset_timestamp")
        if baseline_time <= at <= end:
            resets.add(at)
    if any((at in resets and side is None) or (at not in resets and side is not None)
           for at, _, _, side in points):
        return unavailable("ambiguous_reset_observation_order")
    try:
        lower, upper = _chain_bounds(points, resets, start, end)
    except ValueError as error:
        return unavailable(str(error))
    return dict(result, status="estimated", reason="conditional_counter_bounds",
                lower_bound=lower, upper_bound=upper,
                estimate=lower if lower == upper else None)


def derive_usage_observation(
    accounts: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
    day_start: float, now_ts: float,
) -> dict[str, Any]:
    """Return explicit conditional bounds, or explain why usage is unidentified.

    Optional per-account inputs are event_history_complete=True,
    confirmed_resets=[epoch|{'at': epoch}], and measurement_model='exact' or
    measurement_error_pp=<known error>. These are caller assertions, not facts
    inferred from sample shape. Observation side='pre'/'post' resolves reset ties.
    estimate stays null for a non-singleton range: the midpoint is not measured.
    """
    start, end = _number(day_start), _number(now_ts)
    if start is None or end is None or end < start:
        raise ValueError("invalid retrospective reporting window")
    result: dict[str, Any] = {
        "status": "unavailable", "estimate": None, "lower_bound": None,
        "upper_bound": None, "reason": "no_accounts", "assumptions": [],
        "coverage": {"start_at": start, "end_at": end, "accounts": []},
        "observed_decrease": None,
        "observed_decrease_basis": "positive_sampled_counter_movements_not_daily_usage",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        identity = _identity(row)
        if isinstance(identity, str):
            grouped.setdefault(identity, []).append(row)
    seen: set[str] = set()
    coverage = result["coverage"]["accounts"]
    for account in accounts:
        identity = _identity(account)
        if not isinstance(identity, str) or not identity:
            result["reason"] = "unknown_pool_identity"
            return result
        if identity in seen:
            result["reason"] = "duplicate_pool_identity"
            return result
        seen.add(identity)
        coverage.append(_account_bounds(account, grouped.get(identity, []), start, end))
    if not coverage:
        return result
    result["observed_decrease"] = sum(item["observed_decrease"] for item in coverage)
    unavailable = [item for item in coverage if item["status"] != "estimated"]
    if unavailable:
        result["reason"] = unavailable[0]["reason"]
        return result
    lower = sum(item["lower_bound"] for item in coverage)
    upper = sum(item["upper_bound"] for item in coverage)
    result.update(status="estimated", reason="conditional_counter_bounds",
                  lower_bound=lower, upper_bound=upper,
                  estimate=lower if lower == upper else None,
                  assumptions=list(_MODEL_ASSUMPTIONS))
    return result
