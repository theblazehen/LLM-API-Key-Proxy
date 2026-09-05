"""Conditional weekly-quota credit scenarios, without provider calls or mutation.

The requested rate determines when an account is redeemed and thus changes its
future reset calendar. Feasibility is not monotone in rate: a slower consumer
can miss a credit deadline that a faster consumer reaches. Evaluate named rates;
never use this simulator as a bisection predicate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_CAPACITY = 100.0
_WEEK = 168 * 3600.0
_GUARD = 900.0
_EPSILON = 1e-7
_ASSUMPTIONS = (
    "One coherent, deduplicated pool per account; weekly pp are not token-equivalent.",
    "Weekly quota is fungible and routed earliest-reset-first; other limits and external use are unmodeled.",
    "Future natural resets are full overwrites recurring every168h, conditional on the advertised anchors.",
    "Successful credit redemption overwrites its account and replaces the old anchor with redemption+168h.",
    "Credit expiry removes an option; it does not supply quota.",
    "Redemption succeeds with zero latency in this optimistic capacity scenario; no actual availability guarantee.",
    "Real background evaluation, request latency and failures may interrupt service even when modeled demand is met.",
    "The weekly-only rescue model waits for aggregate depletion and applies900s candidate-specific natural/expiry guards.",
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inputs(
    accounts: Sequence[Mapping[str, Any]], now: float
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], str | None]:
    balances: dict[str, float] = {}
    anchors: dict[str, float] = {}
    credits: list[dict[str, Any]] = []
    credit_ids: set[tuple[str, str]] = set()
    if not accounts:
        return balances, anchors, credits, "no_known_quota_pools"
    for account in accounts:
        if not isinstance(account, Mapping):
            return {}, {}, [], "invalid_account"
        identity = account.get("stable_id", account.get("account_id"))
        if not isinstance(identity, str) or not identity:
            return {}, {}, [], "missing_pool_identity"
        if identity in balances:
            return {}, {}, [], "duplicate_pool_identity"
        remaining, anchor = account.get("remaining_percent"), account.get("reset_at")
        if not _number(remaining) or not 0 <= remaining <= _CAPACITY:
            return {}, {}, [], "invalid_weekly_balance"
        if not _number(anchor) or not now < anchor <= now + _WEEK:
            return {}, {}, [], "invalid_or_overdue_natural_anchor"
        source_at = account.get("source_timestamp", account.get("account_source_timestamp"))
        if not _number(source_at) or not 0 <= now - source_at <= _GUARD:
            return {}, {}, [], "quota_snapshot_unavailable_or_stale"
        if account.get("reset_credits_status") != "success":
            return {}, {}, [], "credit_inventory_unavailable"
        if account.get("reset_credits_stale") is not False:
            return {}, {}, [], "credit_inventory_stale_or_unknown"
        if account.get("reset_credits_complete") is not True:
            return {}, {}, [], "credit_inventory_incomplete"
        inventory = account.get("reset_credits")
        if not isinstance(inventory, (list, tuple)):
            return {}, {}, [], "invalid_credit_inventory"
        balances[identity], anchors[identity] = float(remaining), float(anchor)
        for credit in inventory:
            if not isinstance(credit, Mapping):
                return {}, {}, [], "invalid_credit_option"
            status = credit.get("status")
            if not isinstance(status, str):
                return {}, {}, [], "unknown_credit_status"
            if status != "available":
                continue
            reset_type = credit.get("reset_type")
            if not isinstance(reset_type, str) or not reset_type:
                return {}, {}, [], "unknown_credit_reset_scope"
            if reset_type != "codex_rate_limits":
                # Other benefits are not full weekly reset inventory.
                continue
            credit_id, expiry = credit.get("id"), credit.get("expires_at")
            if not isinstance(credit_id, str) or not credit_id or not _number(expiry):
                return {}, {}, [], "invalid_credit_identity_or_expiry"
            key = (identity, credit_id)
            if key in credit_ids:
                return {}, {}, [], "duplicate_credit_identity"
            credit_ids.add(key)
            credits.append({"account": identity, "id": credit_id, "expires_at": float(expiry)})
    return dict(sorted(balances.items())), dict(sorted(anchors.items())), sorted(
        credits, key=lambda c: (c["expires_at"], c["account"], c["id"])
    ), None


def _record(now: float | None, hours: float, rate: float) -> dict[str, Any]:
    return {
        "status": "ready", "reason": None, "conditional": True,
        "start": now, "end": now + hours * 3600 if now is not None else None,
        "horizon_hours": hours, "rate_pp_day": rate,
        "continuous_feasible_in_optimistic_envelope": None,
        "actual_continuous_availability_guarantee": False,
        "credits_redeemed": 0, "consumed": None, "unfulfilled": None,
        "first_interruption": None, "interruptions": [], "trace": [],
        "assumptions": list(_ASSUMPTIONS),
    }


def _simulate(
    balances: dict[str, float], anchors: dict[str, float], credits: list[dict[str, Any]],
    now: float, hours: float, rate_pp_day: float,
) -> dict[str, Any]:
    result = _record(now, hours, rate_pp_day)
    end, rate = result["end"], rate_pp_day / 86400.0
    values, natural, pending = balances.copy(), anchors.copy(), list(credits)
    initial = sum(values.values())
    consumed = granted = discarded = 0.0
    trace, interruptions = result["trace"], result["interruptions"]
    while now < end:
        # Half-open consumption intervals: a reset exactly at the horizon does
        # not finance any earlier demand. Natural resets win redemption ties.
        for account in natural:
            if natural[account] <= now:
                before = values[account]
                values[account] = _CAPACITY
                discarded += before
                granted += _CAPACITY
                natural[account] += _WEEK
                trace.append({"kind": "natural_reset", "at": now, "account": account,
                              "discarded": before, "grant": _CAPACITY,
                              "capacity_added": _CAPACITY - before,
                              "next_anchor": natural[account]})
        expired = [credit for credit in pending if credit["expires_at"] <= now]
        for credit in expired:
            trace.append({"kind": "credit_expired_without_refill", "at": now, **credit})
        pending = [credit for credit in pending if credit["expires_at"] > now]
        available = sum(values.values())
        if available <= _EPSILON and rate > 0:
            eligible = [credit for credit in pending
                        if values[credit["account"]] <= _EPSILON
                        and credit["expires_at"] - now > _GUARD
                        and natural[credit["account"]] - now > _GUARD]
            if eligible:
                credit = eligible[0]  # Pending remains earliest-expiry sorted.
                account = credit["account"]
                old, before = natural[account], values[account]
                discarded += before
                values[account] = _CAPACITY
                natural[account] = now + _WEEK
                pending.remove(credit)
                granted += _CAPACITY
                result["credits_redeemed"] += 1
                trace.append({"kind": "conditional_credit_redemption", "at": now, **credit,
                              "suppressed_natural_anchor": old, "next_anchor": natural[account],
                              "discarded": before, "grant": _CAPACITY,
                              "capacity_added": _CAPACITY - before})
                continue
            stop = min(end, min(natural.values()),
                       min((credit["expires_at"] for credit in pending), default=end))
            interruption = {
                "kind": "interruption", "start": now, "end": stop,
                "unfulfilled": rate * (stop - now), "reason": "no_policy_eligible_credit",
                "candidate_guards": [
                    {"account": credit["account"], "credit": credit["id"],
                     "natural_guard": natural[credit["account"]] - now <= _GUARD,
                     "expiry_guard": credit["expires_at"] - now <= _GUARD}
                    for credit in pending
                ],
            }
            interruptions.append(interruption)
            trace.append(interruption)
            now = stop
            continue
        stop = min(end, min(natural.values()),
                   min((credit["expires_at"] for credit in pending), default=end))
        if rate > 0:
            stop = min(stop, now + available / rate)
        if stop <= now:
            raise ArithmeticError("credit scenario failed to advance time")
        wanted = demand = rate * (stop - now)
        contributions = []
        for account in sorted(values, key=lambda key: (natural[key], key)):
            amount = min(values[account], wanted)
            if amount > 0:
                values[account] -= amount
                wanted -= amount
                contributions.append({"account": account, "amount": amount})
        if wanted > _EPSILON:
            raise ArithmeticError("credit scenario overallocated inventory")
        consumed += demand - wanted
        trace.append({"kind": "consumption", "start": now, "end": stop,
                      "amount": demand - wanted, "contributions": contributions,
                      "balances_after": values.copy()})
        now = stop
    unmet = sum(item["unfulfilled"] for item in interruptions)
    if not math.isclose(initial + granted - discarded, consumed + sum(values.values()),
                        rel_tol=1e-10, abs_tol=_EPSILON):
        raise ArithmeticError("credit scenario inventory conservation failed")
    if not math.isclose(consumed + unmet, rate * hours * 3600,
                        rel_tol=1e-10, abs_tol=_EPSILON):
        raise ArithmeticError("credit scenario demand conservation failed")
    result.update({
        "continuous_feasible_in_optimistic_envelope": not interruptions,
        "consumed": consumed, "unfulfilled": unmet,
        "first_interruption": interruptions[0]["start"] if interruptions else None,
        "final_balances": values, "final_anchors": natural, "credits_remaining": pending,
        "initial": initial, "granted": granted, "discarded": discarded,
    })
    return result


def build_credit_scenarios(
    accounts: Sequence[Mapping[str, Any]], now_ts: float,
    horizons: Sequence[float] = (168, 336), rates: Sequence[float] = (25, 30, 40, 50, 60),
) -> list[dict[str, Any]]:
    """Evaluate explicit rates; bad observations yield unavailable, never refills.

    Caller-supplied rate/horizon definitions must be finite nonnegative rates
    and positive horizons. Freshness is evaluated once at the input cutoff;
    future scenario inventory is hypothetical, not refreshed provider data.
    """
    if any(not _number(hours) or hours <= 0 for hours in horizons):
        raise ValueError("scenario horizons must be finite positive hours")
    if any(not _number(rate) or rate < 0 for rate in rates):
        raise ValueError("scenario rates must be finite nonnegative pp/day")
    now = float(now_ts) if _number(now_ts) else None
    if now is None:
        balances, anchors, credits, reason = {}, {}, [], "invalid_snapshot_time"
    else:
        balances, anchors, credits, reason = _inputs(accounts, now)
    results = []
    for hours in horizons:
        for rate in rates:
            if reason:
                record = _record(now, hours, rate)
                record.update({"status": "unavailable", "reason": reason})
            else:
                record = _simulate(balances, anchors, credits, now, hours, rate)
            results.append(record)
    return results
