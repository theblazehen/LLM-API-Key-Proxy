"""Named-rate credit policy scenarios are conditional, conservative and nonmonotone."""

from __future__ import annotations

import importlib
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest


_PACKAGE = types.ModuleType("_quota_scenario_contract")
_PACKAGE.__path__ = [str(Path(__file__).parents[1] / "src/rotator_library/usage")]
sys.modules.setdefault(_PACKAGE.__name__, _PACKAGE)
build_credit_scenarios = importlib.import_module(
    f"{_PACKAGE.__name__}.codex_quota_scenarios"
).build_credit_scenarios

HOUR = 3600
WEEK = 168 * HOUR


def pool(identity="a", remaining=10, reset_hours=100, *, expiry_hours=500):
    return {
        "stable_id": identity, "remaining_percent": remaining,
        "reset_at": reset_hours * HOUR, "source_timestamp": 0,
        "reset_credits_status": "success", "reset_credits_stale": False,
        "reset_credits_complete": True,
        "reset_credits": [{"id": f"credit-{identity}", "expires_at": expiry_hours * HOUR,
                           "status": "available", "reset_type": "codex_rate_limits"}],
    }


def scenario(accounts, *, hours=40, rate=24):
    return build_credit_scenarios(accounts, 0, horizons=(hours,), rates=(rate,))[0]


def test_faster_demand_can_unlock_credit_that_slower_demand_misses():
    inputs = [pool(expiry_hours=8)]
    slow, fast = build_credit_scenarios(inputs, 0, horizons=(40,), rates=(24, 48))
    assert slow["continuous_feasible_in_optimistic_envelope"] is False
    assert slow["credits_redeemed"] == 0
    assert slow["first_interruption"] == 10 * HOUR
    assert fast["continuous_feasible_in_optimistic_envelope"] is True
    assert fast["credits_redeemed"] == 1
    assert fast["consumed"] == pytest.approx(80)
    redemption = next(item for item in fast["trace"] if item["kind"] == "conditional_credit_redemption")
    assert redemption["at"] == 5 * HOUR
    assert fast["conditional"] is True
    assert fast["actual_continuous_availability_guarantee"] is False
    assert fast["assumptions"]


def test_successful_conditional_redemption_replaces_old_natural_anchor():
    result = scenario([pool()], hours=200)
    redemptions = [item for item in result["trace"] if item["kind"] == "conditional_credit_redemption"]
    assert len(redemptions) == 1
    assert redemptions[0]["at"] == 10 * HOUR
    assert redemptions[0]["suppressed_natural_anchor"] == 100 * HOUR
    assert redemptions[0]["next_anchor"] == 178 * HOUR
    assert [item["at"] for item in result["trace"] if item["kind"] == "natural_reset"] == [178 * HOUR]


def test_natural_reset_wins_exact_depletion_redemption_tie():
    result = scenario([pool(reset_hours=10)], hours=20)
    assert result["continuous_feasible_in_optimistic_envelope"] is True
    assert result["credits_redeemed"] == 0
    assert result["consumed"] == pytest.approx(20)
    assert any(item["kind"] == "natural_reset" and item["at"] == 10 * HOUR for item in result["trace"])


@pytest.mark.parametrize(("guard", "seconds", "redeemed"), [("natural", 900, 0), ("natural", 901, 1), ("expiry", 900, 0), ("expiry", 901, 1)])
def test_natural_and_credit_expiry_guards_are_strict_and_candidate_specific(guard, seconds, redeemed):
    inputs = pool(remaining=0)
    if guard == "natural":
        inputs["reset_at"] = seconds
    else:
        inputs["reset_credits"][0]["expires_at"] = seconds
    result = scenario([inputs], hours=1)
    assert result["credits_redeemed"] == redeemed
    if redeemed == 0:
        assert result["first_interruption"] == 0
        assert result["actual_continuous_availability_guarantee"] is False


def test_credit_expiry_cannot_supply_capacity():
    result = scenario([pool(remaining=0, expiry_hours=0.25)], hours=1)
    assert result["credits_redeemed"] == 0
    assert result["consumed"] == 0 and result["unfulfilled"] == pytest.approx(1)
    assert any(item["kind"] == "credit_expired_without_refill" and item["at"] == 900 for item in result["trace"])
    assert result["final_balances"] == {"a": 0}


def test_reset_exactly_at_horizon_cannot_finance_earlier_demand():
    inputs = pool(remaining=0, reset_hours=1)
    inputs["reset_credits"] = []
    result = scenario([inputs], hours=1)
    assert result["consumed"] == 0 and result["unfulfilled"] == pytest.approx(1)
    assert not any(item["kind"] == "natural_reset" for item in result["trace"])


def test_earliest_expiring_eligible_credit_is_chosen_not_guarded_candidate():
    guarded = pool("guarded", remaining=0, reset_hours=0.25, expiry_hours=2)
    sooner = pool("sooner", remaining=0, expiry_hours=10)
    later = pool("later", remaining=0, expiry_hours=20)
    result = scenario([later, guarded, sooner], hours=1)
    redemption = next(item for item in result["trace"] if item["kind"] == "conditional_credit_redemption")
    assert redemption["id"] == "credit-sooner"
    assert redemption["at"] == 0


@pytest.mark.parametrize("rate", [0, 25, 30, 40, 50, 60])
def test_named_scenarios_conserve_demand_and_overwrite_inventory(rate):
    inputs = [pool("a", remaining=61, reset_hours=20), pool("b", remaining=0, reset_hours=150)]
    original = deepcopy(inputs)
    result = scenario(inputs, hours=336, rate=rate)
    assert result["status"] == "ready"
    assert result["consumed"] + result["unfulfilled"] == pytest.approx(rate * 14)
    assert result["initial"] + result["granted"] - result["discarded"] == pytest.approx(result["consumed"] + sum(result["final_balances"].values()))
    for item in result["trace"]:
        if item["kind"] in {"natural_reset", "conditional_credit_redemption"}:
            assert item["grant"] == 100
            assert item["capacity_added"] == pytest.approx(100 - item["discarded"])
        elif item["kind"] == "consumption":
            assert sum(part["amount"] for part in item["contributions"]) == pytest.approx(item["amount"])
            assert all(-1e-7 <= balance <= 100 for balance in item["balances_after"].values())
    assert inputs == original
    assert scenario(inputs, hours=336, rate=rate) == result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("source_timestamp", None, "quota_snapshot_unavailable_or_stale"),
    ("source_timestamp", -901, "quota_snapshot_unavailable_or_stale"),
    ("reset_credits_status", "error", "credit_inventory_unavailable"),
    ("reset_credits_stale", None, "credit_inventory_stale_or_unknown"),
    ("reset_credits_complete", False, "credit_inventory_incomplete"),
    ("reset_at", 0, "invalid_or_overdue_natural_anchor"),
    ("reset_at", WEEK + 1, "invalid_or_overdue_natural_anchor"),
])
def test_unknown_or_invalid_inputs_are_not_dry_capacity_simulations(field, value, reason):
    inputs = dict(pool(), **{field: value})
    result = scenario([inputs])
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["consumed"] is None and result["unfulfilled"] is None
    assert result["continuous_feasible_in_optimistic_envelope"] is None
    assert result["trace"] == []


def test_missing_credit_status_is_unknown_not_available():
    inputs = pool(remaining=0)
    inputs["reset_credits"][0].pop("status")
    result = scenario([inputs])
    assert result["status"] == "unavailable"
    assert result["reason"] == "unknown_credit_status"
    assert result["credits_redeemed"] == 0


def test_nonweekly_credit_scope_does_not_restore_weekly_quota():
    inputs = pool(remaining=0)
    inputs["reset_credits"][0]["reset_type"] = "different_benefit"
    result = scenario([inputs], hours=1)
    assert result["credits_redeemed"] == 0
    assert result["consumed"] == 0


def test_duplicate_pool_is_rejected_instead_of_double_counted():
    inputs = pool()
    result = scenario([inputs, deepcopy(inputs)])
    assert result["status"] == "unavailable" and result["reason"] == "duplicate_pool_identity"


@pytest.mark.parametrize(("horizons", "rates"), [((0,), (25,)), ((float("nan"),), (25,)), ((168,), (-1,)), ((168,), (float("inf"),))])
def test_invalid_scenario_definitions_are_rejected(horizons, rates):
    with pytest.raises(ValueError):
        build_credit_scenarios([pool()], 0, horizons=horizons, rates=rates)
