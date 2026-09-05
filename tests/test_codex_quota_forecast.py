"""Behavioral contracts for forward-only weekly quota plans and usage evidence."""

from __future__ import annotations

import importlib
import json
import sys
import types
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


# Load the pure usage modules as a package, without importing the API client or
# replacing rotator_library in sys.modules for unrelated tests.
_PACKAGE = types.ModuleType("_quota_forecast_contract")
_PACKAGE.__path__ = [str(Path(__file__).parents[1] / "src/rotator_library/usage")]
sys.modules.setdefault(_PACKAGE.__name__, _PACKAGE)
build_codex_quota_forecast = importlib.import_module(
    f"{_PACKAGE.__name__}.codex_quota_forecast"
).build_codex_quota_forecast

UTC = timezone.utc
DAY = 86400
WEEK = 7 * DAY


def instant(day: int, hour: int = 6, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=UTC)


def account(account_id, remaining, reset_at, **extra):
    return {
        "stable_id": account_id,
        "remaining_percent": remaining,
        "reset_at": reset_at.timestamp(),
        **extra,
    }


def forecast(accounts, *, now=instant(8, 12), observations=()):
    return build_codex_quota_forecast(
        accounts=accounts, observations=observations, now=now,
        source_timestamp=now.timestamp(),
    )


def observation(at, remaining, *, pool="a", sequence=1, **extra):
    return {
        "stable_id": pool, "pool_id": pool, "id": sequence,
        "observed_at": at.timestamp(), "source_timestamp": at.timestamp(),
        "confirmed_at": at.timestamp(), "source": "api",
        "remaining_percent": remaining, **extra,
    }


def test_local_six_am_boundary_starts_and_labels_quota_day():
    accounts = [account("a", 50, instant(9, 18))]
    before = forecast(accounts, now=instant(8, 5, 59))
    at = forecast(accounts, now=instant(8, 6))
    local = timezone(timedelta(hours=2))
    local_before = forecast(accounts, now=datetime(2026, 8, 8, 5, 59, tzinfo=local))
    assert before["horizon"]["start_at"] == instant(7).timestamp()
    assert at["horizon"]["start_at"] == instant(8).timestamp()
    assert at["planning_horizon"]["end_at"] == instant(22).timestamp()
    assert at["planning_horizon"]["rolling_window_seconds"] == WEEK
    assert len(at["days"]) == 7 and len(at["planning_days"]) == 14
    assert at["planning_days"][:7] == at["days"]
    assert [day["local_date"] for day in at["days"]] == [f"2026-08-{d:02}" for d in range(8, 15)]
    assert at["quota_day_start_hour"] == 6
    assert local_before["horizon"]["start_at"] == datetime(2026, 8, 7, 6, tzinfo=local).timestamp()
    post = next(day for day in at["planning_days"] if day["start_at"] == at["post_reset"]["day_start_at"])
    assert at["post_reset"]["daily_sustainable_pace"] == post["sustainable_daily_rate"]


@pytest.mark.parametrize(("month", "day", "hours"), [(3, 7, 23), (10, 31, 25)])
def test_dst_civil_day_integrates_actual_elapsed_time(month, day, hours):
    zone = ZoneInfo("America/New_York")
    now = datetime(2026, month, day, 6, tzinfo=zone)
    reset = datetime.fromtimestamp(now.timestamp() + 6 * DAY, zone)
    result = forecast([account("a", 100, reset)], now=now)
    today = result["today"]
    assert today["end_at"] - today["start_at"] == hours * 3600
    assert today["expiry_bonus"] == 0
    assert today["baseline_allocation"] == pytest.approx(today["sustainable_daily_rate"] * hours / 24)
    assert result["days"][1]["start_at"] == today["end_at"]


def test_first_run_has_forward_plan_without_inventing_historical_usage():
    result = forecast([account("a", 74.375, instant(9, 18))])
    assert result["schema_version"] == 3 and result["status"] == "ready"
    assert result["today"]["target"] > 0
    assert result["actual"]["status"] == "unavailable"
    assert result["actual"]["estimate"] is None
    assert result["actual"]["lower_bound"] is None
    assert result["accounts"][0]["remaining_percent"] == 74.375


def test_saved_live_snapshot_matches_independent_repeatable_pace():
    now = datetime.fromtimestamp(1788637959.2294838, UTC)
    accounts = [
        account("identity-1", 61, datetime.fromtimestamp(1789231320, UTC)),
        account("identity-4", 0, datetime.fromtimestamp(1788747971, UTC)),
    ]
    result = forecast(accounts, now=now)
    assert result["today"]["sustainable_daily_rate"] == pytest.approx(23.44341030145878, abs=1e-7)
    assert result["today"]["baseline_allocation"] == pytest.approx(9.887684, abs=1e-5)
    assert result["today"]["expiry_bonus"] == 0
    assert result["today"]["segments"][0]["start_at"] == now.timestamp()


@pytest.mark.parametrize(("reset_hours", "expected_rate"), [(84, 100 / 7), (168, 50 / 7)])
def test_finite_week_horizon_cliff_is_not_mistaken_for_repeatable_pace(reset_hours, expected_rate):
    now = instant(8, 6)
    result = forecast([account("a", 50, now + timedelta(hours=reset_hours))], now=now)
    # At84h a finite168h plan can run100/7 pp/day, but the common-period cap
    # still forbids rates above100/7. At168h the refill cannot fund earlier use.
    assert result["today"]["sustainable_daily_rate"] == pytest.approx(expected_rate)
    assert all(day["sustainable_daily_rate"] <= 100 / 7 + 1e-8 for day in result["planning_days"])


def test_finite_horizon_windfall_does_not_raise_the_repeatable_cap():
    now = instant(8, 6)
    result = forecast([account("a", 100, now + timedelta(hours=84))], now=now)
    # Finite168h demand can consume200pp at200/7 daily; that rate cannot repeat.
    assert result["today"]["sustainable_daily_rate"] == pytest.approx(100 / 7)


def test_each_day_conserves_inventory_and_contributions():
    accounts = [account("late", 33.25, instant(12)), account("early", 61.75, instant(9, 18))]
    result = forecast(accounts)
    balances = {item["stable_id"]: item["remaining_percent"] for item in accounts}
    for day in result["planning_days"]:
        assert day["target"] == pytest.approx(day["baseline_allocation"] + day["expiry_bonus"])
        assert sum(item["amount"] for item in day["contributions"]) == pytest.approx(day["target"])
        assert sum(segment["baseline_allocation"] for segment in day["segments"]) == pytest.approx(day["baseline_allocation"])
        assert sum(item["amount"] for item in day["expiry_deadlines"]) == pytest.approx(day["expiry_bonus"])
        for event in day["reset_events"]:
            assert event["kind"] == "natural"
            assert event["balance_after"] == 100
            assert event["capacity_added"] == pytest.approx(100 - event["balance_before"])
            balances[event["account_id"]] += event["capacity_added"]
        for item in day["contributions"]:
            assert item["amount"] > 0
            balances[item["account_id"]] -= item["amount"]
        assert all(-1e-7 <= value <= 100 + 1e-7 for value in balances.values())
    assert result["today"]["contributions"] == [{"account_id": "early", "amount": result["today"]["target"]}]


def test_expiry_bonus_belongs_only_to_old_generation_and_exact_deadline():
    now = instant(8, 12)
    reset = now + timedelta(seconds=1234.56789)
    before = forecast([account("a", 100, reset)], now=now)
    today = before["today"]
    deadline = today["expiry_deadlines"][0]
    assert deadline["at"] == reset.timestamp()
    pre_baseline = today["segments"][0]["baseline_allocation"]
    assert deadline["amount"] == pytest.approx(100 - pre_baseline)
    assert all(segment["expiry_bonus"] == 0 for segment in today["segments"] if segment["start_at"] >= reset.timestamp())
    after = forecast([account("a", 100, reset + timedelta(days=7))], now=reset)
    assert after["today"]["expiry_bonus"] == 0
    assert after["today"]["target"] < today["target"]


def test_simultaneous_resets_apply_once_per_pool_and_preserve_surplus():
    reset = instant(8, 18)
    result = forecast([account("a", 40, reset), account("b", 60, reset)])
    today = result["today"]
    assert [(event["account_id"], event["at"]) for event in today["reset_events"]] == [("a", reset.timestamp()), ("b", reset.timestamp())]
    pre_reset = today["segments"][0]
    assert pre_reset["baseline_allocation"] + pre_reset["expiry_bonus"] == pytest.approx(100)
    assert len(today["expiry_deadlines"]) == 2


def test_every_same_day_reset_is_applied_and_reported():
    accounts = [account("a", 5, instant(8, 14)), account("b", 5, instant(8, 20)), account("c", 90, instant(12))]
    result = forecast(accounts)
    control = forecast([accounts[0], account("b", 5, instant(15, 12)), accounts[2]])
    assert [(event["account_id"], event["at"]) for event in result["today"]["reset_events"]] == [("a", instant(8, 14).timestamp()), ("b", instant(8, 20).timestamp())]
    assert sum(day["target"] for day in result["days"]) > sum(day["target"] for day in control["days"])


def test_reset_at_civil_boundary_is_reported_in_new_day_but_old_bonus_is_not():
    boundary = instant(9)
    result = forecast([account("a", 40, boundary)])
    assert result["days"][0]["reset_events"] == []
    assert result["days"][0]["expiry_deadlines"][0]["at"] == boundary.timestamp()
    assert [(event["account_id"], event["at"]) for event in result["days"][1]["reset_events"]] == [("a", boundary.timestamp())]
    assert result["days"][1]["expiry_bonus"] == 0


@pytest.mark.parametrize("reset", [instant(8, 18), instant(9, 18)])
def test_blocked_prefix_does_not_erase_post_reset_plan(reset):
    result = forecast([account("a", 0, reset)])
    assert result["status"] == "blocked"
    assert result["blocked_until"] == reset.timestamp()
    assert result["risk"]["aggregate_exhaustion"] is True
    segments = [segment for day in result["days"] for segment in day["segments"]]
    assert all(segment["baseline_allocation"] == 0 for segment in segments if segment["end_at"] <= reset.timestamp())
    assert any(segment["baseline_allocation"] > 0 for segment in segments if segment["start_at"] >= reset.timestamp())


@pytest.mark.parametrize("remaining", [1e-5, 1, 10, 33.333333333, 40])
def test_positive_pool_is_not_exhausted_by_rounding_or_solver_residue(remaining):
    result = forecast([account("a", remaining, instant(8, 18))])
    assert result["status"] == "ready"
    assert result["risk"]["aggregate_exhaustion"] is False
    assert result["today"]["sustainable_daily_rate"] > 0


def test_unknown_pool_makes_known_subset_partial_not_complete_or_exhausted():
    result = forecast([account("known", 0, instant(8, 18)), {"stable_id": "unknown"}])
    assert result["status"] == "partial"
    assert result["today"] is not None
    assert result["risk"]["aggregate_exhaustion"] is None
    assert result["unknown_accounts"][0]["account_id"] == "unknown"
    assert result["credit_scenarios"] == []


@pytest.mark.parametrize("accounts", [[], [{"stable_id": "unknown"}], [account("a", float("nan"), instant(9))]])
def test_unavailable_has_no_fabricated_zero_week(accounts):
    result = forecast(accounts)
    assert result["status"] == "unavailable"
    assert result["today"] is None and result["days"] == [] and result["planning_days"] == []
    assert result["risk"]["aggregate_exhaustion"] is None
    assert result["reason"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reset", [instant(7), instant(8, 12), instant(16)])
def test_unconfirmed_expired_or_nonweekly_anchor_is_not_rolled_forward(reset):
    result = forecast([account("a", 80, reset)])
    assert result["status"] == "unavailable"
    assert result["accounts"] == []
    assert result["unknown_accounts"][0]["reasons"]


def test_duplicate_credentials_do_not_double_pool_capacity_and_conflicts_are_unknown():
    single = account("pool", 61, instant(11))
    result = forecast([single, dict(single)])
    assert result["days"] == forecast([single])["days"]
    assert len(result["accounts"]) == 1
    conflict = forecast([single, dict(single, remaining_percent=60)])
    assert conflict["status"] == "unavailable"
    assert "conflicting_duplicate_pool" in conflict["unknown_accounts"][0]["reasons"]


def test_current_balances_replan_instead_of_freezing_historical_target():
    high = forecast([account("a", 80, instant(14, 12))])
    low = forecast([account("a", 20, instant(14, 12))])
    assert low["today"]["target"] < high["today"]["target"]
    assert low["days"][1]["target"] < high["days"][1]["target"]


def test_past_history_never_subtracts_debt_or_recreates_missed_expiry():
    accounts = [account("a", 60, instant(14), history_provenance="trusted_ingestion")]
    no_history = forecast(accounts)
    history = [observation(instant(8), 100), observation(instant(8, 9), 20, sequence=2), observation(instant(8, 10), 90, sequence=3)]
    with_history = forecast(accounts, observations=history)
    assert with_history["actual"]["status"] == "unavailable"
    assert with_history["actual"]["reason"] == "reset_history_incomplete"
    for key in ("today", "days", "planning_days", "post_reset", "risk"):
        assert with_history[key] == no_history[key]


@pytest.mark.parametrize(("start_balance", "end_balance", "resets", "lower", "upper"), [(100, 80.625, [], 19.375, 19.375), (80, 70, [instant(8, 9).timestamp()], 30, 110)])
def test_historical_usage_requires_explicit_complete_reset_and_precision_assumptions(start_balance, end_balance, resets, lower, upper):
    now = instant(8, 12)
    inputs = [account("a", end_balance, instant(14), source_timestamp=now.timestamp(), history_provenance="trusted_ingestion", event_history_complete=True, confirmed_resets=resets, measurement_model="exact")]
    result = forecast(inputs, now=now, observations=[observation(instant(8), start_balance)])
    actual = result["actual"]
    assert actual["status"] == "estimated"
    assert (actual["lower_bound"], actual["upper_bound"]) == (lower, upper)
    assert actual["estimate"] == (lower if lower == upper else None)
    assert actual["assumptions"]


def test_trusted_samples_without_precision_do_not_claim_exact_usage():
    now = instant(8, 12)
    inputs = [account("a", 80, instant(14), source_timestamp=now.timestamp(), history_provenance="trusted_ingestion", event_history_complete=True, confirmed_resets=[])]
    result = forecast(inputs, observations=[observation(instant(8), 100)])
    assert result["actual"]["status"] == "unavailable"
    assert result["actual"]["reason"] == "measurement_precision_unestablished"


def test_usage_crossing_day_boundary_is_a_range_not_all_charged_to_today():
    now = instant(8, 12)
    inputs = [account("a", 60, instant(14), source_timestamp=now.timestamp(), history_provenance="trusted_ingestion", event_history_complete=True, confirmed_resets=[], measurement_model="exact")]
    result = forecast(inputs, observations=[observation(instant(8, 5), 100)])
    actual = result["actual"]
    assert actual["status"] == "estimated"
    assert (actual["lower_bound"], actual["upper_bound"]) == (0, 40)
    assert actual["estimate"] is None


def test_overlapping_same_time_rounded_samples_remain_feasible():
    now = instant(8, 12)
    inputs = [account("a", 60, instant(14), source_timestamp=now.timestamp(), history_provenance="trusted_ingestion", event_history_complete=True, confirmed_resets=[], measurement_error_pp=1)]
    rows = [observation(instant(8), 90), observation(instant(8, 9), 80, sequence=2), observation(instant(8, 9), 81, sequence=3)]
    actual = forecast(inputs, observations=rows)["actual"]
    assert actual["status"] == "estimated"
    assert (actual["lower_bound"], actual["upper_bound"]) == (28, 32)
    assert actual["estimate"] is None


def test_reset_at_day_boundary_requires_explicit_pre_post_observation_order():
    now = instant(8, 12)
    inputs = [account("a", 90, instant(14), source_timestamp=now.timestamp(), history_provenance="trusted_ingestion", event_history_complete=True, confirmed_resets=[instant(8).timestamp()], measurement_model="exact")]
    ambiguous = observation(instant(8), 100)
    assert forecast(inputs, observations=[ambiguous])["actual"]["status"] == "unavailable"
    actual = forecast(inputs, observations=[dict(ambiguous, side="post")])["actual"]
    assert actual["status"] == "estimated"
    assert (actual["lower_bound"], actual["upper_bound"]) == (10, 10)


def credit_account(now, **extra):
    return account("a", 10, now + timedelta(hours=100), source_timestamp=now.timestamp(), reset_credits_status="success", reset_credits_stale=False, reset_credits_complete=True, reset_credits=[{"id": "c", "expires_at": now.timestamp() + 8 * 3600, "status": "available", "reset_type": "codex_rate_limits"}], **extra)


def test_credit_expiry_is_an_option_not_refill_or_replacement_anchor():
    now = instant(8, 12)
    inputs = credit_account(now)
    result = forecast([inputs], now=now)
    without = forecast([dict(inputs, reset_credits=[])], now=now)
    assert result["days"] == without["days"]
    assert result["accounts"][0]["natural_reset_at"] == inputs["reset_at"]
    assert result["accounts"][0]["credit_options"][0]["expires_at"] == inputs["reset_credits"][0]["expires_at"]
    assert len(result["credit_scenarios"]) == 10
    assert all(item["conditional"] and item["actual_continuous_availability_guarantee"] is False for item in result["credit_scenarios"])


@pytest.mark.parametrize(("field", "value"), [("source_timestamp", None), ("source_timestamp", instant(8, 11).timestamp()), ("reset_credits_status", "error"), ("reset_credits_stale", True), ("reset_credits_complete", False)])
def test_credit_scenarios_require_fresh_complete_successful_inputs(field, value):
    inputs = dict(credit_account(instant(8, 12)), **{field: value})
    result = forecast([inputs])
    assert result["status"] == "ready"
    assert result["credit_scenarios"] == [] and result["credit_scenarios_reason"]


@pytest.mark.parametrize("status", [None, "", False])
def test_missing_or_invalid_credit_status_never_becomes_available(status):
    inputs = credit_account(instant(8, 12))
    if status is None:
        inputs["reset_credits"][0].pop("status")
    else:
        inputs["reset_credits"][0]["status"] = status
    result = forecast([inputs])
    assert result["accounts"][0]["credit_options"][0]["status"] is None
    assert result["credit_scenarios"] == []
    assert result["credit_scenarios_reason"] == "unknown_credit_status"


@pytest.mark.parametrize(("age", "stale"), [(60, False), (900, False), (900.001, True), (None, True), (-1, True)])
def test_staleness_respects_source_age_and_exact_threshold(age, stale):
    now = instant(8, 12)
    result = build_codex_quota_forecast(accounts=[account("a", 60, instant(9, 18))], now=now, source_timestamp=None if age is None else now.timestamp() - age, stale_after_seconds=900)
    assert result["stale"] is stale
    if age is None:
        assert result["age_seconds"] is None
    else:
        assert result["age_seconds"] == pytest.approx(max(0, age), rel=0, abs=1e-6)


def test_result_is_json_safe_deterministic_and_does_not_mutate_inputs():
    inputs = [credit_account(instant(8, 12))]
    original = deepcopy(inputs)
    first = forecast(inputs)
    assert forecast(inputs) == first
    assert inputs == original
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    obsolete = {"remaining_target", "expected_used_by_now", "live_sustainable_remaining"}
    assert all(not obsolete.intersection(day) for day in first["days"])


def test_naive_now_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_codex_quota_forecast(accounts=[], now=datetime(2026, 8, 8, 12))


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), True])
def test_invalid_freshness_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="finite non-negative"):
        build_codex_quota_forecast(accounts=[], now=instant(8), stale_after_seconds=threshold)
