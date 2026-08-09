"""Observable contract tests for the pure Codex quota forecast."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "src/rotator_library/usage/codex_quota_forecast.py"
_SPEC = importlib.util.spec_from_file_location("codex_quota_forecast", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_codex_quota_forecast = _MODULE.build_codex_quota_forecast


UTC = timezone.utc


def instant(day: int, hour: int = 6, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=UTC)


def account(
    account_id: str,
    remaining: float,
    reset_at: datetime,
    **extra: object,
) -> dict[str, object]:
    return {
        "stable_id": account_id,
        "email": f"{account_id}@example.test",
        "remaining_percent": remaining,
        "reset_at": reset_at.timestamp(),
        **extra,
    }


def forecast(
    accounts: list[dict[str, object]],
    *,
    now: datetime = instant(8, 12),
    observations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_codex_quota_forecast(
        accounts=accounts,
        observations=observations or [],
        now=now,
        source_timestamp=now.timestamp(),
    )


def contribution_sum(day: dict[str, object]) -> float:
    return sum(segment["amount"] for segment in day["contributions"])


def test_local_six_am_boundary_starts_and_labels_quota_day() -> None:
    accounts = [account("a", 50.0, instant(9, 18))]
    before = forecast(accounts, now=instant(8, 5, 59))
    at = forecast(accounts, now=instant(8, 6, 0))
    local = timezone(timedelta(hours=2))
    local_before = build_codex_quota_forecast(
        accounts=[account("a", 50.0, datetime(2026, 8, 9, 18, tzinfo=local))],
        now=datetime(2026, 8, 8, 5, 59, tzinfo=local),
    )

    assert before["horizon"]["start_at"] == instant(7, 6).timestamp()
    assert at["horizon"]["start_at"] == instant(8, 6).timestamp()
    assert at["planning_horizon"]["end_at"] == instant(22, 6).timestamp()
    assert at["planning_horizon"]["day_count"] == 14
    assert at["planning_horizon"]["rolling_window_seconds"] == 7 * 24 * 60 * 60
    assert len(at["days"]) == 7
    assert len(at["planning_days"]) == 14
    assert at["planning_days"][:7] == at["days"]
    assert at["post_reset"]["day_start_at"] is not None
    selected = next(
        day
        for day in at["planning_days"]
        if day["start_at"] == at["post_reset"]["day_start_at"]
    )
    assert at["post_reset"]["daily_sustainable_pace"] == selected["sustainable_candidate"]
    assert [day["local_date"] for day in at["days"]] == [
        "2026-08-08",
        "2026-08-09",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
        "2026-08-14",
    ]
    assert at["quota_day_start_hour"] == 6
    assert local_before["horizon"]["start_at"] == datetime(
        2026, 8, 7, 6, tzinfo=local
    ).timestamp()


def test_first_run_reports_missing_baseline_and_never_invents_actual_usage() -> None:
    result = forecast([account("a", 74.375, instant(9, 18))])

    assert result["status"] == "ready"
    assert result["actual"] == {
        "status": "baseline_unavailable",
        "used_since_day_start": None,
    }
    assert result["risk"] == {
        "aggregate_exhaustion": False,
        "basis": "forecast_targets",
    }
    assert result["accounts"][0]["remaining_percent"] == pytest.approx(74.375)


def test_observed_actual_preserves_fractional_precision() -> None:
    now = instant(8, 12)
    result = forecast(
        [account("a", 74.375, instant(9, 18))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 80.5},
            {"stable_id": "a", "observed_at": instant(8, 10).timestamp(), "remaining_percent": 76.25},
        ],
    )

    assert result["status"] == "ready"
    assert result["actual"]["used_since_day_start"] == pytest.approx(6.125)
    assert result["today"]["target"] % 1 != 0


def test_actual_since_day_start_sums_each_account_transition() -> None:
    now = instant(8, 12)
    result = forecast(
        [
            account("a", 48.0, instant(9, 18)),
            account("b", 41.0, instant(10, 18)),
        ],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 80.0},
            {"stable_id": "a", "observed_at": instant(8, 9).timestamp(), "remaining_percent": 60.0},
            {"stable_id": "b", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 70.0},
            {"stable_id": "b", "observed_at": instant(8, 10).timestamp(), "remaining_percent": 55.0},
        ],
    )

    # A consumed 20 + 12 = 32 pp; B consumed 15 + 14 = 29 pp.
    # This fails if one account, or only one transition per account, is used.
    assert result["status"] == "ready"
    assert result["actual"] == {"status": "observed", "used_since_day_start": 61.0}


def test_actual_since_day_start_ignores_reset_balance_restoration() -> None:
    now = instant(8, 20)
    reset = instant(8, 18)
    result = forecast(
        [account("a", 71.0, reset)],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 80.0},
            {"stable_id": "a", "observed_at": instant(8, 12).timestamp(), "remaining_percent": 50.0},
            # The natural reset restores the balance; the post-reset decrease
            # must add 29 pp rather than canceling the earlier 30 pp.
            {"stable_id": "a", "observed_at": instant(8, 19).timestamp(), "remaining_percent": 71.0},
        ],
    )

    assert result["status"] == "ready"
    assert result["actual"] == {"status": "observed", "used_since_day_start": 59.0}


def test_observed_actual_reconciles_unobserved_reset_overwrite() -> None:
    now = instant(8, 20)
    result = forecast(
        [account("a", 70.0, instant(8, 18))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 80.0},
        ],
    )

    # 80 remains at 06:00. The 18:00 reset overwrites the unknown pre-reset
    # remainder with 100, and the live 70 proves 30 pp of post-reset usage.
    assert result["actual"] == {"status": "observed", "used_since_day_start": 30.0}


def test_actual_reconstructs_earlier_today_reset_from_next_week_timestamp() -> None:
    now = instant(8, 20)
    earlier_today = instant(8, 18)
    result = forecast(
        [account("a", 70.0, earlier_today + timedelta(days=7))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 80.0},
        ],
    )

    assert result["actual"] == {"status": "observed", "used_since_day_start": 30.0}


def test_reset_at_quota_day_start_owns_new_day_actual_and_candidates() -> None:
    now = instant(8, 12)
    reset_at = instant(8, 6)
    pre_reset = {
        "stable_id": "a",
        "observed_at": instant(8, 5, 59).timestamp(),
        "remaining_percent": 80.0,
    }
    result = forecast(
        [account("a", 90.0, reset_at + timedelta(days=7))],
        now=now,
        observations=[pre_reset],
    )
    post_reset_baseline = forecast(
        [account("a", 90.0, reset_at + timedelta(days=7))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": reset_at.timestamp(), "remaining_percent": 100.0},
        ],
    )

    assert result["actual"] == {"status": "observed", "used_since_day_start": 10.0}
    assert result["today"]["target"] == pytest.approx(post_reset_baseline["today"]["target"])
    assert result["today"]["drain_candidate"] == pytest.approx(
        post_reset_baseline["today"]["drain_candidate"]
    )
    assert result["today"]["sustainable_candidate"] == pytest.approx(
        post_reset_baseline["today"]["sustainable_candidate"]
    )


def test_credit_at_quota_day_start_restores_pre_boundary_baseline_once() -> None:
    now = instant(8, 12)
    reset_at = instant(8, 6)
    accounts = [
        account(
            "a",
            90.0,
            instant(12, 6),
            reset_credits=[{"id": "boundary", "auto_redeem_at": reset_at.timestamp()}],
        )
    ]
    pre_reset = forecast(
        accounts,
        now=now,
        observations=[
            {
                "stable_id": "a",
                "observed_at": instant(8, 5, 59).timestamp(),
                "remaining_percent": 80.0,
            },
        ],
    )
    post_reset = forecast(
        accounts,
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": reset_at.timestamp(), "remaining_percent": 100.0},
        ],
    )

    assert pre_reset["actual"] == {"status": "observed", "used_since_day_start": 10.0}
    assert post_reset["actual"] == pre_reset["actual"]
    assert pre_reset["today"]["target"] == pytest.approx(post_reset["today"]["target"])
    assert pre_reset["today"]["drain_candidate"] == pytest.approx(
        post_reset["today"]["drain_candidate"]
    )
    assert pre_reset["today"]["sustainable_candidate"] == pytest.approx(
        post_reset["today"]["sustainable_candidate"]
    )


def test_natural_reset_overwrites_remainder_not_adds_to_it() -> None:
    now = instant(8, 12)
    reset = instant(8, 18)
    result = forecast([account("a", 40.0, reset)], now=now)
    today = result["today"]
    natural_events = [event for event in today["reset_events"] if event["kind"] == "natural"]

    assert natural_events
    assert natural_events[0]["balance_after"] == 100.0
    assert natural_events[0]["capacity_added"] <= 100.0
    assert natural_events[0]["capacity_added"] != 140.0


def test_timed_credit_overwrites_balance_and_expiry_is_fallback() -> None:
    now = instant(8, 12)
    credit_at = instant(8, 15)
    result = forecast(
        [
            account(
                "a",
                60.0,
                instant(15, 12),
                reset_credits=[
                    {"id": "auto", "auto_redeem_at": credit_at.timestamp(), "expires_at": instant(8, 16).timestamp()},
                    {"id": "expiry", "expires_at": instant(10, 12).timestamp()},
                ],
            )
        ],
        now=now,
    )

    events = [event for day in result["days"] for event in day["reset_events"]]
    credit_events = [event for event in events if event["kind"].startswith("credit_")]
    assert [(event["kind"], event["at"]) for event in credit_events] == [
        ("credit_auto_redeem", credit_at.timestamp()),
        ("credit_expiry", instant(10, 12).timestamp()),
    ]
    assert all(event["balance_after"] == 100.0 for event in credit_events)


def test_missing_weekly_fields_are_excluded_and_reported() -> None:
    result = forecast(
        [
            account("known", 55.0, instant(9, 12)),
            {"stable_id": "missing-balance", "reset_at": instant(9, 12).timestamp()},
            {"stable_id": "missing-reset", "remaining_percent": 44.0},
        ]
    )

    assert [item["account_id"] for item in result["accounts"]] == ["known"]
    assert [item["account_id"] for item in result["unknown_accounts"]] == [
        "missing-balance",
        "missing-reset",
    ]
    assert result["unknown_accounts"][0]["reasons"] == ["missing_or_invalid_remaining_percent"]
    assert result["unknown_accounts"][1]["reasons"] == ["missing_or_invalid_reset_at"]


def test_unavailable_accounts_do_not_claim_aggregate_exhaustion() -> None:
    result = forecast([{"stable_id": "unknown"}])

    assert result["status"] == "unavailable"
    assert result["risk"]["aggregate_exhaustion"] is False


def test_drain_candidate_wins_when_next_reset_requires_faster_use() -> None:
    now = instant(8, 12)
    result = forecast([account("a", 80.0, instant(8, 16))], now=now)
    today = result["today"]

    assert today["drain_candidate"] > today["sustainable_candidate"]
    assert today["selected_reason"] == "next_reset_drain"
    assert today["target"] == pytest.approx(today["drain_candidate"])


def test_slow_current_day_leaves_more_future_capacity_than_overuse() -> None:
    now = instant(8, 18)
    accounts = [
        account("a", 50.0, instant(9, 6)),
        account("b", 100.0, instant(13, 6)),
    ]
    slow = forecast(
        accounts,
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 50.0},
            {"stable_id": "b", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 100.0},
        ],
    )
    overused = forecast(
        [account("a", 5.0, instant(9, 6)), account("b", 100.0, instant(13, 6))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 50.0},
            {"stable_id": "b", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 100.0},
        ],
    )

    assert slow["days"][1]["target"] > overused["days"][1]["target"]
    assert overused["actual"]["used_since_day_start"] == pytest.approx(45.0)


def test_day_zero_target_reconciles_actual_with_live_remaining_capacity() -> None:
    now = instant(8, 18)
    result = forecast(
        [account("a", 5.0, instant(13, 6))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 50.0},
        ],
    )

    assert result["actual"]["used_since_day_start"] == pytest.approx(45.0)
    assert result["today"]["remaining_target"] >= 0.0
    assert result["today"]["target"] == pytest.approx(
        result["actual"]["used_since_day_start"]
        + result["today"]["remaining_target"]
    )


def test_same_day_reset_does_not_publish_unreachable_morning_target() -> None:
    now = instant(8, 18)
    result = forecast(
        [account("a", 55.0, instant(8, 10) + timedelta(days=7))],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 30.0},
        ],
    )

    today = result["today"]
    assert result["actual"]["used_since_day_start"] == pytest.approx(45.0)
    assert today["target"] == pytest.approx(
        result["actual"]["used_since_day_start"] + today["remaining_target"]
    )
    assert today["planned_at_day_start"] == pytest.approx(130.0)
    assert today["target"] < today["planned_at_day_start"]


def test_staggered_same_day_resets_keep_today_target_live_reachable() -> None:
    now = instant(8, 18)
    result = forecast(
        [
            account("a", 55.0, instant(8, 10) + timedelta(days=7)),
            account("b", 70.0, instant(8, 14) + timedelta(days=7)),
            account("c", 40.0, instant(12, 6)),
        ],
        now=now,
        observations=[
            {"stable_id": "a", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 30.0},
            {"stable_id": "b", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 25.0},
            {"stable_id": "c", "observed_at": instant(8, 6).timestamp(), "remaining_percent": 50.0},
        ],
    )

    today = result["today"]
    assert today["target"] == pytest.approx(
        result["actual"]["used_since_day_start"] + today["remaining_target"]
    )
    assert today["target"] <= result["actual"]["used_since_day_start"] + 165.0


def test_fully_dead_known_pool_reports_aggregate_exhaustion() -> None:
    result = forecast([account("a", 0.0, instant(9, 18))])

    assert result["risk"]["aggregate_exhaustion"] is True


def test_materially_dry_pool_before_replenishment_reports_exhaustion() -> None:
    now = instant(8, 12)
    result = forecast([account("a", 0.0, instant(8, 18))], now=now)

    assert result["risk"]["aggregate_exhaustion"] is True


@pytest.mark.parametrize(
    ("remaining", "reset"),
    [
        (40.0, instant(9, 12)),  # 24 hours
        (30.0, instant(14, 12)),  # 144 hours
    ],
)
def test_healthy_reviewer_examples_are_not_aggregate_exhaustion(
    remaining: float, reset: datetime
) -> None:
    result = forecast([account("a", remaining, reset)], now=instant(8, 12))

    assert result["risk"]["aggregate_exhaustion"] is False


def test_low_but_alive_pool_is_not_red_from_solver_residue() -> None:
    now = instant(8, 12)
    result = forecast([account("a", 1.0, instant(8, 18))], now=now)

    assert result["risk"]["aggregate_exhaustion"] is False


def test_continuous_weekly_capacity_is_not_aggregate_exhaustion() -> None:
    result = forecast(
        [
            account("a", 33.333333333, instant(10, 11)),
            account("b", 66.666666667, instant(13, 17)),
        ]
    )

    assert result["risk"]["aggregate_exhaustion"] is False


def test_depletion_exactly_at_replenishment_has_no_dry_interval() -> None:
    now = instant(8, 12)
    result = forecast([account("a", 10.0, instant(8, 18))], now=now)

    assert result["today"]["target"] == pytest.approx(30.0)
    assert result["risk"]["aggregate_exhaustion"] is False


def test_solver_rounding_does_not_report_false_exhaustion() -> None:
    result = forecast(
        [
            account("a", 33.333333333, instant(10, 11)),
            account("b", 66.666666667, instant(13, 17)),
        ]
    )

    assert result["risk"]["aggregate_exhaustion"] is False


def test_contributions_sum_to_each_target_and_are_earliest_reset_first() -> None:
    result = forecast(
        [
            account("late", 33.25, instant(12, 6)),
            account("early", 61.75, instant(9, 18)),
        ]
    )

    for day in result["days"]:
        assert contribution_sum(day) == pytest.approx(day["remaining_target"], abs=1e-7)
    current_contributions = result["today"]["contributions"]
    assert current_contributions[0]["account_id"] == "early"


def test_result_is_json_safe_and_deterministic() -> None:
    now = instant(8, 12)
    accounts = [account("a", 74.375, instant(9, 18))]
    first = forecast(accounts, now=now)
    second = forecast(accounts, now=now)

    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_codex_quota_forecast(
            accounts=[],
            now=datetime(2026, 8, 8, 12),
        )


SELECTION_REASONS = {"next_reset_drain", "rolling_168h_sustainable"}


def horizon_events(result: dict[str, object]) -> list[dict[str, object]]:
    return [event for day in result["days"] for event in day["reset_events"]]


def target_total(result: dict[str, object]) -> float:
    return sum(day["target"] for day in result["days"])


def test_every_reset_in_one_quota_day_is_applied_and_reported() -> None:
    now = instant(8, 12)
    accounts = [
        account("a", 5.0, instant(8, 14)),
        account("b", 5.0, instant(8, 20)),
        account("c", 90.0, instant(12, 6)),
    ]
    result = forecast(accounts, now=now)
    control = forecast(
        [
            account("a", 5.0, instant(8, 14)),
            # The same pool with b's restore pushed outside the event horizon.
            account("b", 5.0, instant(25, 20)),
            account("c", 90.0, instant(12, 6)),
        ],
        now=now,
    )

    # The 14:00 target is met from live balances alone, so the 20:00 overwrite
    # lands after the day's allocation is already complete.
    assert [
        (event["account_id"], event["kind"], event["at"])
        for event in result["today"]["reset_events"]
    ] == [
        ("a", "natural", instant(8, 14).timestamp()),
        ("b", "natural", instant(8, 20).timestamp()),
    ]
    # Reporting the event is not enough: the restored balance must stay in the
    # simulated pool, otherwise every later day silently loses 100 pp.
    assert target_total(result) > target_total(control)


def test_timed_credit_replaces_the_natural_reset_inside_its_new_window() -> None:
    now = instant(8, 12)
    credit_at = instant(9, 12)
    natural_at = instant(14, 12)
    result = forecast(
        [
            account(
                "a",
                50.0,
                natural_at,
                reset_credits=[{"id": "c1", "auto_redeem_at": credit_at.timestamp()}],
            )
        ],
        now=now,
    )
    events = horizon_events(result)

    assert [(event["kind"], event["at"]) for event in events] == [
        ("credit_auto_redeem", credit_at.timestamp())
    ]
    # A redemption restarts the weekly window, so the pre-credit natural reset
    # is no longer real capacity and must not restore a second full window.
    assert target_total(result) <= 150.0 + 1e-6
    # The upstream anchor itself is reported unchanged.
    assert result["accounts"][0]["natural_reset_at"] == natural_at.timestamp()


def test_reset_exactly_on_the_boundary_belongs_to_the_day_it_starts() -> None:
    boundary = instant(9, 6)
    result = forecast([account("a", 40.0, boundary)], now=instant(8, 12))

    assert result["days"][1]["start_at"] == boundary.timestamp()
    assert [event["at"] for event in result["days"][0]["reset_events"]] == []
    assert [
        (event["account_id"], event["at"])
        for event in result["days"][1]["reset_events"]
    ] == [("a", boundary.timestamp())]


def test_stale_flag_follows_source_age_against_the_threshold() -> None:
    now = instant(8, 12)
    accounts = [account("a", 60.0, instant(9, 18))]

    def at_age(age: float | None) -> dict[str, object]:
        return build_codex_quota_forecast(
            accounts=accounts,
            now=now,
            source_timestamp=None if age is None else now.timestamp() - age,
            stale_after_seconds=900.0,
        )

    fresh = at_age(60.0)
    expired = at_age(1800.0)
    unknown = at_age(None)

    assert (fresh["age_seconds"], fresh["stale"]) == (60.0, False)
    assert (expired["age_seconds"], expired["stale"]) == (1800.0, True)
    assert (unknown["age_seconds"], unknown["stale"]) == (None, True)
    # A boundary sample must not be reported stale before the threshold passes.
    assert at_age(900.0)["stale"] is False


def test_selected_reason_is_always_a_published_candidate() -> None:
    result = forecast(
        [
            account("a", 5.0, instant(8, 14)),
            account("b", 62.5, instant(11, 9)),
            account("c", 90.0, instant(12, 6)),
        ],
        now=instant(8, 12),
    )

    assert {day["selected_reason"] for day in result["days"]} <= SELECTION_REASONS
    assert result["today"]["selected_reason"] in SELECTION_REASONS
