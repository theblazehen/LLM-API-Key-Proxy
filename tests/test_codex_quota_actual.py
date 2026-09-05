"""Conditional counter bounds, with no optimizer or live-provider assumptions.

Analytical cases mirror the independent offline usage-bounds oracle: one
overwrite, reporting cuts, shared rounded states, and pre/post ownership.
"""

import pytest

from rotator_library.usage.codex_quota_actual import derive_usage_observation


def account(**changes):
    return {
        "stable_id": "pool-a",
        "history_provenance": "trusted_ingestion",
        "event_history_complete": True,
        "confirmed_resets": [],
        "measurement_model": "exact",
        **changes,
    }


def observations(*samples, pool="pool-a"):
    rows = []
    for sequence, sample in enumerate(samples, 1):
        at, balance, *side = sample
        rows.append({
            "stable_account_id": pool,
            "pool_id": pool,
            "source": "api",
            "id": sequence,
            "observed_at": at,
            "source_timestamp": at,
            "confirmed_at": at,
            "remaining_percent": balance,
            **({"side": side[0]} if side else {}),
        })
    return rows


def assert_bounds(result, lower, upper):
    # A singleton is conditional on explicit premises, never claimed observed.
    assert result["status"] == "estimated"
    assert result["lower_bound"] == pytest.approx(lower)
    assert result["upper_bound"] == pytest.approx(upper)
    assert result["estimate"] == (lower if lower == upper else None)
    assert result["assumptions"]


def assert_unavailable(result, reason):
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert result["estimate"] is None
    assert result["lower_bound"] is None
    assert result["upper_bound"] is None


def test_exact_reset_free_decrease_is_conditional_singleton():
    result = derive_usage_observation([account()], observations((0, 100), (3, 80)), 0, 3)
    assert_bounds(result, 20, 20)


@pytest.mark.parametrize("resets, upper", [([1], 110), ([1, 1, 1], 110), ([1, 2], 210)])
def test_confirmed_overwrites_bound_usage_without_counting_discard(resets, upper):
    result = derive_usage_observation(
        [account(confirmed_resets=resets)], observations((0, 80), (3, 70)), 0, 3,
    )
    assert_bounds(result, 30, upper)


def test_cross_boundary_decrease_is_not_all_assigned_to_today():
    result = derive_usage_observation([account()], observations((5, 100), (7, 60)), 6, 7)
    assert_bounds(result, 0, 40)
    assert result["coverage"]["accounts"][0]["baseline_at"] == 5
    assert result["observed_decrease"] == 40
    assert result["observed_decrease_basis"] == "positive_sampled_counter_movements_not_daily_usage"


def test_rounded_chain_telescopes_shared_states_not_independent_edge_errors():
    result = derive_usage_observation(
        [account(measurement_model=None, measurement_error_pp=0.5)],
        observations((0, 80), (1, 70), (3, 60)), 0, 3,
    )
    # Independent pair widening would incorrectly produce [18, 22].
    assert_bounds(result, 19, 21)


def test_same_time_rounded_measurements_intersect_before_solving():
    result = derive_usage_observation(
        [account(measurement_model=None, measurement_error_pp=1,
                 source_timestamp=3, remaining_percent=60)],
        observations((0, 90), (1, 80), (1, 81), (3, 60)), 0, 3,
    )
    assert_bounds(result, 28, 32)


def test_current_snapshot_intersects_same_time_measurement():
    result = derive_usage_observation(
        [account(measurement_model=None, measurement_error_pp=1,
                 source_timestamp=3, remaining_percent=61)],
        observations((0, 90), (3, 60)), 0, 3,
    )
    assert_bounds(result, 28, 31)


@pytest.mark.parametrize("samples, resets, expected", [
    ([(0, 80), (1, 80, "pre"), (1, 100, "post"), (3, 70)], [1], 30),
    ([(0, 80), (3, 70, "pre"), (3, 100, "post")], [3], 10),
    ([(0, 80, "pre"), (0, 100, "post"), (3, 70)], [0], 30),
])
def test_explicit_reset_sides_own_boundary_consumption(samples, resets, expected):
    result = derive_usage_observation(
        [account(confirmed_resets=resets)], observations(*samples), 0, 3,
    )
    assert_bounds(result, expected, expected)


def test_unobserved_tail_does_not_invent_constant_usage():
    result = derive_usage_observation([account()], observations((0, 100), (1, 80)), 0, 3)
    assert_bounds(result, 20, 100)


def test_confirmation_adds_endpoint_without_rewriting_day_boundary():
    rows = observations((5, 80))
    rows[0]["confirmed_at"] = 7
    result = derive_usage_observation([account(confirmed_resets=[6.5])], rows, 6, 7)
    # The reset permits pre-boundary consumption; unchanged endpoints alone
    # do not prove a flat balance through the reporting cut.
    assert_bounds(result, 20, 100)


@pytest.mark.parametrize("changes, reason", [
    ({"event_history_complete": False}, "reset_history_incomplete"),
    ({"event_history_complete": None}, "reset_history_incomplete"),
    ({"measurement_model": None}, "measurement_precision_unestablished"),
    ({"history_provenance": "legacy"}, "untrusted_account_provenance"),
])
def test_unestablished_premises_never_manufacture_exact_usage(changes, reason):
    result = derive_usage_observation([account(**changes)], observations((0, 100), (3, 80)), 0, 3)
    assert_unavailable(result, reason)
    if reason == "reset_history_incomplete":
        assert result["observed_decrease"] == 20


@pytest.mark.parametrize("changes, reason", [
    ({"source": "legacy"}, "legacy_or_untrusted_history"),
    ({"pool_id": "other-pool"}, "mixed_pool_history"),
    ({"source_timestamp": -1}, "incoherent_observation_timestamps"),
    ({"confirmed_at": -1}, "incoherent_observation_timestamps"),
    ({"remaining_percent": float("nan")}, "invalid_counter_balance"),
])
def test_untrusted_or_incoherent_rows_are_not_usage_evidence(changes, reason):
    rows = observations((0, 100), (3, 80))
    rows[0].update(changes)
    assert_unavailable(derive_usage_observation([account()], rows, 0, 3), reason)


def test_missing_baseline_cannot_be_backfilled_from_later_measurement():
    result = derive_usage_observation([account()], observations((1, 100), (3, 80)), 0, 3)
    assert_unavailable(result, "baseline_unavailable")


@pytest.mark.parametrize("identity", [None, "", 123])
def test_unknown_pool_identity_cannot_be_aggregated(identity):
    result = derive_usage_observation([account(stable_id=identity)], [], 0, 3)
    assert_unavailable(result, "unknown_pool_identity")


def test_duplicate_pool_is_not_double_counted():
    result = derive_usage_observation([account(), account()], observations((0, 100), (3, 80)), 0, 3)
    assert_unavailable(result, "duplicate_pool_identity")


def test_aggregate_requires_evidence_for_every_pool():
    rows = observations((0, 100), (3, 80)) + observations((0, 90), (3, 60), pool="pool-b")
    pools = [account(), account(stable_id="pool-b")]
    assert_bounds(derive_usage_observation(pools, rows, 0, 3), 50, 50)
    pools[1]["event_history_complete"] = False
    assert_unavailable(derive_usage_observation(pools, rows, 0, 3), "reset_history_incomplete")


@pytest.mark.parametrize("samples, changes, reason", [
    ([(0, 80), (3, 90)], {}, "inconsistent_counter_history"),
    ([(0, 100), (1, 90), (1, 80)], {}, "conflicting_same_time_observations"),
    ([(0, 100), (1, 80)], {"confirmed_resets": [1]}, "ambiguous_reset_observation_order"),
])
def test_counter_conflicts_never_infer_an_unreported_reset(samples, changes, reason):
    result = derive_usage_observation([account(**changes)], observations(*samples), 0, 3)
    assert_unavailable(result, reason)


def test_insertion_sequence_cannot_be_silently_sorted_or_reversed():
    rows = observations((0, 100), (1, 90), (1, 90), (3, 80))
    rows[1], rows[2] = rows[2], rows[1]
    assert_unavailable(derive_usage_observation([account()], rows, 0, 3), "out_of_order_observations")


def test_zero_duration_report_has_no_consumption():
    assert_bounds(derive_usage_observation([account()], observations((0, 80)), 0, 0), 0, 0)
