"""Behavioral tests for change-only Codex weekly quota persistence."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import sqlite3
from threading import Barrier

import pytest

from rotator_library.usage.quota_observation_store import QuotaObservationStore


def record(store: QuotaObservationStore, account: str = "account-a", **changes: object) -> bool:
    values: dict[str, object] = {
        "stable_account_id": account,
        "email": f"{account}@example.test",
        "used_percent": "25.6250",
        "remaining_percent": "74.3750",
        "reset_at": 1_700_100_000.25,
        "reset_credit_metadata": {
            "available_count": 1,
            "credits": [{"expires_at": 1_700_200_000.5, "status": "available"}],
        },
        "source_timestamp": 1_700_000_000.0,
        "observed_at": 1_700_000_001.0,
    }
    values.update(changes)
    return store.record(**values)  # type: ignore[arg-type]


def test_first_observation_inserts_and_duplicate_dedupes(tmp_path):
    store = QuotaObservationStore(tmp_path / "usage" / "quota.sqlite3")

    assert record(store) is True
    assert record(
        store,
        source_timestamp=1_700_000_010.0,
        observed_at=1_700_000_011.0,
        email="renamed@example.test",
    ) is False

    history = store.history("account-a")
    assert len(history) == 1
    assert history[0].email == "account-a@example.test"


def test_each_meaningful_quota_reset_or_credit_change_inserts(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store) is True

    assert record(store, used_percent="25.7250", observed_at=2) is True
    assert record(store, remaining_percent="74.2750", observed_at=3) is True
    assert record(store, reset_at=1_700_100_001.25, observed_at=4) is True
    assert record(
        store,
        reset_credit_metadata={
            "available_count": 2,
            "credits": [{"expires_at": 1_700_200_000.5, "status": "available"}],
        },
        observed_at=5,
    ) is True
    assert record(
        store,
        reset_credit_metadata={
            "credits": [{"status": "redeemed", "expires_at": 1_700_200_000.5}],
            "available_count": 2,
        },
        observed_at=6,
    ) is True

    assert len(store.history("account-a")) == 6


def test_credit_metadata_key_order_does_not_create_change(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    metadata = {"available_count": 1, "credits": [{"status": "available", "expires_at": 12}]}
    assert record(store, reset_credit_metadata=metadata) is True
    assert record(
        store,
        reset_credit_metadata={"credits": [{"expires_at": 12, "status": "available"}], "available_count": 1},
        observed_at=2,
    ) is False


def test_reverted_quota_state_is_a_new_transition(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, used_percent="25", remaining_percent="75", observed_at=1) is True
    assert record(store, used_percent="30", remaining_percent="70", observed_at=2) is True
    assert record(store, used_percent="25", remaining_percent="75", observed_at=3) is True

    assert [item.observed_at for item in store.history("account-a")] == [1.0, 2.0, 3.0]


def test_concurrent_identical_state_inserts_once_and_preserves_returned_states(tmp_path):
    path = tmp_path / "quota.sqlite3"
    stores = (QuotaObservationStore(path), QuotaObservationStore(path))
    barrier = Barrier(2)

    def concurrent_record(store: QuotaObservationStore, **changes: object) -> bool:
        barrier.wait()
        return record(store, **changes)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_record, store) for store in stores]
        results = [future.result(timeout=5) for future in futures]

    assert sorted(results) == [False, True]
    history = stores[0].history("account-a")
    assert len(history) == 1
    assert history[0].used_percent_text == "25.6250"
    assert history[0].remaining_percent_text == "74.3750"

    assert record(
        stores[0],
        used_percent="30",
        remaining_percent="70",
        observed_at=1_700_000_002.0,
    ) is True

    barrier.reset()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                concurrent_record,
                store,
                used_percent="30",
                remaining_percent="70",
                observed_at=1_700_000_003.0,
            )
            for store in stores
        ]
        results = [future.result(timeout=5) for future in futures]

    assert results == [False, False]
    assert record(
        stores[0],
        used_percent="25.6250",
        remaining_percent="74.3750",
        observed_at=1_700_000_004.0,
    ) is True
    assert [item.remaining_percent_text for item in stores[0].history("account-a")] == [
        "74.3750",
        "70",
        "74.3750",
    ]


def test_exact_decimal_text_and_numeric_values_round_trip(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(
        store,
        used_percent=Decimal("25.6250"),
        remaining_percent="74.3750",
    ) is True

    observation = store.latest("account-a")
    assert observation is not None
    assert observation.used_percent_text == "25.6250"
    assert observation.remaining_percent_text == "74.3750"
    assert observation.used_percent == 25.625
    assert observation.remaining_percent == 74.375


def test_identical_states_are_isolated_by_stable_account(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, "account-a") is True
    assert record(store, "account-b") is True

    assert [item.stable_account_id for item in store.history("account-a")] == ["account-a"]
    assert [item.stable_account_id for item in store.history("account-b")] == ["account-b"]


def test_reopening_preserves_chronological_bounded_history(tmp_path):
    path = tmp_path / "quota.sqlite3"
    store = QuotaObservationStore(path)
    assert record(store, observed_at=10) is True
    assert record(store, used_percent="30.0", remaining_percent="70.0", observed_at=20) is True
    assert record(store, used_percent="40.0", remaining_percent="60.0", observed_at=30) is True

    reopened = QuotaObservationStore(path)
    assert [item.observed_at for item in reopened.history("account-a", start_at=20, end_at=30)] == [20.0, 30.0]
    latest = reopened.latest("account-a")
    assert latest is not None
    assert latest.observed_at == 30.0


def test_connections_close_after_store_operations(tmp_path, monkeypatch):
    path = tmp_path / "quota.sqlite3"
    store = QuotaObservationStore(path)
    connections = []
    original_connect = store._connect

    def tracked_connect():
        connection = original_connect()
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", tracked_connect)

    assert record(store) is True
    assert record(store, observed_at=1_700_000_002.0) is False
    assert store.latest("account-a") is not None
    assert len(store.history("account-a")) == 1
    assert store.latest_at_or_before("account-a", 1_700_000_001.0) is not None
    assert store.observations_around_boundary(
        "account-a", 1_700_000_001.0
    ).baseline is not None

    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")

    reopened = QuotaObservationStore(path)
    assert len(reopened.history("account-a")) == 1


def test_boundary_read_returns_baseline_and_later_changes(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, observed_at=359) is True
    assert record(store, used_percent="30", remaining_percent="70", observed_at=360) is True
    assert record(store, used_percent="35", remaining_percent="65", observed_at=361) is True
    assert record(store, used_percent="40", remaining_percent="60", observed_at=500) is True

    around = store.observations_around_boundary("account-a", 360, end_at=400)
    assert around.baseline is not None
    assert around.baseline.observed_at == 360.0
    assert [item.observed_at for item in around.changes] == [361.0]


def test_boundary_read_never_invents_missing_baseline(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, observed_at=361) is True

    around = store.observations_around_boundary("account-a", 360)
    assert around.baseline is None
    assert [item.observed_at for item in around.changes] == [361.0]
