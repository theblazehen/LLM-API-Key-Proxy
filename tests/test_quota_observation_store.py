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
    if "observed_at" in changes and "source_timestamp" not in changes:
        values["source_timestamp"] = changes["observed_at"]
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
    assert record(store, observed_at=1) is True

    assert record(store, used_percent="25.7250", remaining_percent="74.2750", observed_at=2) is True
    assert record(store, used_percent="25.8250", remaining_percent="74.1750", observed_at=3) is True
    assert record(store, used_percent="25.8250", remaining_percent="74.1750", reset_at=1_700_100_001.25, observed_at=4) is True
    assert record(
        store,
        used_percent="25.8250", remaining_percent="74.1750", reset_at=1_700_100_001.25,
        reset_credit_metadata={
            "available_count": 2,
            "credits": [{"expires_at": 1_700_200_000.5, "status": "available"}],
        },
        observed_at=5,
    ) is True
    assert record(
        store,
        used_percent="25.8250", remaining_percent="74.1750", reset_at=1_700_100_001.25,
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
    assert record(store, reset_credit_metadata=metadata, observed_at=1) is True
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


def test_legacy_migration_preserves_rows_and_is_idempotent(tmp_path):
    path = tmp_path / "quota.sqlite3"
    original = [
        (7, "account-a", "old@example.test", "0.00", "100.00", 0, 100, 1000, "null", 9, 10, "first-fingerprint"),
        (12, "account-a", None, "10.00", "90.00", 10, 90, 1000, "{}", 11, 10, "second-fingerprint"),
    ]
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE codex_weekly_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stable_account_id TEXT NOT NULL, email TEXT,
                used_percent_text TEXT NOT NULL, remaining_percent_text TEXT NOT NULL,
                used_percent REAL NOT NULL, remaining_percent REAL NOT NULL,
                reset_at REAL NOT NULL, reset_credit_json TEXT NOT NULL,
                source_timestamp REAL NOT NULL, observed_at REAL NOT NULL,
                fingerprint TEXT NOT NULL
            )
        """)
        connection.executemany(
            "INSERT INTO codex_weekly_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            original,
        )

    store = QuotaObservationStore(path)
    migrated = store.history("account-a")
    assert [row.id for row in migrated] == [7, 12]
    assert [row.source for row in migrated] == ["legacy", "legacy"]
    assert [row.pool_id for row in migrated] == [None, None]
    assert [row.confirmed_at for row in migrated] == [10, 11]
    assert QuotaObservationStore(path).history("account-a") == migrated
    with sqlite3.connect(path) as connection:
        preserved = connection.execute("""
            SELECT id, stable_account_id, email, used_percent_text,
                   remaining_percent_text, used_percent, remaining_percent,
                   reset_at, reset_credit_json, source_timestamp, observed_at, fingerprint
            FROM codex_weekly_observations ORDER BY id
        """).fetchall()
        assert preserved == original
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert record(store, observed_at=12, source="api", pool_id="account-a")
    assert store.latest("account-a").id > 12
    assert store.history("account-a")[:2] == migrated


def test_same_time_transitions_keep_authoritative_insertion_order(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    for balance in (100, 90, 80):
        assert record(store, used_percent=100 - balance, remaining_percent=balance,
                      observed_at=10, source="headers", pool_id="account-a")
    history = store.history("account-a")
    assert [row.remaining_percent for row in history] == [100, 90, 80]
    assert [row.id for row in history] == sorted({row.id for row in history})
    assert store.latest("account-a") == history[-1]
    assert store.latest_at_or_before("account-a", 10) == history[-1]
    assert store.observations_around_boundary("account-a", 10).baseline == history[-1]


def test_unchanged_confirmation_preserves_original_boundary_evidence(tmp_path):
    path = tmp_path / "quota.sqlite3"
    store = QuotaObservationStore(path)
    assert record(store, observed_at=10, source="api", pool_id="account-a")
    original = store.latest("account-a")
    assert not record(store, observed_at=20, confirmed_at=25, source="api", pool_id="account-a")
    latest = QuotaObservationStore(path).latest("account-a")
    assert latest.id == original.id
    assert latest.source_timestamp == original.source_timestamp == 10
    assert latest.observed_at == original.observed_at == 10
    assert latest.confirmed_at == 25
    assert store.latest_at_or_before("account-a", 15) == latest
    assert len(store.history("account-a")) == 1


@pytest.mark.parametrize("changed", [False, True])
def test_older_acquisition_cannot_rewrite_confirmed_history(tmp_path, changed):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, observed_at=10, confirmed_at=20)
    original = store.history("account-a")
    change = {"used_percent": 30, "remaining_percent": 70} if changed else {}
    with pytest.raises(ValueError, match="out-of-order"):
        record(store, observed_at=19, **change)
    assert store.history("account-a") == original


def test_new_provenance_does_not_relabel_legacy_or_other_source_rows(tmp_path):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    assert record(store, observed_at=10)
    assert record(store, observed_at=11, source="api", pool_id="account-a")
    assert record(store, observed_at=12, source="headers", pool_id="account-a")
    history = store.history("account-a")
    assert [row.source for row in history] == ["legacy", "api", "headers"]
    assert [row.pool_id for row in history] == [None, "account-a", "account-a"]
    assert [row.confirmed_at for row in history] == [10, 11, 12]


@pytest.mark.parametrize("changes, reason", [
    ({"source": "api"}, "matching quota pool"),
    ({"source": "headers", "pool_id": "other-pool"}, "matching quota pool"),
    ({"source": "guessed"}, "source must"),
    ({"pool_id": ""}, "nonempty"),
    ({"used_percent": 101, "remaining_percent": -1}, "percentages"),
    ({"used_percent": 20, "remaining_percent": 70}, "sum to 100"),
    ({"observed_at": float("nan")}, "finite"),
    ({"reset_at": float("inf")}, "finite"),
    ({"confirmed_at": 9}, "confirmation cannot precede"),
])
def test_invalid_evidence_is_rejected_without_inserting(tmp_path, changes, reason):
    store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    values = {"observed_at": 10, **changes}
    with pytest.raises(ValueError, match=reason):
        record(store, **values)
    assert store.history("account-a") == []
