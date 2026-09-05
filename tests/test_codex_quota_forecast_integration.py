"""Integration coverage for read-only Codex quota forecast composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from rotator_library.client import rotating_client as rotating_client_module
from rotator_library.client.rotating_client import RotatingClient
from rotator_library.providers.utilities.codex_quota_tracker import (
    CodexQuotaSnapshot,
    RateLimitWindow,
)
from rotator_library.usage.quota_observation_store import QuotaObservationStore


NOW = datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc)
QUOTA_DAY_START = NOW.replace(hour=6)
SOURCE_TIMESTAMP = (QUOTA_DAY_START - timedelta(minutes=30)).timestamp()
SOURCE_AGE_SECONDS = NOW.timestamp() - SOURCE_TIMESTAMP
RESET_AT = NOW.timestamp() + 3 * 24 * 60 * 60
PATH = "/credentials/codex-a.json"
POOL = "codex:upstream-account-a:weekly"


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def make_snapshot(*, remaining=74.375, account_id="upstream-account-a"):
    return CodexQuotaSnapshot(
        credential_path=PATH,
        identifier="codex-a.json",
        plan_type="pro",
        primary=None,
        secondary=RateLimitWindow(100 - remaining, remaining, 10_080, RESET_AT),
        credits=None,
        fetched_at=SOURCE_TIMESTAMP,
        status="success",
        error=None,
        account_id=account_id,
        source="api",
    )


class FakeCodexPlugin:
    def __init__(self, snapshot=None, *, source_error=None, history_error=None):
        self.snapshot = snapshot
        self.source_error = source_error
        self.history_error = history_error

    def get_cached_quota(self, credential_path):
        assert credential_path == PATH
        return self.snapshot

    def get_quota_error(self, credential_path):
        assert credential_path == PATH
        return self.source_error

    def get_quota_history_error(self, credential_path):
        assert credential_path == PATH
        return self.history_error


@pytest.fixture(autouse=True)
def fixed_composition_time(monkeypatch):
    monkeypatch.setattr(rotating_client_module, "datetime", FrozenDateTime)
    monkeypatch.setenv("TZ", "UTC")


def test_system_local_timezone_prefers_iana_tz_environment(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    local_timezone = rotating_client_module._system_local_timezone()
    assert isinstance(local_timezone, ZoneInfo)
    assert local_timezone.key == "America/New_York"
    assert datetime(2026, 1, 1, tzinfo=local_timezone).utcoffset() == timedelta(hours=-5)
    assert datetime(2026, 7, 1, tzinfo=local_timezone).utcoffset() == timedelta(hours=-4)


def make_client(tmp_path):
    client = RotatingClient.__new__(RotatingClient)
    client._quota_observation_store = QuotaObservationStore(tmp_path / "quota.sqlite3")
    return client


def make_stats(*, remaining=74.375):
    return {
        "credential_count": 1,
        "active_count": 1,
        "total_requests": 37,
        "tokens": {"input_cached": 1200, "input_uncached": 300, "output": 80},
        "reset_credits": {"available_count": 1, "next_expiry_at": NOW.timestamp() + 86400},
        "credentials": {
            "credential-key": {
                "stable_id": "credential-identity-not-quota-pool",
                "email": "codex-a@example.test",
                "full_path": PATH,
                "status": "active",
                "reset_credits": {
                    "available_count": 1,
                    "status": "success",
                    "stale": False,
                    "details_complete": True,
                    "credits": [{
                        "id": "credit-a", "status": "available",
                        "reset_type": "codex_rate_limits",
                        "expires_at": NOW.timestamp() + 86400,
                    }],
                },
                "group_usage": {"weekly-limit": {"windows": {"daily": {
                    "used_percent": 100 - remaining,
                    "remaining_percent": remaining,
                    "window_minutes": 10_080,
                    "reset_at": RESET_AT,
                    "quota_source": "codex",
                }}}},
            },
        },
    }


def observation_rows(client):
    with sqlite3.connect(client._quota_observation_store.path) as connection:
        return connection.execute(
            "SELECT * FROM codex_weekly_observations ORDER BY id"
        ).fetchall()


def assert_original_stats_preserved(stats, original):
    for key, value in original.items():
        if key != "credentials":
            assert stats[key] == value
    for key, credential in original["credentials"].items():
        assert {field: stats["credentials"][key][field] for field in credential} == credential


def test_get_projects_cached_snapshot_without_inserting_or_confirming_history(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    snapshot = make_snapshot()
    client._observe_codex_quota(snapshot)
    before = observation_rows(client)
    stats = make_stats()
    original = deepcopy(stats)
    plugin = FakeCodexPlugin(snapshot)

    def unexpected_record(**_kwargs):
        pytest.fail("GET must not insert or confirm quota observations")

    monkeypatch.setattr(client._quota_observation_store, "record", unexpected_record)
    client._attach_codex_quota_forecast(stats, plugin)
    forecast = deepcopy(stats["forecast"])
    client._attach_codex_quota_forecast(stats, plugin)

    assert_original_stats_preserved(stats, original)
    assert stats["credentials"]["credential-key"]["quota_pool_id"] == POOL
    assert stats["forecast"] == forecast
    assert observation_rows(client) == before
    assert forecast["schema_version"] == 3
    assert forecast["status"] == "ready"
    assert forecast["source_timestamp"] == SOURCE_TIMESTAMP
    assert forecast["age_seconds"] == pytest.approx(SOURCE_AGE_SECONDS)
    assert forecast["stale"] is True
    assert forecast["timezone"] == "UTC"
    assert forecast["quota_day_start_hour"] == 6
    assert forecast["horizon"] == {
        "start_at": QUOTA_DAY_START.timestamp(),
        "end_at": (QUOTA_DAY_START + timedelta(days=7)).timestamp(),
        "day_count": 7,
    }
    assert forecast["actual"]["status"] == "unavailable"
    assert forecast["actual"]["estimate"] is None
    assert forecast["accounts"][0]["account_id"] == POOL
    assert forecast["accounts"][0]["remaining_percent"] == 74.375
    assert len(forecast["days"]) == 7


def test_changed_manager_balance_and_anchor_cannot_change_cached_forecast(tmp_path):
    client = make_client(tmp_path)
    plugin = FakeCodexPlugin(make_snapshot())
    stats = make_stats()
    client._attach_codex_quota_forecast(stats, plugin)
    forecast = deepcopy(stats["forecast"])
    daily = stats["credentials"]["credential-key"]["group_usage"]["weekly-limit"]["windows"]["daily"]
    daily.update(remaining_percent=0, used_percent=100, reset_at=NOW.timestamp() + 60)

    client._attach_codex_quota_forecast(stats, plugin)

    assert stats["forecast"] == forecast
    assert forecast["accounts"][0]["natural_reset_at"] == RESET_AT
    assert observation_rows(client) == []


def test_successful_ingestion_records_coherent_source_and_get_only_reads_it(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    first = make_snapshot()
    second = replace(
        first, fetched_at=NOW.timestamp() - 60,
        secondary=replace(first.secondary, used_percent=26.125, remaining_percent=73.875),
    )
    client._observe_codex_quota(first)
    client._observe_codex_quota(second)
    rows = observation_rows(client)
    captured = {}
    original_builder = rotating_client_module.build_codex_quota_forecast

    def capture_forecast(**kwargs):
        captured.update(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(rotating_client_module, "build_codex_quota_forecast", capture_forecast)
    stats = make_stats(remaining=0)
    client._attach_codex_quota_forecast(stats, FakeCodexPlugin(second))

    assert observation_rows(client) == rows
    history = captured["observations"]
    assert [row["remaining_percent"] for row in history] == [74.375, 73.875]
    for row in history:
        assert row["pool_id"] == row["stable_id"] == POOL
        assert row["source"] == "api"
        assert row["source_timestamp"] == row["observed_at"] == row["confirmed_at"]
        assert row["reset_at"] == RESET_AT
    assert stats["forecast"]["source_timestamp"] == second.fetched_at
    assert stats["forecast"]["accounts"][0]["remaining_percent"] == 73.875


@pytest.mark.parametrize("missing_snapshot", [False, True])
def test_missing_upstream_identity_never_uses_credential_identity_as_pool(tmp_path, missing_snapshot):
    client = make_client(tmp_path)
    snapshot = None if missing_snapshot else make_snapshot(account_id=None)
    stats = make_stats()

    client._attach_codex_quota_forecast(stats, FakeCodexPlugin(snapshot))

    forecast = stats["forecast"]
    assert stats["credentials"]["credential-key"]["quota_pool_id"] is None
    assert forecast["status"] == "unavailable"
    assert forecast["today"] is None
    assert forecast["days"] == []
    assert forecast["accounts"] == []
    reason = "quota_snapshot_unavailable" if missing_snapshot else "missing_upstream_pool_identity"
    assert reason in forecast["unknown_accounts"][0]["reasons"]
    assert observation_rows(client) == []


def test_refresh_error_exposes_stale_cached_source_without_fabricating_new_measurement(tmp_path):
    client = make_client(tmp_path)
    stats = make_stats()
    plugin = FakeCodexPlugin(make_snapshot(), source_error="HTTP 503")

    client._attach_codex_quota_forecast(stats, plugin)

    assert stats["credentials"]["credential-key"]["quota_source_error"] == "HTTP 503"
    assert stats["forecast"]["status"] == "ready"
    assert stats["forecast"]["stale"] is True
    assert stats["forecast"]["source_timestamp"] == SOURCE_TIMESTAMP
    assert stats["forecast"]["accounts"][0]["remaining_percent"] == 74.375


def test_history_failure_does_not_hide_live_balance_or_claim_trusted_history(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    stats = make_stats()
    error = "quota_observation_persistence_failed"
    plugin = FakeCodexPlugin(make_snapshot(), history_error=error)
    captured = {}
    original_builder = rotating_client_module.build_codex_quota_forecast

    def unavailable_history(*_args, **_kwargs):
        pytest.fail("untrusted history must not be queried")

    def capture_forecast(**kwargs):
        captured.update(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(client._quota_observation_store, "observations_around_boundary", unavailable_history)
    monkeypatch.setattr(rotating_client_module, "build_codex_quota_forecast", capture_forecast)
    client._attach_codex_quota_forecast(stats, plugin)

    credential = stats["credentials"]["credential-key"]
    assert credential["quota_history_error"] == error
    assert credential["quota_source_error"] is None
    assert captured["accounts"][0]["history_provenance"] == "unknown"
    assert captured["accounts"][0]["history_error"] == error
    assert captured["observations"] == []
    assert stats["forecast"]["status"] == "ready"
    assert stats["forecast"]["accounts"][0]["remaining_percent"] == 74.375
    assert stats["forecast"]["actual"]["status"] == "unavailable"


def test_composition_failure_retains_stats_and_returns_no_fabricated_plan(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    stats = make_stats()
    original = deepcopy(stats)

    def fail_forecast(**_kwargs):
        raise RuntimeError("deterministic forecast failure")

    monkeypatch.setattr(rotating_client_module, "build_codex_quota_forecast", fail_forecast)
    client._attach_codex_quota_forecast(stats, FakeCodexPlugin(make_snapshot()))

    assert_original_stats_preserved(stats, original)
    forecast = stats["forecast"]
    assert forecast["schema_version"] == 3
    assert forecast["status"] == "unavailable"
    assert forecast["reason"] == "forecast_composition_failed"
    assert forecast["source_timestamp"] == SOURCE_TIMESTAMP
    assert forecast["age_seconds"] == pytest.approx(SOURCE_AGE_SECONDS)
    assert forecast["stale"] is True
    assert forecast["timezone"] == "UTC"
    assert forecast["quota_day_start_hour"] == 6
    assert forecast["today"] is None
    assert forecast["days"] == forecast["planning_days"] == []
    assert forecast["actual"]["estimate"] is None
    assert forecast["actual"]["lower_bound"] is None
    assert forecast["actual"]["upper_bound"] is None
    assert forecast["risk"] == {"aggregate_exhaustion": None, "basis": "weekly_natural_resets"}
    assert forecast["accounts"] == forecast["credit_scenarios"] == []
