"""Integration coverage for Codex quota forecast endpoint composition."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rotator_library.client import rotating_client as rotating_client_module
from rotator_library.client.rotating_client import RotatingClient
from rotator_library.usage.quota_observation_store import QuotaObservationStore


NOW = datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc)
LOCAL_NOW = NOW.astimezone()
FORECAST_TIMEZONE = str(rotating_client_module._system_local_timezone())
QUOTA_DAY_START = LOCAL_NOW.replace(hour=6, minute=0, second=0, microsecond=0)
if LOCAL_NOW < QUOTA_DAY_START:
    QUOTA_DAY_START -= timedelta(days=1)
HORIZON_END = QUOTA_DAY_START + timedelta(days=7)
SOURCE_TIMESTAMP = (QUOTA_DAY_START - timedelta(minutes=30)).timestamp()
SOURCE_AGE_SECONDS = NOW.timestamp() - SOURCE_TIMESTAMP
RESET_AT = NOW.timestamp() + 3 * 24 * 60 * 60


class FrozenDateTime(datetime):
    """Keep endpoint composition independent from the wall clock."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[no-untyped-def]
        return NOW if tz is None else NOW.astimezone(tz)


class FakeCodexPlugin:
    def get_cached_quota(self, credential_path: str) -> SimpleNamespace:
        assert credential_path == "/credentials/codex-a.json"
        return SimpleNamespace(fetched_at=SOURCE_TIMESTAMP)


def test_system_local_timezone_prefers_iana_tz_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "America/New_York")

    local_timezone = rotating_client_module._system_local_timezone()

    assert isinstance(local_timezone, ZoneInfo)
    assert local_timezone.key == "America/New_York"
    assert datetime(2026, 1, 1, tzinfo=local_timezone).utcoffset() == timedelta(hours=-5)
    assert datetime(2026, 7, 1, tzinfo=local_timezone).utcoffset() == timedelta(hours=-4)


def make_client(tmp_path) -> RotatingClient:  # type: ignore[no-untyped-def]
    client = RotatingClient.__new__(RotatingClient)
    client._quota_observation_store = QuotaObservationStore(
        tmp_path / "usage" / "quota_observations.sqlite3"
    )
    return client


def make_stats(*, remaining: float = 74.375) -> dict[str, object]:
    return {
        "credential_count": 1,
        "active_count": 1,
        "total_requests": 37,
        "tokens": {
            "input_cached": 1200,
            "input_uncached": 300,
            "output": 80,
        },
        "reset_credits": {
            "mode": "observe",
            "available_count": 1,
            "next_auto_redeem_at": NOW.timestamp() + 2 * 24 * 60 * 60,
        },
        "credentials": {
            "credential-key": {
                "stable_id": "codex-account-a",
                "email": "codex-a@example.test",
                "full_path": "/credentials/codex-a.json",
                "status": "active",
                "reset_credits": {
                    "available_count": 1,
                    "credits": [
                        {
                            "id": "credit-a",
                            "status": "available",
                            "expires_at": NOW.timestamp() + 2 * 24 * 60 * 60,
                            "auto_redeems_at": NOW.timestamp() + 2 * 24 * 60 * 60,
                        }
                    ],
                },
                "group_usage": {
                    "weekly-limit": {
                        "windows": {
                            "daily": {
                                "used_percent": 100.0 - remaining,
                                "remaining_percent": remaining,
                                "window_minutes": 10_080,
                                "reset_at": RESET_AT,
                                "quota_source": "codex",
                            }
                        }
                    }
                },
            }
        },
    }


def observation_rows(client: RotatingClient) -> list[sqlite3.Row]:
    connection = sqlite3.connect(client._quota_observation_store.path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT stable_account_id, remaining_percent_text, remaining_percent
            FROM codex_weekly_observations
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()


@pytest.fixture(autouse=True)
def fixed_composition_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rotating_client_module, "datetime", FrozenDateTime)


def test_forecast_is_additive_and_identical_composition_dedupes_observation(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    client = make_client(tmp_path)
    plugin = FakeCodexPlugin()
    stats = make_stats()
    original = deepcopy(stats)

    client._attach_codex_quota_forecast(stats, plugin)

    assert {key: stats[key] for key in original} == original
    forecast = stats["forecast"]
    assert isinstance(forecast, dict)
    assert forecast["schema_version"] == 2
    assert forecast["status"] == "ready"
    assert forecast["unit"] == "weekly_quota_percentage_points"
    assert forecast["source_timestamp"] == SOURCE_TIMESTAMP
    assert forecast["age_seconds"] == pytest.approx(SOURCE_AGE_SECONDS)
    assert SOURCE_AGE_SECONDS > 900.0
    assert forecast["stale"] is True
    assert forecast["timezone"] == FORECAST_TIMEZONE
    assert forecast["quota_day_start_hour"] == 6
    assert forecast["horizon"] == {
        "start_at": QUOTA_DAY_START.timestamp(),
        "end_at": HORIZON_END.timestamp(),
        "day_count": 7,
    }
    assert forecast["actual"] == {
        "status": "observed",
        "used_since_day_start": pytest.approx(0.0),
    }
    assert forecast["accounts"][0]["remaining_percent"] == pytest.approx(74.375)
    assert len(forecast["days"]) == 7
    assert len(observation_rows(client)) == 1

    first_forecast = deepcopy(forecast)
    client._attach_codex_quota_forecast(stats, plugin)

    assert stats["forecast"] == first_forecast
    assert len(observation_rows(client)) == 1


def test_persisted_forecast_observation_includes_its_reset_anchor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client = make_client(tmp_path)
    captured: dict[str, object] = {}

    def capture_forecast(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"schema_version": 2}

    monkeypatch.setattr(
        rotating_client_module, "build_codex_quota_forecast", capture_forecast
    )
    client._attach_codex_quota_forecast(make_stats(), FakeCodexPlugin())

    observations = captured["observations"]
    assert isinstance(observations, list)
    assert observations[0]["reset_at"] == RESET_AT


def test_changed_weekly_balance_inserts_observation_and_updates_forecast(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    client = make_client(tmp_path)
    plugin = FakeCodexPlugin()
    stats = make_stats(remaining=74.375)
    client._attach_codex_quota_forecast(stats, plugin)

    daily = stats["credentials"]["credential-key"]["group_usage"]["weekly-limit"][
        "windows"
    ]["daily"]
    daily["remaining_percent"] = 73.875
    daily["used_percent"] = 26.125
    client._attach_codex_quota_forecast(stats, plugin)

    rows = observation_rows(client)
    assert [row["stable_account_id"] for row in rows] == [
        "codex-account-a",
        "codex-account-a",
    ]
    assert [row["remaining_percent_text"] for row in rows] == ["74.375", "73.875"]
    assert [row["remaining_percent"] for row in rows] == pytest.approx(
        [74.375, 73.875]
    )
    assert stats["forecast"]["accounts"][0]["remaining_percent"] == pytest.approx(
        73.875
    )


def test_composition_failure_retains_stats_and_returns_explicit_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client = make_client(tmp_path)
    plugin = FakeCodexPlugin()
    stats = make_stats()
    original = deepcopy(stats)

    def fail_forecast(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("deterministic forecast failure")

    monkeypatch.setattr(
        rotating_client_module, "build_codex_quota_forecast", fail_forecast
    )

    client._attach_codex_quota_forecast(stats, plugin)

    assert {key: stats[key] for key in original} == original
    forecast = stats["forecast"]
    assert forecast["schema_version"] == 2
    assert forecast["status"] == "unavailable"
    assert forecast["reason"] == "forecast_composition_failed"
    assert forecast["source_timestamp"] == SOURCE_TIMESTAMP
    assert forecast["age_seconds"] == pytest.approx(SOURCE_AGE_SECONDS)
    assert forecast["stale"] is True
    assert forecast["timezone"] == FORECAST_TIMEZONE
    assert forecast["quota_day_start_hour"] == 6
    assert forecast["horizon"] == {
        "start_at": QUOTA_DAY_START.timestamp(),
        "end_at": HORIZON_END.timestamp(),
        "day_count": 7,
    }
    assert forecast["actual"] == {
        "status": "unavailable",
        "used_since_day_start": None,
    }
    assert forecast["risk"] == {
        "aggregate_exhaustion": False,
        "basis": "forecast_targets",
    }
    assert len(forecast["days"]) == 7
    assert forecast["today"] == forecast["days"][0]
    for index, day in enumerate(forecast["days"]):
        start = QUOTA_DAY_START + timedelta(days=index)
        end = QUOTA_DAY_START + timedelta(days=index + 1)
        assert day == {
            "index": index,
            "start_at": start.timestamp(),
            "end_at": end.timestamp(),
            "local_date": start.date().isoformat(),
            "target": 0.0,
            "remaining_target": 0.0,
            "baseline_allocation": 0.0,
            "expiry_bonus": 0.0,
            "sustainable_daily_rate": 0.0,
            "live_sustainable_remaining": 0.0,
            "expected_used_by_now": 0.0,
            "contributions": [],
            "reset_events": [],
        }
    assert forecast["accounts"] == []
    assert forecast["unknown_accounts"] == []
