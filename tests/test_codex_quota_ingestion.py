"""Quota publication ordering and failure isolation, without upstream requests."""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy

import pytest

from rotator_library.providers.utilities import codex_quota_tracker as tracker_module


NOW = 1_786_197_600
OLD_RESET = NOW + 1000
NEW_RESET = NOW + 604800


class RecordingUsageManager:
    """Observable routing state driven by the real tracker reconciler."""

    def __init__(self):
        self.windows = {}
        self.cooldowns = set()
        self.updates = []
        self.clears = []

    async def update_quota_baseline(self, **kwargs):
        self.updates.append(kwargs)
        key = (kwargs["accessor"], kwargs["quota_group"])
        self.windows[key] = kwargs
        if kwargs["apply_exhaustion"]:
            self.cooldowns.add(key)

    async def clear_quota_group_state(self, accessor, group, *, remove_usage=True):
        self.clears.append((accessor, group))
        self.cooldowns.discard((accessor, group))
        if remove_usage:
            self.windows.pop((accessor, group), None)

    async def clear_cooldown_if_exists(self, *, accessor, model_or_group):
        self.cooldowns.discard((accessor, model_or_group))


class Tracker(tracker_module.CodexQuotaTracker):
    def __init__(self):
        self._init_quota_tracker()
        self._credentials_cache = {
            "a": {"account_id": "upstream-pool"},
            "b": {"account_id": "upstream-pool"},
        }
        self.manager = RecordingUsageManager()
        self.set_usage_manager(self.manager)
        self.observations = []
        self.set_quota_observer(self.observations.append)

    async def get_auth_header(self, credential_path):
        return {"Authorization": "Bearer offline-test-only"}

    async def get_account_id(self, credential_path):
        return self._credentials_cache[credential_path]["account_id"]


def payload(used, reset):
    return {"rate_limit": {"secondary_window": {
        "used_percent": used,
        "limit_window_seconds": 604800,
        "reset_at": reset,
    }}}


def headers(used, reset):
    return {
        "x-codex-secondary-used-percent": str(used),
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-reset-at": str(reset),
    }


@pytest.fixture
def http_queue(monkeypatch):
    """Each response may pause after acquisition until explicitly released."""
    responses = deque()

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class HTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            data, acquired, release = responses.popleft()
            if acquired is not None:
                acquired.set()
            if release is not None:
                await release.wait()
            return Response(data)

    monkeypatch.setattr(tracker_module.httpx, "AsyncClient", HTTP)
    monkeypatch.setattr(tracker_module.time, "time", lambda: NOW)
    return responses


@pytest.mark.asyncio
@pytest.mark.parametrize("new_path", ["a", "b"], ids=["same-credential", "credential-alias"])
@pytest.mark.parametrize("new_source", ["api", "headers"])
async def test_delayed_pre_reset_api_cannot_replace_newer_reset_snapshot(http_queue, new_path, new_source):
    tracker = Tracker()
    acquired, release = asyncio.Event(), asyncio.Event()
    http_queue.append((payload(100, OLD_RESET), acquired, release))
    old_task = asyncio.create_task(tracker.fetch_quota_from_api("a"))
    try:
        await asyncio.wait_for(acquired.wait(), timeout=2)
        if new_source == "api":
            http_queue.append((payload(1, NEW_RESET), None, None))
            fresh = await tracker.fetch_quota_from_api(new_path)
        else:
            fresh = tracker.update_quota_from_headers(new_path, headers(1, NEW_RESET))
            await asyncio.gather(*tuple(tracker._quota_push_tasks))
        assert fresh.status == "success"
        assert tracker.manager.windows[(new_path, "weekly-limit")]["quota_remaining_percent"] == 99
        assert tracker.manager.windows[(new_path, "weekly-limit")]["quota_reset_ts"] == NEW_RESET
        release.set()
        obsolete = await asyncio.wait_for(old_task, timeout=2)
    finally:
        release.set()
        if not old_task.done():
            old_task.cancel()
        await asyncio.gather(old_task, return_exceptions=True)

    assert obsolete.status == "error"
    assert obsolete.error == "quota_refresh_superseded"
    assert tracker.get_cached_quota(new_path) is fresh
    assert tracker.observations == [fresh]
    assert len(tracker.manager.updates) == 1
    assert tracker.get_quota_error(new_path) is None
    if new_path != "a":
        assert tracker.get_cached_quota("a") is None
        assert ("a", "weekly-limit") not in tracker.manager.windows


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_window", [
    {"limit_window_seconds": 604800, "reset_at": OLD_RESET},
    {"used_percent": 100, "limit_window_seconds": 604800},
    {"used_percent": float("nan"), "limit_window_seconds": 604800, "reset_at": OLD_RESET},
    {"used_percent": 100, "limit_window_seconds": 0, "reset_at": OLD_RESET},
])
async def test_malformed_present_api_window_preserves_exhausted_cache_and_routing(http_queue, invalid_window):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    exhausted = await tracker.fetch_quota_from_api("a")
    state = deepcopy(tracker.manager.windows)
    clears = list(tracker.manager.clears)
    http_queue.append(({"rate_limit": {
        "primary_window": {"used_percent": 0, "limit_window_seconds": 18000, "reset_at": OLD_RESET},
        "secondary_window": invalid_window,
    }}, None, None))

    rejected = await tracker.fetch_quota_from_api("a")

    assert rejected.status == "error"
    assert tracker.get_cached_quota("a") is exhausted
    assert tracker.manager.windows == state
    assert tracker.manager.clears == clears
    assert ("a", "weekly-limit") in tracker.manager.cooldowns
    assert tracker.observations == [exhausted]
    assert tracker.get_quota_error("a")


@pytest.mark.asyncio
async def test_malformed_present_header_window_preserves_existing_exhaustion(http_queue):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    exhausted = await tracker.fetch_quota_from_api("a")
    malformed = headers(100, OLD_RESET)
    del malformed["x-codex-secondary-used-percent"]
    malformed.update({
        "x-codex-primary-used-percent": "0",
        "x-codex-primary-window-minutes": "300",
        "x-codex-primary-reset-at": str(OLD_RESET),
    })

    assert tracker.update_quota_from_headers("a", malformed) is None
    assert tracker.get_cached_quota("a") is exhausted
    assert ("a", "weekly-limit") in tracker.manager.cooldowns
    assert len(tracker.manager.updates) == 1
    assert tracker.observations == [exhausted]


@pytest.mark.asyncio
async def test_explicitly_absent_window_can_remove_prior_routing_constraint(http_queue):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    await tracker.fetch_quota_from_api("a")
    http_queue.append(({"rate_limit": {"secondary_window": None}}, None, None))

    absent = await tracker.fetch_quota_from_api("a")

    assert absent.status == "success"
    assert tracker.get_cached_quota("a") is absent
    assert absent.weekly_window is None
    assert ("a", "weekly-limit") not in tracker.manager.windows
    assert ("a", "weekly-limit") not in tracker.manager.cooldowns


@pytest.mark.asyncio
@pytest.mark.parametrize("include_empty_reset", [True, False], ids=["empty-reset", "missing-reset"])
async def test_real_headers_with_explicit_inactive_secondary_are_ingested(include_empty_reset):
    tracker = Tracker()
    real_headers = {
        "x-codex-primary-used-percent": "37",
        "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-reset-at": "1788019973",
        "x-codex-secondary-used-percent": "0",
        "x-codex-secondary-window-minutes": "0",
        "x-codex-secondary-reset-after-seconds": "0",
    }
    if include_empty_reset:
        real_headers["x-codex-secondary-reset-at"] = ""
    real_headers.update({key.replace("x-codex-", "x-extra-"): value
                         for key, value in list(real_headers.items())})

    snapshot = tracker.update_quota_from_headers("a", real_headers)
    await asyncio.gather(*tuple(tracker._quota_push_tasks))

    assert snapshot.status == "success"
    assert snapshot.primary.remaining_percent == 63
    assert snapshot.secondary is None
    assert snapshot.weekly_window is snapshot.primary
    assert snapshot.families["codex"][1] is None
    assert snapshot.families["extra"][1] is None
    assert tracker.get_cached_quota("a") is snapshot
    assert tracker.observations == [snapshot]
    assert tracker.get_quota_error("a") is None
    assert tracker.manager.windows[("a", "weekly-limit")]["quota_remaining_percent"] == 63
    assert ("a", "5h-limit") not in tracker.manager.windows


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_secondary", [
    {"used-percent": "1", "window-minutes": "0", "reset-at": ""},
    {"used-percent": "0", "window-minutes": "0", "reset-at": str(OLD_RESET)},
    {"used-percent": "0", "window-minutes": "0", "reset-at": "not-a-time"},
    {"used-percent": "0", "window-minutes": "0", "reset-at": "", "reset-after-seconds": "1"},
    {"window-minutes": "0", "reset-at": ""},
    {"used-percent": "0", "window-minutes": "300", "reset-at": ""},
])
async def test_inactive_secondary_sentinel_does_not_accept_malformed_windows(http_queue, invalid_secondary):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    exhausted = await tracker.fetch_quota_from_api("a")
    state = deepcopy(tracker.manager.windows)
    candidate = {
        "x-codex-primary-used-percent": "37",
        "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-reset-at": str(NEW_RESET),
        **{f"x-codex-secondary-{key}": value for key, value in invalid_secondary.items()},
    }

    assert tracker.update_quota_from_headers("a", candidate) is None
    assert tracker.get_cached_quota("a") is exhausted
    assert tracker.observations == [exhausted]
    assert tracker.manager.windows == state
    assert ("a", "weekly-limit") in tracker.manager.cooldowns
    assert tracker.get_quota_error("a") == "invalid_quota_headers"


@pytest.mark.asyncio
async def test_explicit_inactive_only_headers_remove_a_disabled_constraint(http_queue):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    await tracker.fetch_quota_from_api("a")

    disabled = tracker.update_quota_from_headers("a", {
        "x-codex-secondary-used-percent": "0",
        "x-codex-secondary-window-minutes": "0",
        "x-codex-secondary-reset-at": "",
    })
    await asyncio.gather(*tuple(tracker._quota_push_tasks))

    assert disabled.status == "success"
    assert disabled.secondary is None
    assert disabled.weekly_window is None
    assert tracker.get_cached_quota("a") is disabled
    assert tracker.observations[-1] is disabled
    assert ("a", "weekly-limit") not in tracker.manager.windows
    assert ("a", "weekly-limit") not in tracker.manager.cooldowns


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["api", "headers"])
async def test_observer_failure_cannot_abort_success_or_skip_routing_reconciliation(http_queue, source):
    tracker = Tracker()
    http_queue.append((payload(100, OLD_RESET), None, None))
    await tracker.fetch_quota_from_api("a")

    def fail_observation(_snapshot):
        raise OSError("deterministic SQLite storage failure")

    tracker.set_quota_observer(fail_observation)
    if source == "api":
        http_queue.append((payload(2, NEW_RESET), None, None))
        live = await tracker.fetch_quota_from_api("a")
    else:
        # This synchronous call is the successful provider-response header path.
        live = tracker.update_quota_from_headers("a", headers(2, NEW_RESET))
        await asyncio.gather(*tuple(tracker._quota_push_tasks))

    assert live.status == "success"
    assert tracker.get_cached_quota("a") is live
    assert tracker.get_quota_error("a") is None
    assert tracker.get_quota_history_error("a") == "quota_observation_persistence_failed"
    assert tracker.manager.windows[("a", "weekly-limit")]["quota_remaining_percent"] == 98
    assert tracker.manager.windows[("a", "weekly-limit")]["quota_reset_ts"] == NEW_RESET
    assert ("a", "weekly-limit") not in tracker.manager.cooldowns

    # An acquisition error must not overwrite separate persistence health.
    http_queue.append(({"rate_limit": {"secondary_window": {"used_percent": 2}}}, None, None))
    rejected = await tracker.fetch_quota_from_api("a")
    assert rejected.status == "error"
    assert tracker.get_quota_history_error("a") == "quota_observation_persistence_failed"
    assert tracker.get_cached_quota("a") is live

    tracker.set_quota_observer(tracker.observations.append)
    http_queue.append((payload(3, NEW_RESET), None, None))
    recovered = await tracker.fetch_quota_from_api("a")
    assert recovered.status == "success"
    assert tracker.get_quota_error("a") is None
    assert tracker.get_quota_history_error("a") is None
    assert tracker.observations[-1] is recovered
