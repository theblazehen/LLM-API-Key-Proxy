"""Reserve admission boundaries and authoritative refresh lifecycle."""

import asyncio
from dataclasses import replace

import pytest

from rotator_library.providers.utilities import codex_quota_tracker as quota
from rotator_library.usage.types import CredentialState, CooldownInfo
from rotator_library.usage.limits.cooldowns import CooldownChecker
from test_codex_quota_ingestion import Tracker, http_queue, headers, NOW, NEW_RESET


def reserve_payload(main_used=100):
    return {
        "rate_limit": {
            "allowed": main_used < 100, "limit_reached": main_used >= 100,
            "primary_window": {"used_percent": main_used,
                               "limit_window_seconds": 604800, "reset_at": NEW_RESET},
        },
        "additional_rate_limits": [{
            "limit_name": "gpt-reserve", "metered_feature": "base_model_inference",
            "normal_model_slug": "gpt-5.6-luna",
            "rate_limit": {"allowed": True, "limit_reached": False,
                           "primary_window": {"used_percent": 0,
                                              "limit_window_seconds": 604800,
                                              "reset_at": NEW_RESET}},
        }],
    }


def tracked_state(tracker):
    state = CredentialState(stable_id="account", provider="codex", accessor="a")
    tracker.manager._states["account"] = state
    return state


@pytest.mark.asyncio
async def test_reserve_only_activates_after_regular_exhaustion(http_queue):
    tracker = Tracker()
    state = tracked_state(tracker)
    http_queue.append((reserve_payload(87), None, None))
    await tracker.fetch_quota_from_api("a")
    assert not state.has_usable_luna_reserve("codex/gpt-5.6-luna")
    tracker.update_quota_from_headers("a", headers(100, NEW_RESET))
    await asyncio.gather(*tuple(tracker._quota_push_tasks))
    assert state.has_usable_luna_reserve("codex/gpt-5.6-luna:max")
    assert not state.has_usable_luna_reserve("codex/gpt-6-astra")
    assert not state.has_usable_luna_reserve("other/gpt-5.6-luna")
    assert not state.has_usable_luna_reserve("codex/gpt-5.6-luna:unknown")
    tracker.update_quota_from_headers("a", headers(1, NEW_RESET))
    await asyncio.gather(*tuple(tracker._quota_push_tasks))
    assert not state.has_usable_luna_reserve("gpt-5.6-luna")


@pytest.mark.asyncio
async def test_headers_do_not_extend_reserve_freshness(http_queue, monkeypatch):
    tracker = Tracker()
    state = tracked_state(tracker)
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    monkeypatch.setattr(quota.time, "time", lambda: NOW + 901)
    tracker.update_quota_from_headers("a", headers(100, NEW_RESET))
    await asyncio.gather(*tuple(tracker._quota_push_tasks))
    assert not state.has_usable_luna_reserve("gpt-5.6-luna")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["removed", "denied", "exhausted", "expired", "wrong-model", "malformed", "error"])
async def test_authoritative_loss_of_reserve_fails_closed(http_queue, change):
    tracker = Tracker()
    state = tracked_state(tracker)
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    assert state.has_usable_luna_reserve("gpt-5.6-luna")
    data = reserve_payload()
    entry = data["additional_rate_limits"][0]
    rate = entry["rate_limit"]
    if change == "removed":
        data.pop("additional_rate_limits")
    elif change == "denied":
        rate["allowed"] = False
    elif change == "exhausted":
        rate["primary_window"]["used_percent"] = 100
    elif change == "expired":
        rate["primary_window"]["reset_at"] = NOW
    elif change == "wrong-model":
        entry["normal_model_slug"] = "gpt-6-astra"
    elif change == "malformed":
        rate["allowed"] = "true"
    else:
        data["rate_limit"] = {"primary_window": {"used_percent": "bad"}}
    http_queue.append((data, None, None))
    await tracker.fetch_quota_from_api("a")
    assert not state.has_usable_luna_reserve("gpt-5.6-luna")
    # A subsequent ordinary response cannot resurrect revoked metadata.
    tracker.update_quota_from_headers("a", headers(100, NEW_RESET))
    await asyncio.gather(*tuple(tracker._quota_push_tasks))
    assert not state.has_usable_luna_reserve("gpt-5.6-luna")


@pytest.mark.asyncio
async def test_reserve_bypasses_only_authoritative_main_quota_cooldown(http_queue):
    tracker = Tracker()
    state = tracked_state(tracker)
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    checker = CooldownChecker()
    main = CooldownInfo("quota_exhausted", NEW_RESET, NOW, "api_quota", "weekly-limit")
    state.cooldowns["weekly-limit"] = main
    assert checker.check(state, "codex/gpt-5.6-luna", "codex-global").allowed
    assert not checker.check(state, "codex/gpt-6-astra", "codex-global").allowed
    for source in ("rate_limit", "custom_cap", "provider_hook", "system"):
        state.cooldowns["weekly-limit"] = replace(main, source=source)
        assert not checker.check(state, "codex/gpt-5.6-luna", "codex-global").allowed
    state.cooldowns["weekly-limit"] = main
    for scope in ("_global_", "codex-global", "codex/gpt-5.6-luna"):
        state.cooldowns[scope] = replace(main, model_or_group=scope)
        assert not checker.check(state, "codex/gpt-5.6-luna", "codex-global").allowed
        del state.cooldowns[scope]


@pytest.mark.asyncio
async def test_main_exhaustion_outlives_overlapping_provider_cooldown(http_queue, monkeypatch):
    tracker = Tracker()
    state = tracked_state(tracker)
    state.cooldowns["weekly-limit"] = CooldownInfo(
        "provider_limit", NOW + 60, NOW, "provider_hook", "weekly-limit",
    )
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    checker = CooldownChecker()
    assert not checker.check(state, "codex/gpt-5.6-luna", "codex-global").allowed
    assert not checker.check(state, "codex/gpt-6-astra", "codex-global").allowed
    monkeypatch.setattr(quota.time, "time", lambda: NOW + 61)
    assert checker.check(state, "codex/gpt-5.6-luna", "codex-global").allowed
    blocked = checker.check(state, "codex/gpt-6-astra", "codex-global")
    assert not blocked.allowed
    assert blocked.blocked_until == NEW_RESET


@pytest.mark.asyncio
async def test_cold_alias_headers_preserve_account_reserve(http_queue):
    tracker = Tracker()
    state = tracked_state(tracker)
    alias = CredentialState(stable_id="alias", provider="codex", accessor="b")
    tracker.manager._states["alias"] = alias
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    tracker.update_quota_from_headers("b", headers(100, NEW_RESET))
    await asyncio.gather(*tuple(tracker._quota_push_tasks))
    assert state.has_usable_luna_reserve("gpt-5.6-luna")
    assert alias.has_usable_luna_reserve("gpt-5.6-luna")


@pytest.mark.asyncio
async def test_cold_alias_receives_and_revokes_account_reserve(http_queue):
    tracker = Tracker()
    state = tracked_state(tracker)
    alias = CredentialState(stable_id="alias", provider="codex", accessor="b")
    tracker.manager._states["alias"] = alias
    http_queue.append((reserve_payload(), None, None))
    await tracker.fetch_quota_from_api("a")
    assert alias.has_usable_luna_reserve("gpt-5.6-luna")
    http_queue.append(({"rate_limit": {"primary_window": {"used_percent": "bad"}}}, None, None))
    await tracker.fetch_quota_from_api("b")
    assert not alias.has_usable_luna_reserve("gpt-5.6-luna")
    assert not state.has_usable_luna_reserve("gpt-5.6-luna")
