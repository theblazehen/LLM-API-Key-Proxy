import sys
import time
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "rotator_library"
sys.path.insert(0, str(SRC_ROOT))

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

for package_name, package_path in (
    ("rotator_library.usage", PACKAGE_ROOT / "usage"),
    ("rotator_library.usage.selection", PACKAGE_ROOT / "usage" / "selection"),
    (
        "rotator_library.usage.selection.strategies",
        PACKAGE_ROOT / "usage" / "selection" / "strategies",
    ),
    ("rotator_library.usage.limits", PACKAGE_ROOT / "usage" / "limits"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

fake_error_handler = types.ModuleType("rotator_library.error_handler")
fake_error_handler.mask_credential = lambda credential, style="default": credential
sys.modules.setdefault("rotator_library.error_handler", fake_error_handler)

from rotator_library.usage.selection.strategies.sequential import SequentialStrategy
from rotator_library.usage.selection.engine import SelectionEngine
from rotator_library.usage.limits import cooldowns
from rotator_library.usage.limits.cooldowns import CooldownChecker
from rotator_library.usage.types import (
    CooldownInfo,
    CredentialState,
    GroupStats,
    LimitCheckResult,
    LimitResult,
    RotationMode,
    SelectionContext,
    WindowStats,
)


def cooldown(reason, until, *, scope=None):
    return CooldownInfo(
        reason=reason,
        until=until,
        started_at=0.0,
        model_or_group=scope,
    )


def credential(account, *, remaining=None, reset_at=None, priority=1, source="codex"):
    state = CredentialState(
        stable_id=account,
        provider="codex",
        accessor=f"/credentials/{account}.json",
        priority=priority,
    )
    if remaining is not None or reset_at is not None:
        state.group_usage["weekly-limit"] = GroupStats(
            windows={
                "daily": WindowStats(
                    name="daily",
                    remaining_percent=remaining,
                    reset_at=reset_at,
                    quota_source=source,
                )
            }
        )
    return state


def context(*accounts, provider="codex", group="codex-global", priorities=None, usage=None):
    return SelectionContext(
        provider=provider,
        model="gpt-5-codex",
        quota_group=group,
        candidates=list(accounts),
        priorities=priorities or {account: 1 for account in accounts},
        usage_counts=usage or {},
        rotation_mode=RotationMode.SEQUENTIAL,
        rotation_tolerance=3.0,
        deadline=time.time() + 30,
    )


class _FakeLimits:
    def __init__(self, blocked=None):
        self.blocked = blocked or set()

    def check_all(self, state, _model, _quota_group):
        if state.stable_id in self.blocked:
            return LimitCheckResult.blocked(
                self.blocked[state.stable_id]
                if isinstance(self.blocked, dict)
                else LimitResult.BLOCKED_COOLDOWN,
                "test-unavailable",
            )
        return LimitCheckResult.ok()


class _FakeWindows:
    @staticmethod
    def get_primary_definition():
        return None


def test_generic_group_and_model_cooldowns_are_ineligible(monkeypatch):
    monkeypatch.setattr(cooldowns.time, "time", lambda: 100.0)
    checker = CooldownChecker()
    state = credential("account")
    state.provider = "generic"
    state.cooldowns = {
        "shared-group": cooldown("group rate limit", 200.0, scope="shared-group"),
        "generic-model": cooldown("model rate limit", 250.0, scope="generic-model"),
    }

    result = checker.check(state, "generic-model", "shared-group")
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_COOLDOWN
    assert result.blocked_until == 200.0

    state.cooldowns = {
        "shared-group": cooldown("expired group limit", 100.0, scope="shared-group"),
        "generic-model": cooldown("model rate limit", 180.0, scope="generic-model")
    }
    result = checker.check(state, "generic-model", "shared-group")
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_COOLDOWN
    assert result.blocked_until == 180.0


def test_codex_tracker_limit_cooldowns_are_ineligible(monkeypatch):
    monkeypatch.setattr(cooldowns.time, "time", lambda: 100.0)
    checker = CooldownChecker()
    state = credential("account")
    state.cooldowns = {
        "5h-limit": cooldown("short quota exhausted", 200.0, scope="5h-limit"),
        "weekly-limit": cooldown(
            "weekly quota exhausted", 300.0, scope="weekly-limit"
        ),
    }

    result = checker.check(state, "gpt-5-codex", "codex-global")
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_COOLDOWN
    assert result.blocked_until == 200.0

    state.cooldowns = {
        "weekly-limit": cooldown(
            "weekly quota exhausted", 300.0, scope="weekly-limit"
        )
    }
    result = checker.check(state, "gpt-5-codex", "codex-global")
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_COOLDOWN
    assert result.blocked_until == 300.0


def test_expired_and_unrelated_cooldowns_remain_eligible(monkeypatch):
    monkeypatch.setattr(cooldowns.time, "time", lambda: 100.0)
    checker = CooldownChecker()
    state = credential("account")
    state.provider = "generic"
    state.cooldowns = {
        "shared-group": cooldown("expired", 100.0, scope="shared-group"),
        "generic-model": cooldown("expired", 99.0, scope="generic-model"),
        "other-group": cooldown("unrelated", 200.0, scope="other-group"),
    }

    result = checker.check(state, "generic-model", "shared-group")
    assert result.allowed is True
    assert result.result == LimitResult.ALLOWED


def test_global_cooldown_is_ineligible_for_every_scope(monkeypatch):
    monkeypatch.setattr(cooldowns.time, "time", lambda: 100.0)
    checker = CooldownChecker()
    state = credential("account")
    state.provider = "generic"
    state.cooldowns["_global_"] = cooldown("credential disabled", 300.0)

    result = checker.check(state, "generic-model", "shared-group")
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_COOLDOWN
    assert result.blocked_until == 300.0


def affinity_engine(*, limits=None, clock=None, ttl=30, capacity=8):
    config = types.SimpleNamespace(
        rotation_mode=RotationMode.SEQUENTIAL,
        rotation_tolerance=3.0,
        sequential_fallback_multiplier=1,
        fair_cycle=types.SimpleNamespace(enabled=False),
    )
    return SelectionEngine(
        config,
        limits or _FakeLimits(),
        _FakeWindows(),
        prompt_cache_affinity_ttl_seconds=ttl,
        prompt_cache_affinity_max_entries=capacity,
        clock=clock or (lambda: 0.0),
    )


def select_with_affinity(engine, states, key, *, exclude=None):
    return engine.select(
        provider="codex",
        model="gpt-5-codex",
        states=states,
        quota_group="test-group",
        exclude=exclude,
        prompt_cache_key=key,
    )


def test_unavailable_sticky_falls_back_to_available_account():
    now = time.time()
    states = {
        "sticky": credential("sticky", remaining=50, reset_at=now + 5 * 86400),
        "fallback": credential("fallback", remaining=50, reset_at=now + 6 * 86400),
    }
    strategy = SequentialStrategy()
    strategy.select(context("sticky"), states)

    assert strategy.select(context("fallback"), states) == "fallback"


def test_external_quota_reset_is_honored_from_current_snapshot():
    now = time.time()
    states = {
        "reset_account": credential("reset_account", remaining=80, reset_at=now + 2 * 86400),
        "deadline_account": credential("deadline_account", remaining=70, reset_at=now + 5 * 86400),
    }
    strategy = SequentialStrategy()
    assert strategy.select(context("reset_account", "deadline_account"), states) == "reset_account"

    weekly = states["reset_account"].group_usage["weekly-limit"].windows["daily"]
    weekly.remaining_percent = 100
    weekly.reset_at = now + 8 * 86400

    assert strategy.select(context("reset_account", "deadline_account"), states) == "deadline_account"


def test_missing_weekly_quota_preserves_sequential_priority_order():
    states = {
        "lower_priority": credential("lower_priority", priority=2),
        "primary": credential("primary", priority=1),
    }

    selected = SequentialStrategy().select(
        context(
            "lower_priority",
            "primary",
            priorities={"lower_priority": 2, "primary": 1},
        ),
        states,
    )

    assert selected == "primary"


def test_non_codex_selection_keeps_existing_sequential_behavior():
    now = time.time()
    states = {
        "sticky": credential("sticky", remaining=5, reset_at=now + 25 * 3600, priority=2),
        "primary": credential("primary", remaining=100, reset_at=now + 7 * 86400, priority=1),
    }
    strategy = SequentialStrategy()
    strategy.select(
        context("sticky", provider="anthropic", group="shared", priorities={"sticky": 2}),
        states,
    )

    selected = strategy.select(
        context(
            "sticky",
            "primary",
            provider="anthropic",
            group="shared",
            priorities={"sticky": 2, "primary": 1},
        ),
        states,
    )

    assert selected == "primary"


def test_codex_prompt_cache_key_reuses_eligible_credential_over_normal_selection():
    states = {
        "bound": credential("bound", priority=2),
        "normal": credential("normal", priority=1),
    }
    engine = affinity_engine()

    assert select_with_affinity(engine, states, "conversation-1") == "normal"

    # A different key establishes an otherwise eligible lower-priority binding.
    assert select_with_affinity(engine, {"bound": states["bound"]}, "conversation-2") == "bound"
    assert select_with_affinity(engine, states, "conversation-2") == "bound"


def test_ineligible_affinity_falls_back_normally_and_remaps_key():
    states = {
        "bound": credential("bound", priority=1),
        "fallback": credential("fallback", priority=2),
    }
    limits = _FakeLimits()
    engine = affinity_engine(limits=limits)

    assert select_with_affinity(engine, states, "conversation") == "bound"
    limits.blocked.add("bound")

    assert select_with_affinity(engine, states, "conversation") == "fallback"
    assert engine._prompt_cache_affinity["conversation"][0] == "fallback"


def test_excluded_affinity_is_never_returned():
    states = {
        "bound": credential("bound", priority=1),
        "fallback": credential("fallback", priority=2),
    }
    engine = affinity_engine()

    assert select_with_affinity(engine, states, "conversation") == "bound"
    assert select_with_affinity(
        engine, states, "conversation", exclude={"bound"}
    ) == "fallback"


def test_cooldown_quota_or_concurrency_ineligible_affinity_never_returns_bound_key():
    states = {
        "bound": credential("bound", priority=1),
        "fallback": credential("fallback", priority=2),
    }

    for blocked_by in (
        LimitResult.BLOCKED_COOLDOWN,
        LimitResult.BLOCKED_WINDOW,
        LimitResult.BLOCKED_CONCURRENT,
    ):
        limits = _FakeLimits()
        engine = affinity_engine(limits=limits)
        assert select_with_affinity(engine, states, f"conversation-{blocked_by}") == "bound"

        limits.blocked = {"bound": blocked_by}
        assert (
            select_with_affinity(engine, states, f"conversation-{blocked_by}")
            == "fallback"
        )


def test_affinity_cache_expires_and_evicts_least_recently_used_entries():
    now = [0.0]
    states = {"only": credential("only")}
    engine = affinity_engine(clock=lambda: now[0], ttl=10, capacity=2)

    for key in ("old", "middle", "new"):
        assert select_with_affinity(engine, states, key) == "only"
        now[0] += 1

    assert list(engine._prompt_cache_affinity) == ["middle", "new"]

    now[0] = 13.0
    assert select_with_affinity(engine, states, "fresh") == "only"
    assert list(engine._prompt_cache_affinity) == ["fresh"]


def test_earliest_actual_reset_wins_despite_lower_remaining_and_credit_expiry():
    now = time.time()
    states = {
        "earliest": credential("earliest", remaining=1, reset_at=now + 4 * 86400),
        "later": credential("later", remaining=99, reset_at=now + 5 * 86400),
    }
    states["later"].reset_credit_count = 1
    states["later"].reset_credit_expiry_at = now + 3600
    strategy = SequentialStrategy()
    assert strategy.select(context("later"), states) == "later"
    assert strategy.select(context("later", "earliest"), states) == "earliest"


def select_luna(engine, states, key, *, model="gpt-5.6-luna"):
    return engine.select(
        provider="codex", model=model, states=states,
        quota_group="codex-global", prompt_cache_key=key,
    )


def reserve_credential(account, **kwargs):
    state = credential(account, **kwargs)
    # Selection consumes the eligibility contract; tracker tests own its rules.
    state.has_usable_luna_reserve = lambda model: model == "gpt-5.6-luna"
    return state


def test_luna_reserve_replaces_ordinary_affinity():
    ordinary = credential("ordinary")
    reserve = reserve_credential("reserve")
    engine = affinity_engine()
    assert select_luna(engine, {"ordinary": ordinary}, "chat") == "ordinary"
    assert select_luna(engine, {"ordinary": ordinary, "reserve": reserve}, "chat") == "reserve"


def test_luna_affinity_is_retained_within_reserve_pool():
    now = time.time()
    bound = reserve_credential("bound", remaining=0, reset_at=now + 5 * 86400)
    earlier = reserve_credential("earlier", remaining=0, reset_at=now + 2 * 86400)
    engine = affinity_engine()
    assert select_luna(engine, {"bound": bound}, "chat") == "bound"
    assert select_luna(engine, {"bound": bound, "earlier": earlier}, "chat") == "bound"


def test_regular_affinity_precedes_earliest_reset():
    now = time.time()
    bound = credential("bound", remaining=20, reset_at=now + 5 * 86400)
    earlier = credential("earlier", remaining=1, reset_at=now + 2 * 86400)
    engine = affinity_engine()
    assert select_luna(engine, {"bound": bound}, "chat") == "bound"
    states = {"bound": bound, "earlier": earlier}
    assert select_luna(engine, states, "chat") == "bound"
    assert select_luna(engine, states, "new-chat") == "earlier"


def test_non_luna_request_does_not_prefer_luna_reserve():
    now = time.time()
    ordinary = credential("ordinary", remaining=1, reset_at=now + 2 * 86400)
    reserve = reserve_credential("reserve", remaining=0, reset_at=now + 5 * 86400)
    assert select_luna(
        affinity_engine(), {"reserve": reserve, "ordinary": ordinary},
        "chat", model="gpt-6-astra",
    ) == "ordinary"


def test_unrelated_limits_still_block_reserve_selection():
    ordinary = credential("ordinary")
    reserve = reserve_credential("reserve")
    engine = affinity_engine(limits=_FakeLimits({"reserve": LimitResult.BLOCKED_CONCURRENT}))
    assert select_luna(engine, {"reserve": reserve, "ordinary": ordinary}, "chat") == "ordinary"
