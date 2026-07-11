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
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

fake_error_handler = types.ModuleType("rotator_library.error_handler")
fake_error_handler.mask_credential = lambda credential, style="default": credential
sys.modules.setdefault("rotator_library.error_handler", fake_error_handler)

from rotator_library.usage.selection.strategies.sequential import SequentialStrategy
from rotator_library.usage.types import (
    CredentialState,
    GroupStats,
    RotationMode,
    SelectionContext,
    WindowStats,
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


def test_higher_early_deadline_pressure_wins_over_sequential_priority():
    now = time.time()
    states = {
        "primary": credential("primary", remaining=90, reset_at=now + 7 * 86400, priority=1),
        "urgent": credential("urgent", remaining=45, reset_at=now + 2 * 86400, priority=2),
    }

    selected = SequentialStrategy().select(
        context("primary", "urgent", priorities={"primary": 1, "urgent": 2}), states
    )

    assert selected == "urgent"


def test_sticky_account_remains_when_pressure_difference_is_below_hysteresis():
    now = time.time()
    states = {
        "sticky": credential("sticky", remaining=40, reset_at=now + 4 * 86400),
        "other": credential("other", remaining=35, reset_at=now + 4 * 86400),
    }
    strategy = SequentialStrategy()
    assert strategy.select(context("sticky", "other"), states) == "sticky"

    states["other"].group_usage["weekly-limit"].windows["daily"].remaining_percent = 44

    assert strategy.select(context("sticky", "other"), states) == "sticky"


def test_materially_more_urgent_account_replaces_sticky():
    now = time.time()
    states = {
        "sticky": credential("sticky", remaining=30, reset_at=now + 5 * 86400),
        "urgent": credential("urgent", remaining=70, reset_at=now + 3 * 86400),
    }
    strategy = SequentialStrategy()
    strategy.select(context("sticky"), states)

    assert strategy.select(context("sticky", "urgent"), states) == "urgent"


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
