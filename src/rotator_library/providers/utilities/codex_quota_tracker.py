# src/rotator_library/providers/utilities/codex_quota_tracker.py
"""
Codex Quota Tracking Mixin

Provides quota tracking functionality for the Codex provider by:
1. Fetching rate limit status from the /usage endpoint
2. Parsing rate limit headers from API responses
3. Storing quota baselines in UsageManager

Rate Limit Structure (from Codex API):
- Primary/secondary are positional fields, not stable window identities.
- Window duration determines whether a limit is short-term or weekly.
- Credits: Account credit balance info

Required from provider:
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self.get_account_id(credential_path) -> Optional[str]
    - self._credentials_cache: Dict[str, Dict[str, Any]]
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_credential_identifier(credential_path: str) -> str:
    """Extract a short identifier from a credential path."""
    if credential_path.startswith("env://"):
        return credential_path
    return Path(credential_path).name


def _seconds_to_minutes(seconds: Optional[int]) -> Optional[int]:
    """Convert seconds to minutes, or None if input is None."""
    if seconds is None:
        return None
    return seconds // 60


# =============================================================================
# CONFIGURATION
# =============================================================================

# Codex usage API endpoint
# The Codex CLI uses different paths based on PathStyle:
# - If base contains /backend-api: use /wham/usage (ChatGptApi style)
# - Otherwise: use /api/codex/usage (CodexApi style)
# Since we use chatgpt.com/backend-api, we need /wham/usage
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
CODEX_RESET_CREDITS_URL = (
    "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
)
CODEX_RESET_CONSUME_URL = f"{CODEX_RESET_CREDITS_URL}/consume"

# Rate limit header names (from Codex API)
HEADER_PRIMARY_USED_PERCENT = "x-codex-primary-used-percent"
HEADER_PRIMARY_WINDOW_MINUTES = "x-codex-primary-window-minutes"
HEADER_PRIMARY_RESET_AT = "x-codex-primary-reset-at"
HEADER_SECONDARY_USED_PERCENT = "x-codex-secondary-used-percent"
HEADER_SECONDARY_WINDOW_MINUTES = "x-codex-secondary-window-minutes"
HEADER_SECONDARY_RESET_AT = "x-codex-secondary-reset-at"
HEADER_CREDITS_HAS_CREDITS = "x-codex-credits-has-credits"
HEADER_CREDITS_UNLIMITED = "x-codex-credits-unlimited"
HEADER_CREDITS_BALANCE = "x-codex-credits-balance"

# Default quota refresh interval (5 minutes)
DEFAULT_QUOTA_REFRESH_INTERVAL = 300

# Stale threshold - quota data older than this is considered stale (15 minutes)
QUOTA_STALE_THRESHOLD_SECONDS = 900

# Upstream currently uses a seven-day window for the weekly account limit.
# It may appear in either the primary or secondary payload position.
WEEKLY_WINDOW_MINUTES = 7 * 24 * 60
RESET_CREDITS_STALE_SECONDS = 15 * 60
RESET_AUTO_MODE = os.getenv("CODEX_RESET_MODE", "observe").strip().lower()
RESET_NATURAL_GUARD_SECONDS = int(
    float(os.getenv("CODEX_RESET_NATURAL_GUARD_HOURS", "24")) * 3600
)
RESET_AUTO_WAIT_SECONDS = int(os.getenv("CODEX_RESET_AUTO_WAIT_SECONDS", "900"))
RESET_DRAIN_LEAD_SECONDS = int(
    float(os.getenv("CODEX_RESET_DRAIN_LEAD_HOURS", "24")) * 3600
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class RateLimitWindow:
    """Rate limit window info from Codex API."""

    used_percent: float  # 0-100
    remaining_percent: float  # 100 - used_percent
    window_minutes: Optional[int]
    reset_at: Optional[float]  # Unix timestamp

    def __post_init__(self) -> None:
        for value in (self.used_percent, self.remaining_percent):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("quota percentages must be finite and between0 and100")
        if not math.isclose(self.used_percent + self.remaining_percent, 100.0, abs_tol=1e-6):
            raise ValueError("quota percentages must sum to100")
        if self.reset_at is not None and (isinstance(self.reset_at, bool) or not isinstance(self.reset_at, (int, float)) or not math.isfinite(self.reset_at) or self.reset_at <= 0):
            raise ValueError("quota reset timestamp must be positive and finite")

    @property
    def remaining_fraction(self) -> float:
        """Get remaining quota as a fraction (0.0 to 1.0)."""
        return max(0.0, min(1.0, (100 - self.used_percent) / 100))

    @property
    def is_exhausted(self) -> bool:
        """Check if this window's quota is exhausted."""
        return self.used_percent >= 100

    def seconds_until_reset(self) -> Optional[float]:
        """Calculate seconds until reset, or None if unknown."""
        if self.reset_at is None:
            return None
        return max(0, self.reset_at - time.time())


@dataclass(frozen=True)
class CreditsInfo:
    """Credits info from Codex API."""

    has_credits: bool
    unlimited: bool
    balance: Optional[str]  # Could be numeric string or "unlimited"


@dataclass(frozen=True)
class CodexQuotaSnapshot:
    """Complete quota snapshot for a Codex credential."""

    credential_path: str
    identifier: str
    plan_type: Optional[str]
    primary: Optional[RateLimitWindow]
    secondary: Optional[RateLimitWindow]
    credits: Optional[CreditsInfo]
    fetched_at: float
    status: str  # "success" or "error"
    error: Optional[str]
    # Every rate-limit family advertised on the response, keyed by limit id
    # (e.g. "codex", "codex_secondary"). Observational: routing still runs off
    # primary/secondary above. Defaulted so existing constructors keep working.
    families: Mapping[str, Tuple[Optional[RateLimitWindow], Optional[RateLimitWindow]]] = (
        field(default_factory=dict)
    )
    account_id: Optional[str] = None
    source: str = "headers"

    def __post_init__(self) -> None:
        if not math.isfinite(self.fetched_at) or self.fetched_at <= 0:
            raise ValueError("quota source timestamp must be positive and finite")
        if self.account_id is not None and (not isinstance(self.account_id, str) or not self.account_id.strip()):
            raise ValueError("quota account identity must be a nonempty string")
        object.__setattr__(self, "families", MappingProxyType(dict(self.families)))

    @property
    def quota_pool_id(self) -> Optional[str]:
        return f"codex:{self.account_id}:weekly" if self.account_id else None

    @property
    def weekly_window(self) -> Optional[RateLimitWindow]:
        windows = [window for window in (self.primary, self.secondary)
                   if window and window.window_minutes == WEEKLY_WINDOW_MINUTES]
        return windows[0] if len(windows) == 1 else None

    @property
    def weekly_unknown_reason(self) -> Optional[str]:
        if not self.account_id:
            return "missing_upstream_pool_identity"
        windows = [window for window in (self.primary, self.secondary)
                   if window and window.window_minutes == WEEKLY_WINDOW_MINUTES]
        if len(windows) > 1:
            return "ambiguous_weekly_windows"
        if not windows:
            return "weekly_window_unavailable"
        if windows[0].reset_at is None:
            return "weekly_reset_unavailable"
        return None

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS


@dataclass(frozen=True)
class ResetCredit:
    id: str
    reset_type: str
    status: str
    granted_at: Optional[float]
    expires_at: Optional[float]
    title: Optional[str]
    description: Optional[str]


@dataclass(frozen=True)
class ResetCreditsSnapshot:
    available_count: int
    credits: tuple[ResetCredit, ...]
    fetched_at: float
    status: str
    error: Optional[str] = None
    policy_action: str = "observe"
    policy_reason: str = "not evaluated"
    last_outcome: Optional[str] = None
    last_redeemed_at: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > RESET_CREDITS_STALE_SECONDS

    @property
    def next_expiry_at(self) -> Optional[float]:
        return min(
            (
                credit.expires_at
                for credit in self.credits
                if credit.status == "available" and credit.expires_at is not None
            ),
            default=None,
        )


def _parse_credit_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) and value > 0 else None
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _window_to_dict(window: RateLimitWindow) -> Dict[str, Any]:
    """Convert RateLimitWindow to dict for JSON serialization."""
    return {
        "remaining_percent": window.remaining_percent,
        "remaining_fraction": window.remaining_fraction,
        "used_percent": window.used_percent,
        "window_minutes": window.window_minutes,
        "reset_at": window.reset_at,
        "reset_in_seconds": window.seconds_until_reset(),
        "is_exhausted": window.is_exhausted,
    }


def _classify_quota_windows(*windows: Any) -> Dict[str, Any]:
    """Classify positional upstream windows by their advertised duration."""
    classified: Dict[str, Any] = {}
    for window in windows:
        if not window:
            continue
        minutes = (
            window.get("window_minutes")
            if isinstance(window, dict)
            else window.window_minutes
        )
        group = "weekly-limit" if minutes == WEEKLY_WINDOW_MINUTES else "5h-limit"
        existing = classified.get(group)
        if existing is None:
            classified[group] = window
            continue
        # Upstream can advertise more than one limit family with the same
        # advertised duration (observed in production: two weekly windows whose
        # reset times differ by hours). Overwriting here lets a nearly-full
        # limit mask an exhausted one, and the rotation router then believes a
        # dead credential still has capacity. Keep the MOST CONSTRAINING window.
        if _window_used_percent(window) > _window_used_percent(existing):
            classified[group] = window
        lib_logger.info(
            "Codex quota group collision on %s: used_percent %.0f (reset %s) vs "
            "%.0f (reset %s); keeping the most constrained",
            group,
            _window_used_percent(existing),
            _window_reset_at(existing),
            _window_used_percent(window),
            _window_reset_at(window),
        )
    return classified


def _window_used_percent(window: Any) -> float:
    """Read used_percent from either a dict or a RateLimitWindow."""
    value = (
        window.get("used_percent")
        if isinstance(window, dict)
        else getattr(window, "used_percent", None)
    )
    return float(value) if isinstance(value, (int, float)) else 0.0


def _window_reset_at(window: Any) -> Any:
    """Read reset_at from either a dict or a RateLimitWindow."""
    return (
        window.get("reset_at")
        if isinstance(window, dict)
        else getattr(window, "reset_at", None)
    )


def _credits_to_dict(credits: CreditsInfo) -> Dict[str, Any]:
    """Convert CreditsInfo to dict for JSON serialization."""
    return {
        "has_credits": credits.has_credits,
        "unlimited": credits.unlimited,
        "balance": credits.balance,
    }


# =============================================================================
# HEADER PARSING
# =============================================================================


def parse_rate_limit_headers(headers: Dict[str, str]) -> CodexQuotaSnapshot:
    """
    Parse rate limit information from Codex API response headers.

    Args:
        headers: Response headers dict

    Returns:
        CodexQuotaSnapshot with parsed rate limit data
    """
    primary = _parse_window_from_headers(
        headers,
        HEADER_PRIMARY_USED_PERCENT,
        HEADER_PRIMARY_WINDOW_MINUTES,
        HEADER_PRIMARY_RESET_AT,
    )

    secondary = _parse_window_from_headers(
        headers,
        HEADER_SECONDARY_USED_PERCENT,
        HEADER_SECONDARY_WINDOW_MINUTES,
        HEADER_SECONDARY_RESET_AT,
    )

    credits = _parse_credits_from_headers(headers)

    malformed = False
    has_inactive_window = False
    for used_key, minutes_key, reset_key, window in (
        (HEADER_PRIMARY_USED_PERCENT, HEADER_PRIMARY_WINDOW_MINUTES, HEADER_PRIMARY_RESET_AT, primary),
        (HEADER_SECONDARY_USED_PERCENT, HEADER_SECONDARY_WINDOW_MINUTES, HEADER_SECONDARY_RESET_AT, secondary),
    ):
        present = any(key in headers for key in (used_key, minutes_key, reset_key))
        inactive = _inactive_header_window(headers, used_key, minutes_key, reset_key)
        has_inactive_window = has_inactive_window or inactive
        if present and not inactive and (window is None or window.window_minutes is None or window.window_minutes <= 0 or window.reset_at is None):
            malformed = True

    return CodexQuotaSnapshot(
        credential_path="",
        identifier="",
        plan_type=None,
        primary=primary,
        secondary=secondary,
        credits=credits,
        fetched_at=time.time(),
        status="error" if malformed else ("success" if (primary or secondary or credits or has_inactive_window) else "no_data"),
        error="invalid_quota_headers" if malformed else None,
        families=parse_all_rate_limit_families(headers),
    )


_PRIMARY_USED_SUFFIX = "-primary-used-percent"


def parse_all_rate_limit_families(
    headers: Dict[str, str],
) -> Dict[str, Tuple[Optional[RateLimitWindow], Optional[RateLimitWindow]]]:
    """Discover every rate-limit family advertised on a response.

    Upstream templates its headers as ``x-{limit_id}-primary-used-percent``
    (and ``-window-minutes`` / ``-reset-at``, plus the same for ``secondary``),
    where ``limit_id`` defaults to ``codex`` but may be anything. Reading only
    the canonical pair hides additional limits, which is how an exhausted
    weekly window ended up masked by a nearly-full one.

    Returns a mapping of normalised limit id -> (primary, secondary).
    """
    lowered = {str(k).lower(): v for k, v in headers.items()}

    limit_ids = {"codex"}
    for name in lowered:
        if not name.endswith(_PRIMARY_USED_SUFFIX):
            continue
        raw = name[: -len(_PRIMARY_USED_SUFFIX)]
        if raw.startswith("x-"):
            raw = raw[2:]
        if raw:
            limit_ids.add(raw.replace("-", "_"))

    families: Dict[
        str, Tuple[Optional[RateLimitWindow], Optional[RateLimitWindow]]
    ] = {}
    for limit_id in sorted(limit_ids):
        prefix = "x-" + limit_id.replace("_", "-")
        primary = _parse_window_from_headers(
            lowered,
            f"{prefix}-primary-used-percent",
            f"{prefix}-primary-window-minutes",
            f"{prefix}-primary-reset-at",
        )
        secondary = _parse_window_from_headers(
            lowered,
            f"{prefix}-secondary-used-percent",
            f"{prefix}-secondary-window-minutes",
            f"{prefix}-secondary-reset-at",
        )
        if primary or secondary:
            families[limit_id] = (primary, secondary)
    return families


def _inactive_header_window(
    headers: Dict[str, str], used_key: str, minutes_key: str, reset_key: str,
) -> bool:
    """Recognize the provider's explicit disabled-window sentinel, not missing data."""
    if used_key not in headers or minutes_key not in headers:
        return False
    if headers.get(reset_key) not in (None, ""):
        return False
    try:
        if float(headers[used_key]) != 0 or float(headers[minutes_key]) != 0:
            return False
        after_key = reset_key.removesuffix("-reset-at") + "-reset-after-seconds"
        return after_key not in headers or float(headers[after_key]) == 0
    except (TypeError, ValueError):
        return False


def _parse_window_from_headers(
    headers: Dict[str, str],
    used_percent_header: str,
    window_minutes_header: str,
    reset_at_header: str,
) -> Optional[RateLimitWindow]:
    """Parse a single rate limit window from headers."""
    if _inactive_header_window(headers, used_percent_header, window_minutes_header, reset_at_header):
        return None
    used_percent_str = headers.get(used_percent_header)
    if not used_percent_str:
        return None

    try:
        used_percent = float(used_percent_str)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(used_percent) or not 0 <= used_percent <= 100:
        return None

    # Parse optional fields
    window_minutes = None
    window_minutes_str = headers.get(window_minutes_header)
    if window_minutes_str:
        try:
            window_minutes = int(window_minutes_str)
        except (ValueError, TypeError):
            pass

    reset_at = None
    reset_at_str = headers.get(reset_at_header)
    if reset_at_str:
        try:
            reset_at = int(reset_at_str)
            if reset_at <= 0:
                reset_at = None
        except (ValueError, TypeError):
            pass

    return RateLimitWindow(
        used_percent=used_percent,
        remaining_percent=100 - used_percent,
        window_minutes=window_minutes,
        reset_at=reset_at,
    )


def _parse_api_window(data: Any) -> Optional[RateLimitWindow]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("present quota window must be an object")
    used = data.get("used_percent")
    if isinstance(used, bool) or not isinstance(used, (int, float)):
        raise ValueError("present quota window has missing or invalid usage")
    if not math.isfinite(used) or not 0 <= used <= 100:
        raise ValueError("present quota window has invalid usage")
    seconds = data.get("limit_window_seconds")
    minutes = None
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0 and seconds % 60 == 0:
        minutes = int(seconds / 60)
    if minutes is None:
        raise ValueError("present quota window has missing or invalid duration")
    reset = data.get("reset_at")
    if isinstance(reset, bool) or not isinstance(reset, (int, float)) or not math.isfinite(reset) or reset <= 0:
        raise ValueError("present quota window has missing or invalid reset timestamp")
    return RateLimitWindow(float(used), 100.0 - float(used), minutes, reset)


def _parse_credits_from_headers(headers: Dict[str, str]) -> Optional[CreditsInfo]:
    """Parse credits info from headers."""
    has_credits_str = headers.get(HEADER_CREDITS_HAS_CREDITS)
    if has_credits_str is None:
        return None

    has_credits = has_credits_str.lower() in ("true", "1")
    unlimited_str = headers.get(HEADER_CREDITS_UNLIMITED, "false")
    unlimited = unlimited_str.lower() in ("true", "1")
    balance = headers.get(HEADER_CREDITS_BALANCE)

    return CreditsInfo(
        has_credits=has_credits,
        unlimited=unlimited,
        balance=balance,
    )


# =============================================================================
# QUOTA TRACKER MIXIN
# =============================================================================


class CodexQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Codex provider.

    This mixin adds the following capabilities:
    - Fetch rate limit status from the Codex /usage API endpoint
    - Parse rate limit headers from streaming responses
    - Store quota baselines in UsageManager
    - Get structured quota info for all credentials

    Usage:
        class CodexProvider(OpenAIOAuthBase, CodexQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_cache: Dict[str, CodexQuotaSnapshot] = {}
        self._quota_refresh_interval: int = 300
    """

    # Type hints for attributes from provider
    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, CodexQuotaSnapshot]
    _quota_refresh_interval: int

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Provider mixin contract."""
        raise NotImplementedError

    async def get_account_id(self, credential_path: str) -> Optional[str]:
        """Provider mixin contract."""
        raise NotImplementedError

    def set_quota_observer(self, callback: Callable[[CodexQuotaSnapshot], None]) -> None:
        self._quota_observer = callback

    def _advance_quota_epoch(self, key: str) -> int:
        epoch = self._quota_epochs.get(key, 0) + 1
        self._quota_epochs[key] = epoch
        return epoch

    def _quota_guard_current(self, guard: Mapping[str, int]) -> bool:
        return all(self._quota_epochs.get(key) == epoch for key, epoch in guard.items())

    def _publish_quota_snapshot(self, snapshot: CodexQuotaSnapshot) -> None:
        """Publish one coherent immutable state and synchronously persist its evidence."""
        self._advance_quota_epoch(f"path:{snapshot.credential_path}")
        if snapshot.account_id:
            self._advance_quota_epoch(f"account:{snapshot.account_id}")
        self._quota_cache[snapshot.credential_path] = snapshot
        self._quota_errors.pop(snapshot.credential_path, None)
        if self._quota_observer is None:
            self._quota_history_errors[snapshot.credential_path] = "quota_observer_unavailable"
            return
        try:
            self._quota_observer(snapshot)
        except Exception:
            # A storage outage invalidates historical evidence, not a successful
            # provider response or its coherent current quota. Reconciliation
            # must still run after this publication.
            self._quota_history_errors[snapshot.credential_path] = "quota_observation_persistence_failed"
            lib_logger.exception("Failed to persist Codex quota observation")
        else:
            self._quota_history_errors.pop(snapshot.credential_path, None)

    def get_quota_error(self, credential_path: str) -> Optional[str]:
        return self._quota_errors.get(credential_path)

    def get_quota_history_error(self, credential_path: str) -> Optional[str]:
        return self._quota_history_errors.get(credential_path)

    def _init_quota_tracker(self):
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, CodexQuotaSnapshot] = {}
        self._quota_refresh_interval: int = DEFAULT_QUOTA_REFRESH_INTERVAL
        self._usage_manager: Optional["UsageManager"] = None
        self._initial_baselines_fetched: bool = False
        self._quota_observer: Optional[Callable[[CodexQuotaSnapshot], None]] = None
        self._quota_errors: Dict[str, str] = {}
        self._quota_history_errors: Dict[str, str] = {}
        self._quota_epochs: Dict[str, int] = {}
        self._quota_push_locks: Dict[str, asyncio.Lock] = {}
        self._quota_push_tasks: set[asyncio.Task] = set()
        self._reset_credits_cache: Dict[str, ResetCreditsSnapshot] = {}
        self._reset_locks: Dict[str, asyncio.Lock] = {}

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        """Set the UsageManager reference for pushing quota updates."""
        self._usage_manager = usage_manager

    # =========================================================================
    # QUOTA API FETCHING
    # =========================================================================

    async def _account_headers(self, credential_path: str) -> Dict[str, str]:
        headers = {
            **(await self.get_auth_header(credential_path)),
            "Content-Type": "application/json",
            "User-Agent": "codex-cli",
        }
        account_id = await self.get_account_id(credential_path)
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        return headers

    async def fetch_reset_credits(
        self, credential_path: str
    ) -> ResetCreditsSnapshot:
        """Refresh one account's reset-credit list outside request routing."""
        previous = self._reset_credits_cache.get(credential_path)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    CODEX_RESET_CREDITS_URL,
                    headers=await self._account_headers(credential_path),
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
            credits = []
            for item in payload.get("credits") or []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                credits.append(
                    ResetCredit(
                        id=str(item["id"]),
                        reset_type=str(item.get("reset_type") or "unknown"),
                        status=str(item.get("status") or "unknown").lower(),
                        granted_at=_parse_credit_timestamp(item.get("granted_at")),
                        expires_at=_parse_credit_timestamp(item.get("expires_at")),
                        title=item.get("title"),
                        description=item.get("description"),
                    )
                )
            credits.sort(
                key=lambda credit: credit.expires_at
                if credit.expires_at is not None
                else float("inf")
            )
            snapshot = ResetCreditsSnapshot(
                available_count=max(0, int(payload.get("available_count") or 0)),
                credits=tuple(credits),
                fetched_at=time.time(),
                status="success",
                last_outcome=previous.last_outcome if previous else None,
                last_redeemed_at=previous.last_redeemed_at if previous else None,
            )
        except Exception as exc:
            snapshot = ResetCreditsSnapshot(
                available_count=previous.available_count if previous else 0,
                credits=previous.credits if previous else (),
                fetched_at=time.time(),
                status="error",
                error=str(exc),
                last_outcome=previous.last_outcome if previous else None,
                last_redeemed_at=previous.last_redeemed_at if previous else None,
                policy_reason="reset-credit refresh failed",
            )
        self._reset_credits_cache[credential_path] = snapshot
        return snapshot

    async def redeem_reset_credit(
        self,
        credential_path: str,
        credit_id: Optional[str] = None,
        redeem_request_id: Optional[str] = None,
        policy_credentials: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Redeem explicitly, serialized per account and reconciled afterward."""
        lock = self._reset_locks.setdefault(credential_path, asyncio.Lock())
        async with lock:
            quota = await self.fetch_quota_from_api(credential_path)
            credits = await self.fetch_reset_credits(credential_path)
            if quota.status != "success" or credits.status != "success":
                raise RuntimeError("fresh quota and reset-credit state are required")
            if credit_id:
                matching = [
                    credit
                    for credit in credits.credits
                    if credit.id == credit_id and credit.status == "available"
                ]
                if not matching:
                    raise ValueError("reset credit is not available")
            if policy_credentials is not None:
                # Automatic redemption must still be justified by fresh state
                # after acquiring the account lock. Manual redemption is an
                # explicit operator override and does not pass this argument.
                await asyncio.gather(
                    *(
                        self.fetch_quota_from_api(path)
                        for path in policy_credentials
                        if path != credential_path
                    ),
                    return_exceptions=True,
                )
                action, reason, fresh_credit_id = self._reset_policy(
                    credential_path, policy_credentials
                )
                if action != "redeem" or fresh_credit_id != credit_id:
                    raise RuntimeError(
                        f"automatic redemption no longer eligible: {reason}"
                    )
            body: Dict[str, str] = {
                "redeem_request_id": redeem_request_id or str(uuid.uuid4())
            }
            if credit_id:
                body["credit_id"] = credit_id
            consume_headers = await self._account_headers(credential_path)
            epoch_keys = [f"path:{credential_path}"]
            if consume_headers.get("ChatGPT-Account-Id"):
                epoch_keys.append(f"account:{consume_headers['ChatGPT-Account-Id']}")
            for key in epoch_keys:
                self._advance_quota_epoch(key)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    CODEX_RESET_CONSUME_URL,
                    headers=consume_headers,
                    json=body,
                    timeout=30,
                )
                response.raise_for_status()
                outcome = response.json()
            for key in epoch_keys:
                self._advance_quota_epoch(key)
            reconciled = await self.fetch_quota_from_api(credential_path)
            if reconciled.status != "success":
                raise RuntimeError("reset request completed but quota reconciliation failed; refresh before any further redemption")
            refreshed = await self.fetch_reset_credits(credential_path)
            outcome_name = str(outcome.get("code") or outcome.get("status") or "unknown")
            self._reset_credits_cache[credential_path] = ResetCreditsSnapshot(
                **{
                    **refreshed.__dict__,
                    "last_outcome": outcome_name,
                    "last_redeemed_at": time.time(),
                }
            )
            return {**outcome, "redeem_request_id": body["redeem_request_id"]}

    def get_reset_credit_info(self, credential_path: str) -> Optional[Dict[str, Any]]:
        snapshot = self._reset_credits_cache.get(credential_path)
        if snapshot is None:
            return None
        return {
            "available_count": snapshot.available_count,
            "status": snapshot.status,
            "error": snapshot.error,
            "fetched_at": snapshot.fetched_at,
            "stale": snapshot.is_stale,
            "details_complete": (
                len([credit for credit in snapshot.credits if credit.status == "available"])
                == snapshot.available_count
                == len({credit.id for credit in snapshot.credits if credit.status == "available"})
            ),
            "next_expiry_at": snapshot.next_expiry_at,
            "policy": {
                "action": snapshot.policy_action,
                "reason": snapshot.policy_reason,
            },
            "last_outcome": snapshot.last_outcome,
            "last_redeemed_at": snapshot.last_redeemed_at,
            "credits": [
                {
                    "id": credit.id,
                    "reset_type": credit.reset_type,
                    "status": credit.status,
                    "granted_at": credit.granted_at,
                    "expires_at": credit.expires_at,
                    "title": credit.title,
                    "description": credit.description,
                }
                for credit in snapshot.credits
            ],
        }

    def _reset_policy(
        self, credential_path: str, all_credentials: List[str]
    ) -> tuple[str, str, Optional[str]]:
        """Evaluate cached state only; safe to call from routing/status paths."""
        reset = self._reset_credits_cache.get(credential_path)
        quota = self._quota_cache.get(credential_path)
        if not reset or reset.status != "success" or reset.is_stale:
            return "observe", "reset-credit state unavailable or stale", None
        if not quota or quota.status != "success" or quota.is_stale:
            return "observe", "quota state unavailable or stale", None
        available = [c for c in reset.credits if c.status == "available"]
        if reset.available_count <= 0:
            return "observe", "no reset credits available", None
        if not available:
            return "manual", "credit details unavailable", None

        windows = [w for w in (quota.primary, quota.secondary) if w is not None]
        if not windows:
            return "observe", "quota windows unavailable", None
        if not any(window.is_exhausted for window in windows):
            return "drain", "preserve reset while quota remains", available[0].id

        now = time.time()
        natural_resets = [w.reset_at for w in windows if w.is_exhausted and w.reset_at]
        natural_reset = min(natural_resets) if natural_resets else None
        alternative_available = False
        for other_path in all_credentials:
            if other_path == credential_path:
                continue
            other = self._quota_cache.get(other_path)
            if not other or other.status != "success" or other.is_stale:
                continue
            other_windows = [w for w in (other.primary, other.secondary) if w]
            if other_windows and not any(w.is_exhausted for w in other_windows):
                alternative_available = True
                break
        if natural_reset and natural_reset - now <= RESET_AUTO_WAIT_SECONDS:
            return "wait", "natural reset is imminent", available[0].id
        if (
            reset.next_expiry_at
            and reset.next_expiry_at - now <= RESET_AUTO_WAIT_SECONDS
        ):
            return "wait", "credit expiry is imminent; automatic refill unverified", available[0].id
        if (
            alternative_available
            and natural_reset
            and natural_reset - now <= RESET_NATURAL_GUARD_SECONDS
        ):
            return "wait", "alternative capacity covers a near natural reset", available[0].id
        if alternative_available:
            return "wait", "another account has usable capacity", available[0].id
        if RESET_AUTO_MODE != "automatic":
            return "manual", "blocked demand warrants redemption", available[0].id
        return "redeem", "all accounts are quota-blocked", available[0].id

    async def _refresh_and_evaluate_resets(self, credentials: List[str]) -> None:
        """Background-only refresh and optional redemption; never request-path awaited."""
        await asyncio.gather(
            *(self.fetch_reset_credits(path) for path in credentials),
            return_exceptions=True,
        )
        for path in credentials:
            snapshot = self._reset_credits_cache.get(path)
            if snapshot is None:
                continue
            action, reason, credit_id = self._reset_policy(path, credentials)
            self._reset_credits_cache[path] = ResetCreditsSnapshot(
                **{
                    **snapshot.__dict__,
                    "policy_action": action,
                    "policy_reason": reason,
                }
            )
            if action != "redeem" or not credit_id:
                continue
            try:
                await self.redeem_reset_credit(
                    path,
                    credit_id=credit_id,
                    policy_credentials=credentials,
                )
            except Exception as exc:
                current = self._reset_credits_cache[path]
                self._reset_credits_cache[path] = ResetCreditsSnapshot(
                    **{
                        **current.__dict__,
                        "policy_action": "error",
                        "policy_reason": f"automatic redemption failed: {exc}",
                    }
                )
                lib_logger.error(
                    "Codex automatic reset failed for %s: %s",
                    _get_credential_identifier(path),
                    exc,
                )
            break

    def _publish_reset_routing_hints(self, usage_manager: "UsageManager") -> None:
        """Publish immutable scalar hints consumed synchronously by routing."""
        for state in usage_manager._states.values():
            snapshot = self._reset_credits_cache.get(state.accessor)
            if snapshot is None or snapshot.status != "success" or snapshot.is_stale:
                state.reset_credit_count = 0
                state.reset_credit_expiry_at = None
                continue
            state.reset_credit_count = snapshot.available_count
            state.reset_credit_expiry_at = snapshot.next_expiry_at

    async def fetch_quota_from_api(
        self,
        credential_path: str,
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> CodexQuotaSnapshot:
        """
        Fetch quota information from the Codex /usage API endpoint.

        Args:
            credential_path: Path to credential file or env:// URI
            api_base: Base URL for the Codex API

        Returns:
            CodexQuotaSnapshot with rate limit and credits info
        """
        identifier = _get_credential_identifier(credential_path)
        path_key = f"path:{credential_path}"
        request_guard = {path_key: self._advance_quota_epoch(path_key)}

        try:
            # Get auth headers
            auth_headers = await self.get_auth_header(credential_path)
            account_id = await self.get_account_id(credential_path)
            if not self._quota_guard_current(request_guard):
                raise RuntimeError("quota_refresh_superseded")
            if account_id:
                account_key = f"account:{account_id}"
                request_guard[account_key] = self._advance_quota_epoch(account_key)

            headers = {
                **auth_headers,
                "Content-Type": "application/json",
                "User-Agent": "codex-cli",  # Required by Codex API
            }
            if account_id:
                headers["ChatGPT-Account-Id"] = (
                    account_id  # Exact capitalization from Codex CLI
                )

            # Use the correct Codex API URL
            url = CODEX_USAGE_URL

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()

            # A newer request/publication (including a reset refresh or headers
            # from a credential alias of this pool) owns the current generation.
            # Never relabel this older response with a fresh completion time.
            if not self._quota_guard_current(request_guard):
                raise RuntimeError("quota_refresh_superseded")

            # Parse response
            plan_type = data.get("plan_type")

            # Parse rate_limit section
            rate_limit = data.get("rate_limit")
            primary = None
            secondary = None

            if rate_limit is not None and not isinstance(rate_limit, dict):
                raise ValueError("rate_limit must be an object or null")
            if isinstance(rate_limit, dict):
                primary = _parse_api_window(rate_limit.get("primary_window"))
                secondary = _parse_api_window(rate_limit.get("secondary_window"))

            # Parse credits section
            credits_data = data.get("credits")
            credits = None
            if credits_data:
                credits = CreditsInfo(
                    has_credits=credits_data.get("has_credits", False),
                    unlimited=credits_data.get("unlimited", False),
                    balance=credits_data.get("balance"),
                )

            snapshot = CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=plan_type,
                primary=primary,
                secondary=secondary,
                credits=credits,
                fetched_at=time.time(),
                status="success",
                error=None,
            )

            snapshot = replace(snapshot, account_id=account_id, source="api")
            self._publish_quota_snapshot(snapshot)
            request_guard = {key: self._quota_epochs[key] for key in request_guard}
            await self._apply_quota_to_usage_manager(credential_path, snapshot)

            lib_logger.debug(
                f"Fetched Codex quota for {identifier}: "
                f"primary={primary.remaining_percent:.1f}% remaining"
                if primary
                else f"Fetched Codex quota for {identifier}: no rate limit data"
            )

            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if self._quota_guard_current(request_guard):
                self._quota_errors[credential_path] = error_msg
            lib_logger.warning(
                f"Failed to fetch Codex quota for {identifier}: {error_msg}"
            )
            return CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=None,
                primary=None,
                secondary=None,
                credits=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            if self._quota_guard_current(request_guard):
                self._quota_errors[credential_path] = error_msg
            lib_logger.warning(
                f"Failed to fetch Codex quota for {identifier}: {error_msg}"
            )
            return CodexQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                plan_type=None,
                primary=None,
                secondary=None,
                credits=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

    def update_quota_from_headers(
        self,
        credential_path: str,
        headers: Dict[str, str],
    ) -> Optional[CodexQuotaSnapshot]:
        """
        Update cached quota info from response headers.

        Call this after each API response to keep quota cache up-to-date.
        Also pushes quota data to the UsageManager if available.

        Args:
            credential_path: Credential that made the request
            headers: Response headers dict

        Returns:
            Updated CodexQuotaSnapshot or None if no quota headers present
        """
        snapshot = parse_rate_limit_headers(headers)

        if snapshot.status == "error":
            self._quota_errors[credential_path] = snapshot.error
            lib_logger.warning("Rejected malformed Codex quota headers")
            return None
        if snapshot.status == "no_data":
            if any(key.lower().endswith("-used-percent") for key in headers):
                self._quota_errors[credential_path] = "invalid_quota_headers"
                lib_logger.warning("Rejected invalid Codex quota headers")
            return None

        existing = self._quota_cache.get(credential_path)
        credential = self._credentials_cache.get(credential_path, {})
        account_id = credential.get("account_id") or credential.get("_proxy_metadata", {}).get("account_id")
        snapshot = replace(
            snapshot, credential_path=credential_path,
            identifier=_get_credential_identifier(credential_path),
            plan_type=existing.plan_type if existing else None,
            account_id=account_id, source="headers",
        )
        self._publish_quota_snapshot(snapshot)

        # Log every advertised family, not just `primary`. Logging primary alone
        # was actively misleading: when upstream sends two families the stored
        # state can come from one while the log showed the other, so the log and
        # the payload disagreed with no way to tell why.
        if snapshot.families or snapshot.primary:
            parts: List[str] = []
            for limit_id, (fam_primary, fam_secondary) in sorted(
                snapshot.families.items()
            ):
                for slot, window in (
                    ("primary", fam_primary),
                    ("secondary", fam_secondary),
                ):
                    if window is None:
                        continue
                    parts.append(
                        f"{limit_id}.{slot}="
                        f"{window.remaining_percent:.0f}%rem"
                        f"/{window.window_minutes}min"
                        f"/reset={window.reset_at}"
                    )
            if not parts and snapshot.primary:
                parts.append(
                    f"codex.primary={snapshot.primary.remaining_percent:.0f}%rem"
                    f"/{snapshot.primary.window_minutes}min"
                    f"/reset={snapshot.primary.reset_at}"
                )
            lib_logger.debug(
                "Codex quota from headers (%s): %s",
                snapshot.identifier,
                " ".join(parts),
            )

        # Push quota data to UsageManager if available
        if self._usage_manager:
            self._push_quota_to_usage_manager(credential_path, snapshot)

        return snapshot

    def _push_quota_to_usage_manager(
        self, credential_path: str, snapshot: CodexQuotaSnapshot,
    ) -> None:
        if self._usage_manager is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("Codex quota publication requires a running event loop")
        task = loop.create_task(self._reconcile_header_quota(credential_path, snapshot))
        self._quota_push_tasks.add(task)
        task.add_done_callback(self._quota_push_finished)

    async def _reconcile_header_quota(self, credential_path: str, snapshot: CodexQuotaSnapshot) -> None:
        try:
            await self._apply_quota_to_usage_manager(credential_path, snapshot)
        except Exception:
            self._quota_errors[credential_path] = "quota_manager_reconciliation_failed"
            raise

    def _quota_push_finished(self, task: asyncio.Task) -> None:
        self._quota_push_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            error = task.exception()
            lib_logger.error("Failed to reconcile Codex quota with UsageManager",
                             exc_info=(type(error), error, error.__traceback__))

    async def _apply_quota_to_usage_manager(
        self, credential_path: str, snapshot: CodexQuotaSnapshot,
    ) -> None:
        usage_manager = self._usage_manager
        if usage_manager is None:
            return
        lock = self._quota_push_locks.setdefault(credential_path, asyncio.Lock())
        async with lock:
            # Queued work never replays its old payload over a newer publication.
            snapshot = self._quota_cache.get(credential_path)
            while snapshot is not None:
                provider = getattr(self, "provider_env_name", "codex")
                windows = _classify_quota_windows(snapshot.primary, snapshot.secondary)
                for group, model in (("5h-limit", "_5h_window"), ("weekly-limit", "_weekly_window")):
                    window = windows.get(group)
                    if window is None or window.reset_at is None:
                        await usage_manager.clear_quota_group_state(credential_path, group)
                        continue
                    await usage_manager.update_quota_baseline(
                        accessor=credential_path, model=f"{provider}/{model}",
                        quota_reset_ts=window.reset_at, quota_group=group,
                        force=True, apply_exhaustion=window.is_exhausted,
                        quota_used_percent=window.used_percent,
                        quota_remaining_percent=window.remaining_percent,
                        quota_window_minutes=window.window_minutes, quota_source="codex",
                    )
                    if not window.is_exhausted:
                        await usage_manager.clear_cooldown_if_exists(
                            accessor=credential_path, model_or_group=group,
                        )
                await usage_manager.clear_cooldown_if_exists(
                    accessor=credential_path, model_or_group="codex-global",
                )
                await usage_manager.clear_quota_group_state(
                    credential_path, "codex-global", remove_usage=False,
                )
                latest = self._quota_cache.get(credential_path)
                if latest is snapshot:
                    return
                snapshot = latest

    def get_cached_quota(
        self,
        credential_path: str,
    ) -> Optional[CodexQuotaSnapshot]:
        """
        Get cached quota snapshot for a credential.

        Args:
            credential_path: Credential to look up

        Returns:
            Cached CodexQuotaSnapshot or None if not cached
        """
        return self._quota_cache.get(credential_path)

    # =========================================================================
    # QUOTA INFO AGGREGATION
    # =========================================================================

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> Dict[str, Any]:
        """
        Get quota info for all credentials.

        Args:
            credential_paths: List of credential paths to query
            force_refresh: If True, fetch fresh data; if False, use cache if available
            api_base: Base URL for the Codex API

        Returns:
            {
                "credentials": {
                    "identifier": {
                        "identifier": str,
                        "file_path": str | None,
                        "plan_type": str | None,
                        "status": "success" | "error" | "cached",
                        "error": str | None,
                        "primary": {
                            "remaining_percent": float,
                            "remaining_fraction": float,
                            "used_percent": float,
                            "window_minutes": int | None,
                            "reset_at": int | None,
                            "reset_in_seconds": float | None,
                            "is_exhausted": bool,
                        } | None,
                        "secondary": {...} | None,
                        "credits": {
                            "has_credits": bool,
                            "unlimited": bool,
                            "balance": str | None,
                        } | None,
                        "fetched_at": float,
                        "is_stale": bool,
                    }
                },
                "summary": {
                    "total_credentials": int,
                    "by_plan_type": Dict[str, int],
                    "exhausted_count": int,
                },
                "timestamp": float,
            }
        """
        results = {}
        plan_type_counts: Dict[str, int] = {}
        exhausted_count = 0

        for cred_path in credential_paths:
            identifier = _get_credential_identifier(cred_path)

            # Check cache first unless force_refresh
            cached = self._quota_cache.get(cred_path)
            if not force_refresh and cached and not cached.is_stale:
                snapshot = cached
                status = "cached"
            else:
                snapshot = await self.fetch_quota_from_api(cred_path, api_base)
                status = snapshot.status

            # Count plan types
            if snapshot.plan_type:
                plan_type_counts[snapshot.plan_type] = (
                    plan_type_counts.get(snapshot.plan_type, 0) + 1
                )

            # Check if exhausted
            if snapshot.primary and snapshot.primary.is_exhausted:
                exhausted_count += 1

            # Build result entry
            entry = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "plan_type": snapshot.plan_type,
                "status": status,
                "error": snapshot.error,
                "primary": _window_to_dict(snapshot.primary)
                if snapshot.primary
                else None,
                "secondary": _window_to_dict(snapshot.secondary)
                if snapshot.secondary
                else None,
                "credits": _credits_to_dict(snapshot.credits)
                if snapshot.credits
                else None,
                "fetched_at": snapshot.fetched_at,
                "is_stale": snapshot.is_stale,
            }

            results[identifier] = entry

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "by_plan_type": plan_type_counts,
                "exhausted_count": exhausted_count,
            },
            "timestamp": time.time(),
        }

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return configuration for quota refresh background job.

        Returns:
            Background job config dict
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "codex_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Execute periodic quota refresh for active credentials.

        Called by BackgroundRefresher at the configured interval.
        On first run, fetches baselines for ALL credentials and applies
        exhaustion cooldowns so we don't waste requests on depleted keys.

        Args:
            usage_manager: UsageManager instance (for future baseline storage)
            credentials: List of credential paths for this provider
        """
        if not credentials:
            return

        # On first run, fetch baselines for ALL credentials to detect exhaustion
        if not self._initial_baselines_fetched:
            self._initial_baselines_fetched = True
            try:
                quota_results = await self.fetch_initial_baselines(credentials)
                stored = await self._store_baselines_to_usage_manager(
                    quota_results,
                    usage_manager,
                    force=True,
                    is_initial_fetch=True,
                )
                # Log any exhausted credentials detected on startup
                exhausted = []
                for cred_path, data in quota_results.items():
                    if data.get("status") != "success":
                        continue
                    primary = data.get("primary")
                    secondary = data.get("secondary")
                    if primary and primary.get("is_exhausted"):
                        exhausted.append(
                            f"{_get_credential_identifier(cred_path)} (5h window)"
                        )
                    if secondary and secondary.get("is_exhausted"):
                        exhausted.append(
                            f"{_get_credential_identifier(cred_path)} (weekly)"
                        )
                if exhausted:
                    lib_logger.warning(
                        f"Codex startup: {len(exhausted)} exhausted quota(s) detected, "
                        f"cooldowns applied: {', '.join(exhausted)}"
                    )
                else:
                    lib_logger.info(
                        f"Codex startup: {stored} baselines stored, no exhausted credentials"
                    )
                await self._refresh_and_evaluate_resets(credentials)
                self._publish_reset_routing_hints(usage_manager)
            except Exception as e:
                lib_logger.error(f"Codex startup baseline fetch failed: {e}")
            return

        # Subsequent runs: only refresh credentials that have been used recently
        now = time.time()
        active_credentials = []

        for cred_path in credentials:
            cached = self._quota_cache.get(cred_path)
            # Refresh if cached and was fetched within the last hour
            if cached and (now - cached.fetched_at) < 3600:
                active_credentials.append(cred_path)

        if not active_credentials:
            lib_logger.debug("No active Codex credentials to refresh quota for")
            return

        lib_logger.debug(
            f"Refreshing Codex quota for {len(active_credentials)} active credentials"
        )

        # Fetch quotas with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                snapshot = await self.fetch_quota_from_api(cred_path)
                return cred_path, snapshot

        tasks = [fetch_with_semaphore(cred) for cred in active_credentials]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quota_results: Dict[str, Dict[str, Any]] = {}
        for result in results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Codex quota refresh error: {result}")
                continue

            cred_path, snapshot = result
            if snapshot.status != "success":
                continue

            quota_results[cred_path] = {
                "status": "success",
                "error": None,
                "plan_type": snapshot.plan_type,
                "primary": _window_to_dict(snapshot.primary)
                if snapshot.primary
                else None,
                "secondary": _window_to_dict(snapshot.secondary)
                if snapshot.secondary
                else None,
            }

        stored = await self._store_baselines_to_usage_manager(
            quota_results,
            usage_manager,
            force=True,
        )
        success_count = len(quota_results)
        await self._refresh_and_evaluate_resets(active_credentials)
        self._publish_reset_routing_hints(usage_manager)

        lib_logger.debug(
            f"Codex quota refresh complete: {success_count}/{len(active_credentials)} "
            f"successful, {stored} baselines stored"
        )

    # =========================================================================
    # USAGE MANAGER INTEGRATION
    # =========================================================================

    async def _store_baselines_to_usage_manager(
        self, quota_results: Dict[str, Dict[str, Any]], usage_manager: "UsageManager",
        force: bool = False, is_initial_fetch: bool = False,
    ) -> int:
        """Reconcile the latest publications, never detached older result dictionaries."""
        self.set_usage_manager(usage_manager)
        count = 0
        for path, result in quota_results.items():
            snapshot = self._quota_cache.get(path)
            if result.get("status") != "success" or snapshot is None:
                continue
            await self._apply_quota_to_usage_manager(path, snapshot)
            count += len(_classify_quota_windows(snapshot.primary, snapshot.secondary))
        return count

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
        api_base: str = "https://chatgpt.com/backend-api/codex",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        This matches the interface expected by RotatingClient for quota tracking.

        Args:
            credential_paths: All credential paths to fetch baselines for
            api_base: Base URL for the Codex API

        Returns:
            Dict mapping credential_path -> quota data in format:
            {
                "status": "success" | "error",
                "error": str | None,
                "primary": {
                    "remaining_fraction": float,
                    "remaining_percent": float,
                    "used_percent": float,
                    "reset_at": int | None,
                    ...
                },
                "secondary": {...} | None,
                "plan_type": str | None,
            }
        """
        if not credential_paths:
            return {}

        lib_logger.info(
            f"codex: Fetching initial quota baselines for {len(credential_paths)} credentials..."
        )

        results: Dict[str, Dict[str, Any]] = {}

        # Fetch quotas concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                snapshot = await self.fetch_quota_from_api(cred_path, api_base)
                return cred_path, snapshot

        tasks = [fetch_with_semaphore(cred) for cred in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Codex quota fetch error: {result}")
                continue

            cred_path, snapshot = result

            # Convert snapshot to dict format expected by client.py
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "plan_type": snapshot.plan_type,
                    "primary": {
                        "remaining_fraction": snapshot.primary.remaining_fraction
                        if snapshot.primary
                        else 0,
                        "remaining_percent": snapshot.primary.remaining_percent
                        if snapshot.primary
                        else 0,
                        "used_percent": snapshot.primary.used_percent
                        if snapshot.primary
                        else 100,
                        "reset_at": snapshot.primary.reset_at
                        if snapshot.primary
                        else None,
                        "window_minutes": snapshot.primary.window_minutes
                        if snapshot.primary
                        else None,
                        "is_exhausted": snapshot.primary.is_exhausted
                        if snapshot.primary
                        else True,
                    }
                    if snapshot.primary
                    else None,
                    "secondary": {
                        "remaining_fraction": snapshot.secondary.remaining_fraction,
                        "remaining_percent": snapshot.secondary.remaining_percent,
                        "used_percent": snapshot.secondary.used_percent,
                        "reset_at": snapshot.secondary.reset_at,
                        "window_minutes": snapshot.secondary.window_minutes,
                        "is_exhausted": snapshot.secondary.is_exhausted,
                    }
                    if snapshot.secondary
                    else None,
                    "credits": {
                        "has_credits": snapshot.credits.has_credits,
                        "unlimited": snapshot.credits.unlimited,
                        "balance": snapshot.credits.balance,
                    }
                    if snapshot.credits
                    else None,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        success_count = sum(1 for v in results.values() if v.get("status") == "success")
        lib_logger.info(
            f"codex: Fetched {success_count}/{len(credential_paths)} quota baselines"
        )

        return results
