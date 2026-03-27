"""
Anthropic Quota Tracking Mixin

Provides quota tracking functionality for the Anthropic provider by:
1. Fetching quota status from the /api/oauth/usage endpoint
2. Parsing dynamic window-based usage data
3. Storing quota baselines in UsageManager

Anthropic quota API structure:
- Dynamic time windows (e.g., five_hour, seven_day, seven_day_sonnet)
- Monthly credit limits
- Extra usage flags

Required from provider:
    - self.get_access_token(credential_path) -> str
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_USAGE_BETA = "oauth-2025-04-20"
ANTHROPIC_USAGE_USER_AGENT = "claude-code/2.1.69"
DEFAULT_QUOTA_REFRESH_INTERVAL = 300
QUOTA_STALE_THRESHOLD_SECONDS = 900


def _get_credential_identifier(credential_path: str) -> str:
    """Extract a short identifier from a credential path."""
    if credential_path.startswith("env://"):
        return credential_path
    return Path(credential_path).name


def _parse_iso_timestamp(timestamp: Optional[str]) -> Optional[int]:
    """Parse ISO 8601 timestamp to Unix timestamp."""
    if not timestamp:
        return None

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, AttributeError, TypeError):
        return None


@dataclass
class AnthropicQuotaWindow:
    """Quota window info from Anthropic usage API."""

    key: str
    utilization: float
    remaining_fraction: float
    resets_at: Optional[int]
    is_enabled: bool

    @property
    def is_exhausted(self) -> bool:
        """Check if this window's quota is exhausted."""
        return self.is_enabled and self.utilization >= 1.0

    def seconds_until_reset(self) -> Optional[float]:
        """Calculate seconds until reset, or None if unknown."""
        if self.resets_at is None:
            return None
        return max(0, self.resets_at - time.time())


@dataclass
class AnthropicMonthlyLimit:
    """Monthly credit limit info from Anthropic usage API."""

    monthly_limit: Optional[float]
    used_credits: Optional[float]
    is_enabled: bool

    @property
    def remaining_credits(self) -> Optional[float]:
        """Get remaining credits if monthly limit is known."""
        if self.monthly_limit is None or self.used_credits is None:
            return None
        return max(0.0, self.monthly_limit - self.used_credits)


@dataclass
class AnthropicQuotaSnapshot:
    """Complete quota snapshot for an Anthropic credential."""

    credential_path: str
    identifier: str
    windows: Dict[str, AnthropicQuotaWindow]
    monthly_limit: Optional[AnthropicMonthlyLimit]
    extra_usage_enabled: Optional[bool]
    fetched_at: float
    status: str
    error: Optional[str]

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS


def _window_to_dict(window: AnthropicQuotaWindow) -> Dict[str, Any]:
    """Convert AnthropicQuotaWindow to dict for JSON serialization."""
    return {
        "key": window.key,
        "utilization": window.utilization,
        "remaining_fraction": window.remaining_fraction,
        "resets_at": window.resets_at,
        "reset_in_seconds": window.seconds_until_reset(),
        "is_enabled": window.is_enabled,
        "is_exhausted": window.is_exhausted,
    }


def _monthly_limit_to_dict(monthly_limit: AnthropicMonthlyLimit) -> Dict[str, Any]:
    """Convert AnthropicMonthlyLimit to dict for JSON serialization."""
    return {
        "monthly_limit": monthly_limit.monthly_limit,
        "used_credits": monthly_limit.used_credits,
        "remaining_credits": monthly_limit.remaining_credits,
        "is_enabled": monthly_limit.is_enabled,
    }


class AnthropicQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Anthropic provider.

    This mixin adds the following capabilities:
    - Fetch quota status from the Anthropic OAuth usage endpoint
    - Parse dynamic time-window quota data
    - Store quota baselines in UsageManager
    - Get structured quota info for all credentials
    """

    _quota_cache: Dict[str, AnthropicQuotaSnapshot]
    _quota_refresh_interval: int

    def _init_quota_tracker(self):
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, AnthropicQuotaSnapshot] = {}
        self._quota_refresh_interval = DEFAULT_QUOTA_REFRESH_INTERVAL
        self._initial_baselines_fetched: bool = False

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> AnthropicQuotaSnapshot:
        """Fetch quota information from Anthropic OAuth usage API."""
        identifier = _get_credential_identifier(credential_path)

        try:
            access_token = await self.get_access_token(credential_path)
            headers = {
                "Authorization": f"Bearer {access_token}",
                "anthropic-beta": ANTHROPIC_USAGE_BETA,
                "User-Agent": ANTHROPIC_USAGE_USER_AGENT,
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    ANTHROPIC_USAGE_URL,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

            windows: Dict[str, AnthropicQuotaWindow] = {}
            monthly_limit: Optional[AnthropicMonthlyLimit] = None
            extra_usage_enabled: Optional[bool] = None

            for key, value in data.items():
                if not isinstance(value, dict):
                    continue

                if key == "monthly_limit":
                    monthly_limit = AnthropicMonthlyLimit(
                        monthly_limit=value.get("monthly_limit"),
                        used_credits=value.get("used_credits"),
                        is_enabled=bool(value.get("is_enabled", False)),
                    )
                    continue

                if key == "extra_usage":
                    if "is_enabled" in value:
                        extra_usage_enabled = bool(value.get("is_enabled"))
                    continue

                if "utilization" in value and "resets_at" in value:
                    utilization = value.get("utilization", 0.0)
                    try:
                        utilization_float = float(utilization)
                    except (TypeError, ValueError):
                        utilization_float = 0.0

                    utilization_float = max(0.0, min(1.0, utilization_float))

                    windows[key] = AnthropicQuotaWindow(
                        key=key,
                        utilization=utilization_float,
                        remaining_fraction=max(0.0, min(1.0, 1.0 - utilization_float)),
                        resets_at=_parse_iso_timestamp(value.get("resets_at")),
                        is_enabled=bool(value.get("is_enabled", False)),
                    )

            snapshot = AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                windows=windows,
                monthly_limit=monthly_limit,
                extra_usage_enabled=extra_usage_enabled,
                fetched_at=time.time(),
                status="success",
                error=None,
            )

            self._quota_cache[credential_path] = snapshot
            lib_logger.debug(
                f"Fetched Anthropic quota for {identifier}: {len(windows)} window(s)"
            )
            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(
                f"Failed to fetch Anthropic quota for {identifier}: {error_msg}"
            )
            snapshot = AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                windows={},
                monthly_limit=None,
                extra_usage_enabled=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )
            self._quota_cache[credential_path] = snapshot
            return snapshot
        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(
                f"Failed to fetch Anthropic quota for {identifier}: {error_msg}"
            )
            snapshot = AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                windows={},
                monthly_limit=None,
                extra_usage_enabled=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )
            self._quota_cache[credential_path] = snapshot
            return snapshot

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch quota baselines for all credentials."""
        if not credential_paths:
            return {}

        lib_logger.info(
            f"anthropic: Fetching initial quota baselines for {len(credential_paths)} credentials..."
        )

        results: Dict[str, Dict[str, Any]] = {}
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                snapshot = await self.fetch_quota_from_api(cred_path)
                return cred_path, snapshot

        fetch_results: List[Any] = await asyncio.gather(
            *[fetch_with_semaphore(cred) for cred in credential_paths],
            return_exceptions=True,
        )

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Anthropic quota fetch error: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 2:
                lib_logger.warning(
                    f"Anthropic quota fetch returned invalid result: {result}"
                )
                continue

            cred_path, snapshot = result
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "windows": {
                        key: {
                            "utilization": window.utilization,
                            "remaining_fraction": window.remaining_fraction,
                            "resets_at": window.resets_at,
                            "is_enabled": window.is_enabled,
                            "is_exhausted": window.is_exhausted,
                        }
                        for key, window in snapshot.windows.items()
                    },
                    "monthly_limit": _monthly_limit_to_dict(snapshot.monthly_limit)
                    if snapshot.monthly_limit
                    else None,
                    "extra_usage_enabled": snapshot.extra_usage_enabled,
                    "fetched_at": snapshot.fetched_at,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        success_count = sum(1 for v in results.values() if v.get("status") == "success")
        lib_logger.info(
            f"anthropic: Fetched {success_count}/{len(credential_paths)} quota baselines"
        )
        return results

    async def refresh_active_quota_baselines(
        self,
        credential_paths: List[str],
        usage_data: Dict[str, Any],
        interval_seconds: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Refresh quota baselines for recently active credentials."""
        if interval_seconds is None:
            interval_seconds = self._quota_refresh_interval

        now = time.time()
        active_credentials = []
        for cred_path in credential_paths:
            cred_usage = usage_data.get(cred_path, {})
            last_used = cred_usage.get("last_used_ts", 0)
            if now - last_used < interval_seconds:
                active_credentials.append(cred_path)

        if not active_credentials:
            lib_logger.debug("No recently active Anthropic credentials to refresh")
            return {}

        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                snapshot = await self.fetch_quota_from_api(cred_path)
                return cred_path, snapshot

        fetch_results: List[Any] = await asyncio.gather(
            *[fetch_with_semaphore(cred) for cred in active_credentials],
            return_exceptions=True,
        )

        results: Dict[str, Dict[str, Any]] = {}
        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Anthropic quota refresh error: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 2:
                lib_logger.warning(
                    f"Anthropic quota refresh returned invalid result: {result}"
                )
                continue

            cred_path, snapshot = result
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "windows": {
                        key: {
                            "utilization": window.utilization,
                            "remaining_fraction": window.remaining_fraction,
                            "resets_at": window.resets_at,
                            "is_enabled": window.is_enabled,
                            "is_exhausted": window.is_exhausted,
                        }
                        for key, window in snapshot.windows.items()
                    },
                    "monthly_limit": _monthly_limit_to_dict(snapshot.monthly_limit)
                    if snapshot.monthly_limit
                    else None,
                    "extra_usage_enabled": snapshot.extra_usage_enabled,
                    "fetched_at": snapshot.fetched_at,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        return results

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
        force: bool = False,
        is_initial_fetch: bool = False,
    ) -> int:
        """Store Anthropic quota baselines into UsageManager."""
        stored_count = 0
        window_mappings = {
            "five_hour": ("anthropic/_5h_window", "5h-limit"),
            "seven_day": ("anthropic/_7d_window", "7d-limit"),
        }

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            windows = quota_data.get("windows", {})
            for window_key, (model_name, quota_group) in window_mappings.items():
                window = windows.get(window_key)
                if not window or not window.get("is_enabled"):
                    continue

                utilization = window.get("utilization", 0.0)
                try:
                    quota_used = int(round(float(utilization) * 100))
                except (TypeError, ValueError):
                    quota_used = 0

                reset_ts = window.get("resets_at")
                apply_exhaustion = bool(
                    window.get("is_exhausted") and is_initial_fetch and reset_ts
                )

                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=model_name,
                        quota_max_requests=100,
                        quota_reset_ts=reset_ts,
                        quota_used=quota_used,
                        quota_group=quota_group,
                        force=force,
                        apply_exhaustion=apply_exhaustion,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Anthropic baseline for {_get_credential_identifier(cred_path)} "
                        f"window {window_key}: {e}"
                    )

        return stored_count

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get quota info for all credentials."""
        results = {}
        exhausted_count = 0

        for cred_path in credential_paths:
            identifier = _get_credential_identifier(cred_path)
            cached = self._quota_cache.get(cred_path)
            if not force_refresh and cached and not cached.is_stale:
                snapshot = cached
                status = "cached"
            else:
                snapshot = await self.fetch_quota_from_api(cred_path)
                status = snapshot.status

            exhausted_windows = [
                window for window in snapshot.windows.values() if window.is_exhausted
            ]
            if exhausted_windows:
                exhausted_count += 1

            results[identifier] = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "status": status,
                "error": snapshot.error,
                "windows": {
                    key: _window_to_dict(window)
                    for key, window in snapshot.windows.items()
                },
                "monthly_limit": _monthly_limit_to_dict(snapshot.monthly_limit)
                if snapshot.monthly_limit
                else None,
                "extra_usage_enabled": snapshot.extra_usage_enabled,
                "fetched_at": snapshot.fetched_at,
                "is_stale": snapshot.is_stale,
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "exhausted_count": exhausted_count,
            },
            "timestamp": time.time(),
        }

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Return configuration for quota refresh background job."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "anthropic_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """Execute periodic quota refresh for Anthropic credentials."""
        if not credentials:
            return

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
                lib_logger.info(f"Anthropic startup: {stored} baselines stored")
            except Exception as e:
                lib_logger.error(f"Anthropic startup baseline fetch failed: {e}")
            return

        usage_data = await usage_manager.get_usage_snapshot()
        quota_results = await self.refresh_active_quota_baselines(
            credentials,
            usage_data,
        )
        if not quota_results:
            return

        try:
            stored = await self._store_baselines_to_usage_manager(
                quota_results,
                usage_manager,
                force=False,
                is_initial_fetch=False,
            )
            if stored > 0:
                lib_logger.debug(
                    f"Anthropic quota refresh: updated {stored} window baselines"
                )
        except Exception as e:
            lib_logger.error(f"Anthropic quota refresh failed: {e}")
