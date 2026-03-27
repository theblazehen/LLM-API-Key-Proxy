"""
Copilot Quota Tracking Mixin

Provides quota tracking functionality for the Copilot provider by:
1. Fetching account quota status from the /copilot_internal/user endpoint
2. Parsing dynamic quota snapshot buckets
3. Storing quota baselines in UsageManager

Required from provider:
    - self._load_credentials(credential_path) -> Dict[str, Any]
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


COPILOT_USER_ENDPOINT = "https://api.github.com/copilot_internal/user"
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
class CopilotQuotaBucket:
    """Quota bucket info from Copilot user endpoint."""

    key: str
    entitlement: Optional[int]
    remaining: Optional[int]
    percent_remaining: Optional[float]
    unlimited: bool
    overage_count: int
    overage_permitted: bool
    timestamp_utc: Optional[int]

    @property
    def used(self) -> Optional[int]:
        """Get used requests if entitlement and remaining are known."""
        if self.entitlement is None or self.remaining is None:
            return None
        return max(0, self.entitlement - self.remaining)

    @property
    def is_exhausted(self) -> bool:
        """Check if bucket is exhausted."""
        if self.unlimited or self.remaining is None:
            return False
        return self.remaining <= 0


@dataclass
class CopilotQuotaSnapshot:
    """Complete quota snapshot for a Copilot credential."""

    credential_path: str
    identifier: str
    login: Optional[str]
    plan: Optional[str]
    quota_reset_ts: Optional[int]
    buckets: Dict[str, CopilotQuotaBucket]
    fetched_at: float
    status: str
    error: Optional[str]

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS


def _bucket_to_dict(bucket: CopilotQuotaBucket) -> Dict[str, Any]:
    """Convert CopilotQuotaBucket to dict for JSON serialization."""
    return {
        "key": bucket.key,
        "entitlement": bucket.entitlement,
        "remaining": bucket.remaining,
        "used": bucket.used,
        "percent_remaining": bucket.percent_remaining,
        "unlimited": bucket.unlimited,
        "overage_count": bucket.overage_count,
        "overage_permitted": bucket.overage_permitted,
        "timestamp_utc": bucket.timestamp_utc,
        "is_exhausted": bucket.is_exhausted,
    }


class CopilotQuotaTracker:
    """Mixin class providing quota tracking functionality for Copilot provider."""

    _quota_cache: Dict[str, CopilotQuotaSnapshot]
    _quota_refresh_interval: int

    def _init_quota_tracker(self):
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, CopilotQuotaSnapshot] = {}
        self._quota_refresh_interval = DEFAULT_QUOTA_REFRESH_INTERVAL
        self._initial_baselines_fetched: bool = False

    async def _get_github_oauth_token(self, credential_path: str) -> str:
        """Get the long-lived GitHub OAuth token for Copilot quota calls."""
        creds = await self._load_credentials(credential_path)
        github_token = creds.get("refresh_token")
        if not github_token:
            raise ValueError(
                "No GitHub OAuth token (refresh_token) found in credentials"
            )
        return github_token

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> CopilotQuotaSnapshot:
        """Fetch quota information from GitHub Copilot internal user endpoint."""
        identifier = _get_credential_identifier(credential_path)

        try:
            github_token = await self._get_github_oauth_token(credential_path)
            headers = {
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/json",
                "User-Agent": "GitHubCopilotChat/0.32.4",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    COPILOT_USER_ENDPOINT,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

            buckets: Dict[str, CopilotQuotaBucket] = {}
            for key, bucket_data in data.get("quota_snapshots", {}).items():
                if not isinstance(bucket_data, dict):
                    continue

                buckets[key] = CopilotQuotaBucket(
                    key=key,
                    entitlement=bucket_data.get("entitlement"),
                    remaining=bucket_data.get("remaining"),
                    percent_remaining=bucket_data.get("percent_remaining"),
                    unlimited=bool(bucket_data.get("unlimited", False)),
                    overage_count=int(bucket_data.get("overage_count", 0) or 0),
                    overage_permitted=bool(bucket_data.get("overage_permitted", False)),
                    timestamp_utc=_parse_iso_timestamp(
                        bucket_data.get("timestamp_utc")
                    ),
                )

            snapshot = CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login=data.get("login"),
                plan=data.get("copilot_plan"),
                quota_reset_ts=_parse_iso_timestamp(data.get("quota_reset_date_utc")),
                buckets=buckets,
                fetched_at=time.time(),
                status="success",
                error=None,
            )

            self._quota_cache[credential_path] = snapshot
            lib_logger.debug(
                f"Fetched Copilot quota for {identifier}: {len(buckets)} bucket(s)"
            )
            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(
                f"Failed to fetch Copilot quota for {identifier}: {error_msg}"
            )
            snapshot = CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login=None,
                plan=None,
                quota_reset_ts=None,
                buckets={},
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )
            self._quota_cache[credential_path] = snapshot
            return snapshot
        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(
                f"Failed to fetch Copilot quota for {identifier}: {error_msg}"
            )
            snapshot = CopilotQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                login=None,
                plan=None,
                quota_reset_ts=None,
                buckets={},
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
            f"copilot: Fetching initial quota baselines for {len(credential_paths)} credentials..."
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
                lib_logger.warning(f"Copilot quota fetch error: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 2:
                lib_logger.warning(
                    f"Copilot quota fetch returned invalid result: {result}"
                )
                continue

            cred_path, snapshot = result
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "login": snapshot.login,
                    "plan": snapshot.plan,
                    "quota_reset_ts": snapshot.quota_reset_ts,
                    "buckets": {
                        key: _bucket_to_dict(bucket)
                        for key, bucket in snapshot.buckets.items()
                    },
                    "fetched_at": snapshot.fetched_at,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        success_count = sum(1 for v in results.values() if v.get("status") == "success")
        lib_logger.info(
            f"copilot: Fetched {success_count}/{len(credential_paths)} quota baselines"
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
            lib_logger.debug("No recently active Copilot credentials to refresh")
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
                lib_logger.warning(f"Copilot quota refresh error: {result}")
                continue

            if not isinstance(result, tuple) or len(result) != 2:
                lib_logger.warning(
                    f"Copilot quota refresh returned invalid result: {result}"
                )
                continue

            cred_path, snapshot = result
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "login": snapshot.login,
                    "plan": snapshot.plan,
                    "quota_reset_ts": snapshot.quota_reset_ts,
                    "buckets": {
                        key: _bucket_to_dict(bucket)
                        for key, bucket in snapshot.buckets.items()
                    },
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
        """Store Copilot quota baselines into UsageManager."""
        stored_count = 0

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            buckets = quota_data.get("buckets", {})
            reset_ts = quota_data.get("quota_reset_ts")

            # Store baselines for all non-unlimited buckets with entitlement data
            for bucket_key, bucket in buckets.items():
                if not isinstance(bucket, dict):
                    continue

                if bucket.get("unlimited"):
                    continue

                entitlement = bucket.get("entitlement")
                remaining = bucket.get("remaining")
                if entitlement is None or remaining is None:
                    continue

                try:
                    quota_max_requests = int(entitlement)
                    if quota_max_requests <= 0:
                        continue
                    quota_used = max(0, quota_max_requests - int(remaining))
                except (TypeError, ValueError):
                    continue

                apply_exhaustion = bool(
                    is_initial_fetch
                    and not bucket.get("overage_permitted", False)
                    and quota_used >= quota_max_requests
                    and reset_ts
                )

                # Use bucket key as the virtual model/group name
                model_name = f"copilot/_{bucket_key}_window"
                quota_group = bucket_key.replace("_", "-")

                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=model_name,
                        quota_max_requests=quota_max_requests,
                        quota_reset_ts=reset_ts,
                        quota_used=quota_used,
                        quota_group=quota_group,
                        force=force,
                        apply_exhaustion=apply_exhaustion,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Copilot baseline for "
                        f"{_get_credential_identifier(cred_path)} bucket {bucket_key}: {e}"
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

            if any(bucket.is_exhausted for bucket in snapshot.buckets.values()):
                exhausted_count += 1

            results[identifier] = {
                "identifier": identifier,
                "file_path": cred_path if not cred_path.startswith("env://") else None,
                "login": snapshot.login,
                "plan": snapshot.plan,
                "quota_reset_ts": snapshot.quota_reset_ts,
                "status": status,
                "error": snapshot.error,
                "buckets": {
                    key: _bucket_to_dict(bucket)
                    for key, bucket in snapshot.buckets.items()
                },
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
            "name": "copilot_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """Execute periodic quota refresh for Copilot credentials."""
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
                lib_logger.info(f"Copilot startup: {stored} baselines stored")
            except Exception as e:
                lib_logger.error(f"Copilot startup baseline fetch failed: {e}")
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
                    f"Copilot quota refresh: updated {stored} bucket baselines"
                )
        except Exception as e:
            lib_logger.error(f"Copilot quota refresh failed: {e}")
