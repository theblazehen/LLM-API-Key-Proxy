import asyncio
import time
import os
import logging
from typing import Dict, Set

lib_logger = logging.getLogger("rotator_library")


class CooldownManager:
    """
    Manages global cooldown periods for API providers to handle IP-based rate limiting.
    This ensures that once a 429 error is received for a provider, all subsequent
    requests to that provider are paused for a specified duration.

    Cooldowns can be disabled per-provider via environment variables:
        DISABLE_COOLDOWN_<PROVIDER>=true

    Example: DISABLE_COOLDOWN_ANTIGRAVITY=true
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._disabled_providers: Set[str] = self._load_disabled_providers()

    def _load_disabled_providers(self) -> Set[str]:
        disabled = set()
        for key, value in os.environ.items():
            if key.startswith("DISABLE_COOLDOWN_") and value.lower() in (
                "true",
                "1",
                "yes",
            ):
                provider = key.replace("DISABLE_COOLDOWN_", "").lower()
                disabled.add(provider)
                lib_logger.info(f"Cooldowns disabled for provider: {provider}")
        return disabled

    async def is_cooling_down(self, provider: str) -> bool:
        if provider in self._disabled_providers:
            return False
        async with self._lock:
            return (
                provider in self._cooldowns and time.time() < self._cooldowns[provider]
            )

    async def start_cooldown(self, provider: str, duration: int):
        if provider in self._disabled_providers:
            lib_logger.debug(
                f"Cooldown skipped for {provider} (disabled via DISABLE_COOLDOWN_{provider.upper()}=true)"
            )
            return
        async with self._lock:
            self._cooldowns[provider] = time.time() + duration

    async def get_cooldown_remaining(self, provider: str) -> float:
        if provider in self._disabled_providers:
            return 0
        async with self._lock:
            if provider in self._cooldowns:
                remaining = self._cooldowns[provider] - time.time()
                return max(0, remaining)
            return 0
