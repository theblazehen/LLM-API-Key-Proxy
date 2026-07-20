# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Sequential rotation strategy.

Uses one credential until exhausted, then moves to the next.
Good for providers that benefit from request caching.
"""

import logging
import time
from typing import Dict, List, Optional

from ...types import CredentialState, SelectionContext, RotationMode
from ....error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")

_CODEX_QUOTA_GROUP = "codex-global"
_WEEKLY_QUOTA_GROUP = "weekly-limit"
_EFFECTIVE_DEADLINE_LEAD_HOURS = 24.0
_MIN_PRESSURE_HOURS = 0.25
_PRESSURE_HYSTERESIS = 1.25
_MATERIAL_RESET_CHANGE_SECONDS = 6 * 60 * 60


class SequentialStrategy:
    """
    Sequential credential rotation strategy.

    Sticks to one credential until it's exhausted (rate limited,
    quota exceeded, etc.), then moves to the next in priority order.

    This is useful for providers where repeated requests to the same
    credential benefit from caching (e.g., context caching in LLMs).
    """

    def __init__(self, fallback_multiplier: int = 1):
        """
        Initialize sequential strategy.

        Args:
            fallback_multiplier: Default concurrent slots per priority
                when not explicitly configured
        """
        self.fallback_multiplier = fallback_multiplier
        # Track current "sticky" credential per (provider, model_group)
        self._current: Dict[tuple, str] = {}
        self._codex_weekly_generation: Dict[tuple, float] = {}

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def mode(self) -> RotationMode:
        return RotationMode.SEQUENTIAL

    def select(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
    ) -> Optional[str]:
        """
        Select a credential using sequential/sticky selection.

        Prefers the currently active credential if it's still available.
        Otherwise, selects the first available by priority.

        Args:
            context: Selection context with candidates and usage info
            states: Dict of stable_id -> CredentialState

        Returns:
            Selected stable_id, or None if no candidates
        """
        if not context.candidates:
            return None

        if len(context.candidates) == 1:
            return context.candidates[0]

        key = (context.provider, context.quota_group or context.model)

        if context.provider == "codex" and context.quota_group == _CODEX_QUOTA_GROUP:
            selected = self._select_codex_by_weekly_deadline(context, states, key)
            if selected is not None:
                return selected

        # Check if current sticky credential is still available
        current = self._current.get(key)
        if current and current in context.candidates:
            # Check if a higher-priority credential has become available
            # (e.g., primary credential's cooldown expired)
            current_priority = context.priorities.get(current, 999)
            best_available_priority = min(
                context.priorities.get(c, 999) for c in context.candidates
            )
            if best_available_priority < current_priority:
                # Higher-priority credential is back — switch to it
                selected = self._select_by_priority(
                    context.candidates,
                    context.priorities,
                    context.usage_counts,
                    states,
                )
                if selected and selected != current:
                    self._current[key] = selected
                    masked = (
                        mask_credential(states[selected].accessor, style="full")
                        if selected in states
                        else mask_credential(selected, style="full")
                    )
                    lib_logger.info(
                        f"Sequential: returning to higher-priority credential "
                        f"{masked} for {key}"
                    )
                    return selected
            return current

        # Current not available - select new one by tier -> usage -> recency
        selected = self._select_by_priority(
            context.candidates,
            context.priorities,
            context.usage_counts,
            states,
        )

        # Make it sticky
        if selected:
            self._current[key] = selected
            masked = (
                mask_credential(states[selected].accessor, style="full")
                if selected in states
                else mask_credential(selected, style="full")
            )
            lib_logger.debug(f"Sequential: switched to credential {masked} for {key}")

        return selected

    def _select_codex_by_weekly_deadline(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
        key: tuple,
    ) -> Optional[str]:
        """Select using complete, current Codex weekly quota snapshots.

        Returning ``None`` delegates to the unchanged sequential algorithm;
        known and unknown weekly state must not be mixed.
        """
        now = time.time()
        snapshots = {}
        for candidate in context.candidates:
            group = states[candidate].group_usage.get(_WEEKLY_QUOTA_GROUP)
            if not group:
                return None

            window = group.windows.get("daily")
            if window is None:
                quota_windows = [
                    item
                    for item in group.windows.values()
                    if item.quota_source == "codex"
                ]
                if len(quota_windows) != 1:
                    return None
                window = quota_windows[0]

            remaining = window.remaining_percent
            reset_at = window.reset_at
            if (
                window.quota_source != "codex"
                or remaining is None
                or not 0.0 <= remaining <= 100.0
                or reset_at is None
                or reset_at <= now
            ):
                return None

            effective_hours = (reset_at - now) / 3600.0 - _EFFECTIVE_DEADLINE_LEAD_HOURS
            auto_redeem_at = states[candidate].reset_auto_redeem_at
            if (
                states[candidate].reset_credit_count > 0
                and auto_redeem_at is not None
                and auto_redeem_at > now
            ):
                # Existing capacity will also be overwritten when an earned
                # credit auto-redeems, so drain toward the earlier deadline.
                effective_hours = min(
                    effective_hours,
                    (auto_redeem_at - now) / 3600.0
                    - _EFFECTIVE_DEADLINE_LEAD_HOURS,
                )
            snapshots[candidate] = (
                remaining / max(effective_hours, _MIN_PRESSURE_HOURS),
                effective_hours <= 0.0,
                reset_at,
            )

        current = self._current.get(key)
        most_urgent = max(context.candidates, key=lambda c: snapshots[c][0])
        selected = most_urgent
        if current in context.candidates:
            current_pressure, _, current_reset = snapshots[current]
            other_pressure, other_past_deadline, _ = snapshots[most_urgent]
            prior_reset = self._codex_weekly_generation.get(key)
            generation_changed = (
                prior_reset is not None
                and abs(current_reset - prior_reset) >= _MATERIAL_RESET_CHANGE_SECONDS
            )
            if (
                most_urgent == current
                or (
                    not generation_changed
                    and not other_past_deadline
                    and (
                        other_pressure == current_pressure
                        or other_pressure < current_pressure * _PRESSURE_HYSTERESIS
                    )
                )
            ):
                selected = current

        self._current[key] = selected
        self._codex_weekly_generation[key] = snapshots[selected][2]
        return selected

    def mark_exhausted(self, provider: str, model_or_group: str) -> None:
        """
        Mark current credential as exhausted, forcing rotation.

        Args:
            provider: Provider name
            model_or_group: Model or quota group
        """
        key = (provider, model_or_group)
        if key in self._current:
            old = self._current[key]
            del self._current[key]
            self._codex_weekly_generation.pop(key, None)
            lib_logger.debug(
                f"Sequential: marked {mask_credential(old, style='full')} exhausted for {key}"
            )

    def get_current(self, provider: str, model_or_group: str) -> Optional[str]:
        """
        Get the currently sticky credential.

        Args:
            provider: Provider name
            model_or_group: Model or quota group

        Returns:
            Current sticky credential stable_id, or None
        """
        key = (provider, model_or_group)
        return self._current.get(key)

    def _select_by_priority(
        self,
        candidates: List[str],
        priorities: Dict[str, int],
        usage_counts: Optional[Dict[str, int]] = None,
        states: Optional[Dict[str, CredentialState]] = None,
    ) -> Optional[str]:
        """
        Select credential by: tier (priority) -> usage (highest) -> recency (most recent).

        Sequential mode prefers most-used credentials within the window to maximize
        cache hits. When selecting a new sticky credential:
        1. Highest tier (lowest priority number) first
        2. Within same tier, prefer highest usage count
        3. Within same usage, prefer most recently used

        Args:
            candidates: List of available credential stable_ids
            priorities: Dict of stable_id -> priority (lower = higher tier)
            usage_counts: Dict of stable_id -> request count for relevant window
            states: Dict of stable_id -> CredentialState for recency lookup

        Returns:
            Selected stable_id, or None if no candidates
        """
        if not candidates:
            return None

        usage_counts = usage_counts or {}
        states = states or {}

        def sort_key(c: str):
            # 1. Priority/tier (lower number = higher tier = preferred)
            priority = priorities.get(c, 999)

            # 2. Usage count (higher = preferred, so negate for ascending sort)
            usage = -(usage_counts.get(c, 0))

            # 3. Recency (more recent = preferred, so negate for ascending sort)
            state = states.get(c)
            last_used = -(state.totals.last_used_at or 0) if state else 0

            return (priority, usage, last_used)

        sorted_candidates = sorted(candidates, key=sort_key)
        return sorted_candidates[0]

    def clear_sticky(self, provider: Optional[str] = None) -> None:
        """
        Clear sticky credential state.

        Args:
            provider: If specified, only clear for this provider
        """
        if provider:
            keys_to_remove = [k for k in self._current if k[0] == provider]
            for key in keys_to_remove:
                del self._current[key]
                self._codex_weekly_generation.pop(key, None)
        else:
            self._current.clear()
            self._codex_weekly_generation.clear()
