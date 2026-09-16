# SPDX-License-Identifier: LGPL-3.0-only
"""Public OpenAI Realtime transport using the existing Codex OAuth pool."""

from __future__ import annotations

import httpx

from ..error_handler import classify_error
from .codex_live import CodexLiveBackend, LiveBinding

API_BASE = "https://api.openai.com/v1/realtime"
SOCKET_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2.1"


class RealtimeBackend(CodexLiveBackend):
    # Public Realtime remains usable when the account's Codex text quota is
    # exhausted. Keep shared concurrency/global failures, not text-window limits.
    _quota_group = "openai-realtime"

    @staticmethod
    def _headers(binding: LiveBinding, token: str) -> dict[str, str]:
        # Public Realtime accepts Codex OAuth directly. Quicksilver headers belong
        # only to the separate Codex GPT-Live endpoint, not this protocol.
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": binding.account_id,
        }

    async def request(
        self, binding: LiveBinding, path: str, **kwargs
    ) -> httpx.Response:
        response = await self._client.http_client.post(
            API_BASE + path,
            headers=await self.headers(binding),
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
            **kwargs,
        )
        if response.is_success:
            self.succeeded(binding, dict(response.headers))
        elif path != f"/calls/{binding.call_id}/hangup":
            error = httpx.HTTPStatusError(
                "Realtime setup rejected",
                request=httpx.Request("POST", API_BASE + path),
                response=httpx.Response(response.status_code, headers=response.headers),
            )
            binding._lease.mark_failure(classify_error(error, provider="codex"))
        return response

    def succeeded(self, binding: LiveBinding, headers: dict | None = None) -> None:
        binding._lease.mark_success(response_headers=headers or {})

    async def create_call(
        self, binding: LiveBinding, sdp: str, session: str
    ) -> httpx.Response:
        return await self.request(
            binding,
            "/calls",
            files={
                "sdp": (None, sdp, "application/sdp"),
                "session": (None, session, "application/json"),
            },
        )

    async def client_secret(
        self, binding: LiveBinding, payload: dict
    ) -> httpx.Response:
        return await self.request(binding, "/client_secrets", json=payload)

    async def hangup(self, binding: LiveBinding) -> httpx.Response:
        return await self.request(binding, f"/calls/{binding.call_id}/hangup")
