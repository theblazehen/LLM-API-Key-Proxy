# SPDX-License-Identifier: LGPL-3.0-only
"""OAuth-only Codex live setup, with a credential lease lasting for the call."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from ..error_handler import NoAvailableKeysError, classify_error
from ..providers import codex_provider
from ..usage import CredentialContext

if TYPE_CHECKING:
    from .rotating_client import RotatingClient

_SETUP_TIMEOUT = 60.0
_AUTH_TIMEOUT = 30.0
_SAFE_HEADERS = frozenset({"content-type", "retry-after", "x-request-id", "openai-request-id"})
_CALL_ID = re.compile(r"rtc_[A-Za-z0-9_-]+", re.ASCII)


class LiveBackendError(Exception):
    """A local failure, with a client-safe message and HTTP status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(eq=False)
class LiveBinding:
    call_id: str
    credential: str = field(repr=False)
    account_id: str = field(repr=False)
    session_id: str
    thread_id: str
    x_session_id: str
    attestation: str | None = field(default=None, repr=False)
    _lease: CredentialContext | None = field(default=None, repr=False)
    _release_task: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class LiveCallResult:
    status_code: int
    body: bytes
    headers: dict[str, str]
    binding: LiveBinding | None = None


class CodexLiveBackend:
    def __init__(self, client: RotatingClient):
        self._client = client

    def _provider(self):
        if codex_provider.USE_OPENAI_API:
            raise LiveBackendError(503, "Codex live requires the ChatGPT OAuth backend")
        provider = self._client._get_provider_instance("codex")
        if provider is None:
            raise LiveBackendError(503, "Codex OAuth is not configured")
        return provider

    async def _identity(self, credential: str) -> tuple[str, str]:
        provider = self._provider()
        try:
            async with asyncio.timeout(_AUTH_TIMEOUT):
                return await provider.get_live_oauth_identity(credential)
        except (TimeoutError, httpx.TimeoutException):
            raise LiveBackendError(504, "Codex OAuth refresh timed out") from None
        except Exception:
            raise LiveBackendError(503, "Codex OAuth authentication is unavailable") from None

    @staticmethod
    def _headers(binding: LiveBinding, token: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": binding.account_id,
            "OpenAI-Alpha": "quicksilver=v2",
            "originator": "Codex Desktop",
            "User-Agent": "Codex Desktop/0.153.0",
            "version": "0.153.0",
            "session-id": binding.session_id,
            "thread-id": binding.thread_id,
            "x-session-id": binding.x_session_id,
        }
        if binding.attestation is not None:
            headers["x-oai-attestation"] = binding.attestation
        return headers

    async def headers(self, binding: LiveBinding) -> dict[str, str]:
        """Refresh only this call's credential; never rotate accounts."""
        if binding._release_task is not None or binding._lease is None:
            raise LiveBackendError(410, "Codex live call has been released")
        token, account = await self._identity(binding.credential)
        if account != binding.account_id:
            raise LiveBackendError(409, "Codex live account identity changed")
        if binding._release_task is not None:
            raise LiveBackendError(410, "Codex live call has been released")
        return self._headers(binding, token)

    async def release(self, binding: LiveBinding) -> None:
        """Release exactly once; cleanup survives caller cancellation."""
        if binding._release_task is None:
            if binding._lease is None:
                return
            binding._release_task = asyncio.create_task(
                binding._lease.__aexit__(None, None, None)
            )
        await asyncio.shield(binding._release_task)

    async def create(self, payload: dict, attestation: str | None = None) -> LiveCallResult:
        """Send one setup POST, preserving its response and leasing its account."""
        binding = None
        retained = False
        try:
            async with asyncio.timeout(_SETUP_TIMEOUT):
                provider = self._provider()
                session = payload.get("session")
                model = session.get("model") if isinstance(session, dict) else None
                if not isinstance(model, str) or not model:
                    raise LiveBackendError(400, "Codex live session.model is required")
                if not isinstance(payload.get("sdp"), str) or not payload["sdp"]:
                    raise LiveBackendError(400, "Codex live sdp is required")
                policy_model = model if model.startswith("codex/") else f"codex/{model}"
                if not self._client._model_resolver.is_model_allowed(policy_model, "codex"):
                    raise LiveBackendError(403, "Codex live model is disabled")
                manager = self._client._usage_managers.get("codex")
                if manager is None:
                    raise LiveBackendError(503, "Codex credential usage manager is unavailable")
                all_credentials = self._client.all_credentials.get("codex", [])
                filters = self._client._credential_filter.filter_by_tier(
                    all_credentials, policy_model, "codex"
                )
                if not manager.initialized:
                    await manager.initialize(
                        all_credentials, priorities=filters.priorities, tiers=filters.tier_names
                    )
                oauth = set(self._client.oauth_credentials.get("codex", []))
                candidates = [
                    cred for cred in filters.all_usable
                    if cred in oauth and provider.is_credential_available(cred)
                ]
                if not candidates:
                    raise LiveBackendError(503, "No compatible Codex OAuth credentials")
                remaining = await self._client.cooldown_manager.get_remaining_cooldown("codex")
                if remaining > 0:
                    if remaining >= _SETUP_TIMEOUT:
                        raise LiveBackendError(503, "Codex provider is cooling down")
                    await asyncio.sleep(remaining)
                lease = await manager.acquire_credential(
                    model=policy_model,
                    quota_group=manager.get_model_quota_group(policy_model),
                    candidates=candidates,
                    priorities=filters.priorities,
                    deadline=time.time() + _SETUP_TIMEOUT,
                )
                session_id = str(uuid.uuid4())
                binding = LiveBinding(
                    call_id="", credential=lease.credential, account_id="",
                    session_id=session_id, thread_id=session_id,
                    x_session_id=str(uuid.uuid4()), attestation=attestation, _lease=lease,
                )
                # Until a valid setup response is accepted, cleanup records failure.
                lease.mark_failure(classify_error(LiveBackendError(502, "Codex live setup failed")))
                token, binding.account_id = await self._identity(lease.credential)
                response = await self._client.http_client.post(
                    f"{codex_provider.CODEX_API_BASE.rstrip('/')}/realtime/calls",
                    params={"intent": "quicksilver", "architecture": "avas"},
                    json=payload, headers=self._headers(binding, token),
                    timeout=httpx.Timeout(30.0), follow_redirects=False,
                )
                response_headers = dict(response.headers)
                provider.update_quota_from_headers(lease.credential, response_headers)
                safe_headers = {
                    key: value for key, value in response_headers.items() if key in _SAFE_HEADERS
                }
                if not response.is_success:
                    # Classify without retaining auth headers, credentials, or response body.
                    safe_response = httpx.Response(
                        response.status_code, headers=safe_headers,
                        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/realtime/calls"),
                    )
                    lease.mark_failure(classify_error(httpx.HTTPStatusError(
                        "Codex live setup rejected", request=safe_response.request, response=safe_response
                    ), provider="codex"))
                    return LiveCallResult(response.status_code, response.content, safe_headers)
                try:
                    segments = urlsplit(response.headers.get("location", "")).path.split("/")
                    ids = [segment for segment in segments if _CALL_ID.fullmatch(segment)]
                except ValueError:
                    ids = []
                if len(ids) != 1 or not response.content.startswith((b"v=0\r\n", b"v=0\n")):
                    raise LiveBackendError(502, "Codex live returned an invalid setup response")
                binding.call_id = ids[0]
                lease.mark_success(response_headers=response_headers)
                result = LiveCallResult(response.status_code, response.content, safe_headers, binding)
            retained = True
            return result
        except LiveBackendError:
            raise
        except NoAvailableKeysError:
            raise LiveBackendError(503, "No Codex OAuth credential is available") from None
        except (TimeoutError, httpx.TimeoutException):
            raise LiveBackendError(504, "Codex live setup timed out") from None
        except httpx.RequestError:
            raise LiveBackendError(502, "Codex live upstream connection failed") from None
        except Exception:
            raise LiveBackendError(502, "Codex live setup failed") from None
        finally:
            if binding is not None and not retained:
                await self.release(binding)
