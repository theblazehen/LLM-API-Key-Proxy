# SPDX-License-Identifier: MIT
"""Codex WebRTC signaling and opaque, credential-affine sideband relay.

Call state belongs to one process: route setup, sideband and DELETE to the same
worker. A process restart ends its calls; it never recreates upstream sessions.
Media and agent execution belong to the client, not this gateway.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.responses import Response
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from proxy_app.proxy_auth import bearer_token, resolve_proxy_identity
from rotator_library.client.codex_live import CodexLiveBackend, LiveBackendError, LiveBinding

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/codex/realtime")
MODEL = "gpt-live-1-codex"
MAX_MESSAGE_BYTES = 1024 * 1024
SIDEBAND_BASE = "wss://api.openai.com/v1/live/"
TICKET_PREFIX = "proxy-ticket."


class _LiveConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # A redirected handshake must never forward the pinned OAuth token to
        # another origin or silently retry an established call's attachment.
        return exc


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class _Call:
    binding: LiveBinding = field(repr=False)
    owner: str
    key_digest: str = field(repr=False)
    ticket_digest: str = field(repr=False)
    active: bool = False
    close_sent: bool = False
    upstream: object | None = field(default=None, repr=False)
    downstream: WebSocket | None = field(default=None, repr=False)
    timer: asyncio.Task | None = field(default=None, repr=False)
    cleanup: asyncio.Task | None = field(default=None, repr=False)


class CodexLiveGateway:
    """Bounded process-local call ownership, with no transcript/event buffering."""

    def __init__(self, backend: CodexLiveBackend, keys: dict[str, str], *,
                 max_calls: int = 64, attach_timeout: float = 60,
                 call_timeout: float = 7200):
        self.backend = backend
        self.keys = keys
        self.max_calls = max_calls
        self.attach_timeout = attach_timeout
        self.call_timeout = call_timeout
        self.calls: dict[str, _Call] = {}
        self.pending = 0
        self.closed = False

    def authenticate(self, authorization: str | None) -> tuple[str, str]:
        if not self.keys:
            raise HTTPException(503, "Codex live requires configured proxy API keys")
        key = bearer_token(authorization)
        identity = resolve_proxy_identity(key, self.keys)
        if identity is None:
            raise HTTPException(401, "Invalid or missing API Key")
        return identity.user, _digest(key)

    def owned(self, call_id: str, owner: str) -> _Call:
        call = self.calls.get(call_id)
        if call is None or call.owner != owner or call.cleanup is not None:
            raise HTTPException(404, "Live call not found")
        return call

    async def create(self, payload: dict, owner: str, key_digest: str,
                     attestation: str | None):
        if self.closed or len(self.calls) + self.pending >= self.max_calls:
            raise HTTPException(503, "Live call capacity exhausted")
        self.pending += 1
        try:
            result = await self.backend.create(payload, attestation)
            if result.binding is None:
                return result, None
            binding = result.binding
            if binding.call_id in self.calls:
                await self.backend.release(binding)
                raise HTTPException(502, "Unexpected live call identity")
            ticket = secrets.token_urlsafe(32)
            call = _Call(binding, owner, key_digest, _digest(ticket))
            if self.closed:
                await self.finish(call)
                raise HTTPException(503, "Live gateway is shutting down")
            self.calls[binding.call_id] = call
            call.timer = asyncio.create_task(self._expire(call, self.attach_timeout))
            return result, ticket
        finally:
            self.pending -= 1

    async def _expire(self, call: _Call, delay: float):
        await asyncio.sleep(delay)
        await self.finish(call)

    async def finish(self, call: _Call):
        if call.cleanup is None:
            call.cleanup = asyncio.create_task(self._finish(call))
        await asyncio.shield(call.cleanup)

    async def _finish(self, call: _Call):
        if call.timer is not None:
            call.timer.cancel()
        try:
            if call.active and call.upstream is None:
                # The in-flight attachment observes cleanup before accepting.
                # A rejected handshake must not be retried invisibly to close.
                logger.warning("Live call ended before sideband attachment; upstream closure unconfirmed")
                return
            # Closing the WebSocket alone isn't a request to end WebRTC media.
            # Explicitly send session.close, including for abandoned offers.
            async with asyncio.timeout(20):
                upstream = call.upstream
                if upstream is None:
                    headers = await self.backend.headers(call.binding)
                    upstream = await _LiveConnect(
                        SIDEBAND_BASE + call.binding.call_id,
                        additional_headers=headers, open_timeout=10, close_timeout=3,
                        max_size=MAX_MESSAGE_BYTES, max_queue=4,
                    )
                try:
                    if not call.close_sent:
                        await upstream.send('{"type":"session.close"}')
                        call.close_sent = True
                finally:
                    await upstream.close()
        except (OSError, TimeoutError, WebSocketException, LiveBackendError):
            logger.warning("Unable to confirm upstream live call closure")
        finally:
            try:
                if call.downstream is not None and call.downstream.application_state == WebSocketState.CONNECTED:
                    async with asyncio.timeout(5):
                        await call.downstream.close(code=1000)
            except (OSError, RuntimeError, TimeoutError, WebSocketDisconnect):
                logger.debug("Live client already disconnected during cleanup")
            finally:
                try:
                    await self.backend.release(call.binding)
                finally:
                    if self.calls.get(call.binding.call_id) is call:
                        self.calls.pop(call.binding.call_id)

    async def close(self):
        self.closed = True
        await asyncio.gather(*(self.finish(call) for call in list(self.calls.values())))


def _gateway(connection) -> CodexLiveGateway:
    return connection.app.state.codex_live


async def _payload(request: Request) -> dict:
    data = bytearray()
    async with asyncio.timeout(15):
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_MESSAGE_BYTES:
                raise HTTPException(413, "Live offer exceeds 1 MiB")
            data.extend(chunk)
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid JSON offer") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("sdp"), str) or not payload["sdp"].strip():
        raise HTTPException(400, "A nonempty sdp offer is required")
    session = payload.get("session")
    if not isinstance(session, dict) or session.get("model") != MODEL:
        raise HTTPException(400, f"session.model must be {MODEL}")
    return payload


@router.post(
    "/calls",
    response_class=Response,
    tags=["Codex live (experimental)"],
    summary="Create a credential-affine Codex WebRTC call",
    responses={
        201: {
            "description": "Raw SDP answer; Location identifies the call, Link its sideband",
            "content": {"application/sdp": {"schema": {"type": "string"}}},
            "headers": {
                "Location": {"schema": {"type": "string"}},
                "Link": {"schema": {"type": "string"}},
                "X-Live-WebSocket-Token": {"schema": {"type": "string"}},
            },
        },
        401: {"description": "Missing or invalid proxy Bearer key, or upstream rejection"},
        503: {"description": "Live authentication or capacity unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {
                "type": "object", "required": ["sdp", "session"],
                "properties": {
                    "sdp": {"type": "string", "minLength": 1},
                    "session": {
                        "type": "object", "required": ["model"],
                        "properties": {"model": {"type": "string", "enum": [MODEL]}},
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            }}},
        },
    },
)
async def create_call(request: Request):
    """Requires Authorization: Bearer <proxy key>. Native session fields pass through.

See DOCUMENTATION.md, Codex live gateway v1, for the WebSocket event protocol.
    """
    gateway = _gateway(request)
    owner, key_digest = gateway.authenticate(request.headers.get("authorization"))
    try:
        payload = await _payload(request)
        result, ticket = await gateway.create(
            payload, owner, key_digest, request.headers.get("x-oai-attestation")
        )
    except TimeoutError:
        raise HTTPException(408, "Live offer timed out") from None
    except LiveBackendError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    headers = dict(result.headers)
    headers["cache-control"] = "no-store"
    if result.binding is not None:
        path = request.scope.get("root_path", "") + router.prefix + "/calls/" + result.binding.call_id
        headers["location"] = path
        headers["link"] = f'<{path}/events>; rel="live-sideband"'
        # Browser WebSocket cannot set Authorization. This single-use ticket is
        # bound to this call and its creating key; it is never an OAuth token.
        headers["x-live-websocket-token"] = ticket
    return Response(result.body, status_code=result.status_code, headers=headers)


@router.delete("/calls/{call_id}", status_code=204, response_class=Response,
               tags=["Codex live (experimental)"], summary="Close an owned live call")
async def close_call(call_id: str, request: Request):
    gateway = _gateway(request)
    owner, _ = gateway.authenticate(request.headers.get("authorization"))
    call = gateway.owned(call_id, owner)
    await gateway.finish(call)
    return Response(status_code=204)


async def _deny(socket: WebSocket, status: int, body: bytes, headers: dict | None = None):
    if "websocket.http.response" in socket.scope.get("extensions", {}):
        await socket.send_denial_response(Response(body, status_code=status, headers=headers))
    else:
        # ASGI servers without denial responses cannot express an HTTP body.
        await socket.close(code=1008, reason=f"Live handshake rejected ({status})")


async def _relay(socket: WebSocket, upstream, call: _Call):
    async def to_upstream():
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("text")
            if data is None:
                data = message.get("bytes")
            if data is None:
                continue
            size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
            if size > MAX_MESSAGE_BYTES:
                await socket.close(code=1009, reason="Live message exceeds 1 MiB")
                return
            # Await every send: no unbounded per-event tasks or queues.
            await upstream.send(data)
            if isinstance(data, str):
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") == "session.close":
                    call.close_sent = True
                    return

    async def to_downstream():
        while True:
            data = await upstream.recv()
            if isinstance(data, str):
                await socket.send_text(data)
            else:
                await socket.send_bytes(data)

    tasks = [asyncio.create_task(to_upstream()), asyncio.create_task(to_downstream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@router.websocket("/calls/{call_id}/events")
async def call_events(call_id: str, socket: WebSocket):
    gateway = _gateway(socket)
    call = None
    claimed = False
    try:
        protocols = socket.scope.get("subprotocols", [])
        tickets = [p[len(TICKET_PREFIX):] for p in protocols if p.startswith(TICKET_PREFIX)]
        if socket.headers.get("authorization"):
            owner, _ = gateway.authenticate(socket.headers.get("authorization"))
            call = gateway.owned(call_id, owner)
        elif len(tickets) == 1:
            candidate = gateway.calls.get(call_id)
            if (candidate is None or not candidate.ticket_digest or
                not hmac.compare_digest(candidate.ticket_digest, _digest(tickets[0])) or
                not any(owner == candidate.owner and hmac.compare_digest(candidate.key_digest, _digest(key))
                        for key, owner in gateway.keys.items())):
                raise HTTPException(401, "Invalid live WebSocket ticket")
            call = gateway.owned(call_id, candidate.owner)
        else:
            gateway.authenticate(None)
        if call.active:
            raise HTTPException(409, "Live sideband is already attached")
        call.active = True
        call.ticket_digest = ""
        claimed = True
        call.timer.cancel()
        call.timer = asyncio.create_task(gateway._expire(call, gateway.call_timeout))
        headers = await gateway.backend.headers(call.binding)
        async with _LiveConnect(
            SIDEBAND_BASE + call.binding.call_id, additional_headers=headers,
            open_timeout=15, close_timeout=5, max_size=MAX_MESSAGE_BYTES,
            max_queue=16, write_limit=32768,
        ) as upstream:
            if call.cleanup is not None:
                await upstream.send('{"type":"session.close"}')
                await _deny(socket, 410, b'{"detail":"Live call closed during attachment"}')
                return
            call.upstream = upstream
            await socket.accept(subprotocol="codex-live" if "codex-live" in protocols else None)
            call.downstream = socket
            if call.cleanup is not None:
                async with asyncio.timeout(5):
                    await socket.close(code=1000)
                return
            try:
                await _relay(socket, upstream, call)
            except ConnectionClosed as exc:
                if socket.application_state == WebSocketState.CONNECTED:
                    code = exc.rcvd.code if exc.rcvd is not None else 1011
                    reason = exc.rcvd.reason if exc.rcvd is not None else "Upstream live connection lost"
                    async with asyncio.timeout(5):
                        await socket.close(code=code, reason=reason)
            except (OSError, TimeoutError):
                if socket.application_state == WebSocketState.CONNECTED:
                    async with asyncio.timeout(5):
                        await socket.close(code=1011, reason="Live sideband connection failed")
            finally:
                await gateway.finish(call)
    except HTTPException as exc:
        await _deny(socket, exc.status_code, json.dumps({"detail": exc.detail}).encode(), {"content-type": "application/json"})
    except LiveBackendError as exc:
        await _deny(socket, exc.status_code, json.dumps({"detail": str(exc)}).encode(), {"content-type": "application/json"})
    except InvalidStatus as exc:
        response = exc.response
        safe = {name: value for name, value in response.headers.raw_items()
                if name.lower() in {"content-type", "retry-after", "x-request-id"}}
        await _deny(socket, response.status_code, bytes(response.body), safe)
    except ConnectionClosed as exc:
        if socket.application_state == WebSocketState.CONNECTED:
            code = exc.rcvd.code if exc.rcvd is not None else 1011
            reason = exc.rcvd.reason if exc.rcvd is not None else "Upstream live connection lost"
            await socket.close(code=code, reason=reason)
    except (OSError, TimeoutError, WebSocketException):
        if socket.application_state == WebSocketState.CONNECTING:
            await _deny(socket, 502, b'{"detail":"Live sideband connection failed"}', {"content-type": "application/json"})
        elif socket.application_state == WebSocketState.CONNECTED:
            await socket.close(code=1011, reason="Live sideband connection failed")
    except WebSocketDisconnect:
        pass
    finally:
        if claimed:
            await gateway.finish(call)
