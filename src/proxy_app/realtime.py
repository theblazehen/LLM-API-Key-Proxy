# SPDX-License-Identifier: MIT
"""Standard Realtime sessions, with one OAuth lease per proxied conversation.

WebRTC media stays direct. Calls and their sidebands must reach the same process.
Client secrets are native upstream credentials for clients connecting directly;
the proxy can account for their creation, not the lifetime of those direct calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import Response
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from proxy_app.codex_live import _LiveConnect, _deny
from proxy_app.proxy_auth import bearer_token, resolve_proxy_identity
from rotator_library.client.codex_live import LiveBackendError, LiveBinding
from rotator_library.client.realtime import DEFAULT_MODEL, SOCKET_URL, RealtimeBackend

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/realtime", tags=["Realtime"])
MAX_BODY = 1024 * 1024
SAFE_HEADERS = {"content-type", "retry-after", "x-request-id", "openai-request-id"}
CALL_ID = re.compile(
    r"(?:rtc_[A-Za-z0-9_-]+|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\Z"
)


@dataclass(eq=False)
class _Session:
    id: str
    owner: str
    binding: LiveBinding = field(repr=False)
    operation: asyncio.Task | None = None
    upstream: object | None = field(default=None, repr=False)
    downstream: WebSocket | None = field(default=None, repr=False)
    active: bool = False
    timer: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None


class RealtimeGateway:
    def __init__(
        self,
        backend: RealtimeBackend,
        keys: dict[str, str],
        *,
        max_sessions: int = 64,
        session_timeout: float = 3600,
    ):
        self.backend, self.keys = backend, keys
        self.max_sessions, self.session_timeout = max_sessions, session_timeout
        self.sessions: dict[str, _Session] = {}
        self.pending = 0
        self.closed = False

    def authenticate(self, authorization: str | None) -> str:
        if not self.keys:
            raise HTTPException(503, "Realtime requires configured proxy API keys")
        identity = resolve_proxy_identity(bearer_token(authorization), self.keys)
        if identity is None:
            raise HTTPException(401, "Invalid or missing API Key")
        return identity.user

    async def acquire(self, model: str, owner: str) -> _Session:
        if self.closed or len(self.sessions) + self.pending >= self.max_sessions:
            raise HTTPException(503, "Realtime session capacity exhausted")
        self.pending += 1
        try:
            binding, _ = await self.backend.acquire(model)
            if self.closed:
                await self.backend.release(binding)
                raise HTTPException(503, "Realtime gateway is shutting down")
            session = _Session(str(uuid.uuid4()), owner, binding)
            self.sessions[session.id] = session
            session.timer = asyncio.create_task(self._expire(session))
            return session
        finally:
            self.pending -= 1

    def owned(self, call_id: str, owner: str) -> _Session:
        session = self.sessions.get(call_id)
        if session is None or session.owner != owner or session.cleanup is not None:
            raise HTTPException(404, "Realtime call not found")
        return session

    async def _expire(self, session: _Session):
        await asyncio.sleep(self.session_timeout)
        await self.finish(session)

    async def create_call(
        self, session: _Session, sdp: str, config: dict
    ) -> httpx.Response:
        response = await self.backend.create_call(
            session.binding, sdp, json.dumps(config)
        )
        if not response.is_success:
            return response
        location = urlsplit(response.headers.get("location", ""))
        call_id = location.path.rstrip("/").rsplit("/", 1)[-1]
        if not CALL_ID.fullmatch(call_id) or call_id in self.sessions:
            raise LiveBackendError(502, "Invalid upstream Realtime call identity")
        self.sessions.pop(session.id)
        session.id = session.binding.call_id = call_id
        self.sessions[call_id] = session
        if not response.content.startswith(b"v=0"):
            raise LiveBackendError(502, "Invalid upstream Realtime SDP answer")
        return response

    async def connect(self, session: _Session, model: str | None):
        query = (
            {"call_id": session.binding.call_id}
            if session.binding.call_id
            else {"model": model}
        )
        session.upstream = await _LiveConnect(
            SOCKET_URL + "?" + urlencode(query),
            additional_headers=await self.backend.headers(session.binding),
            open_timeout=15,
            close_timeout=3,
            max_size=MAX_BODY,
            max_queue=16,
            write_limit=32768,
        )
        self.backend.succeeded(session.binding)
        return session.upstream

    async def finish(self, session: _Session) -> httpx.Response | None:
        if session.cleanup is None:
            session.cleanup = asyncio.create_task(self._finish(session))
        return await asyncio.shield(session.cleanup)

    async def _finish(self, session: _Session) -> httpx.Response | None:
        if session.timer is not None:
            session.timer.cancel()
        response = None
        try:
            # Settle any in-flight creation/attachment before releasing its
            # credential. A late successful offer must still be hung up.
            if session.operation is not None:
                await asyncio.gather(session.operation, return_exceptions=True)
            if session.binding.call_id:
                response = await self.backend.hangup(session.binding)
                if response.status_code not in {200, 204, 404, 410}:
                    logger.warning(
                        "Realtime hangup rejected (HTTP %d)", response.status_code
                    )
        except (OSError, TimeoutError, httpx.HTTPError, LiveBackendError):
            logger.warning("Unable to confirm upstream Realtime call closure")
            response = httpx.Response(
                502, json={"detail": "Upstream Realtime hangup failed"}
            )
        finally:
            try:
                if session.upstream is not None:
                    await session.upstream.close()
            finally:
                try:
                    if (
                        session.downstream is not None
                        and session.downstream.application_state
                        == WebSocketState.CONNECTED
                    ):
                        async with asyncio.timeout(5):
                            await session.downstream.close(code=1000)
                except (OSError, RuntimeError, TimeoutError, WebSocketDisconnect):
                    pass
                finally:
                    try:
                        await self.backend.release(session.binding)
                    finally:
                        if self.sessions.get(session.id) is session:
                            self.sessions.pop(session.id)
        return response

    async def close(self):
        self.closed = True
        await asyncio.gather(*(self.finish(s) for s in list(self.sessions.values())))


def _gateway(connection) -> RealtimeGateway:
    return connection.app.state.realtime


def _response(response: httpx.Response) -> Response:
    headers = {k: v for k, v in response.headers.items() if k.lower() in SAFE_HEADERS}
    headers["cache-control"] = "no-store"
    return Response(response.content, status_code=response.status_code, headers=headers)


async def _bounded_stream(request: Request):
    size = 0
    async with asyncio.timeout(15):
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY:
                raise HTTPException(413, "Realtime request exceeds 1 MiB")
            yield chunk


async def _json(request: Request):
    body = bytearray()
    async for chunk in _bounded_stream(request):
        body.extend(chunk)
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid Realtime JSON request") from None


def _config(value) -> dict:
    if not isinstance(value, dict):
        raise HTTPException(400, "session must be an object")
    config = dict(value)
    config.setdefault("type", "realtime")
    config.setdefault("model", DEFAULT_MODEL)
    model = config["model"]
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise HTTPException(400, "A nonempty Realtime model is required")
    return config


async def _offer(request: Request):
    media_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if media_type == "multipart/form-data":
        try:
            form = await MultiPartParser(
                request.headers,
                _bounded_stream(request),
                max_files=0,
                max_fields=2,
                max_part_size=MAX_BODY,
            ).parse()
            sdp = form.get("sdp")
            config = json.loads(form.get("session", "{}"))
        except (MultiPartException, ValueError, TypeError):
            raise HTTPException(
                400, "Expected multipart sdp and session fields"
            ) from None
    elif media_type in {"application/sdp", "text/plain"}:
        body = bytearray()
        async for chunk in _bounded_stream(request):
            body.extend(chunk)
        try:
            sdp = body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Invalid SDP encoding") from None
        config = {"model": request.query_params.get("model", DEFAULT_MODEL)}
    elif media_type == "application/json":
        payload = await _json(request)
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected sdp and session")
        sdp, config = payload.get("sdp"), payload.get("session", {})
    else:
        raise HTTPException(
            415, "Expected multipart/form-data, application/sdp, or application/json"
        )
    if not isinstance(sdp, str) or not sdp.strip():
        raise HTTPException(400, "A nonempty SDP offer is required")
    return sdp, _config(config)


@router.post("/client_secrets", summary="Create a native ephemeral Realtime credential")
async def client_secrets(request: Request):
    gateway = _gateway(request)
    owner = gateway.authenticate(request.headers.get("authorization"))
    session = None
    try:
        payload = await _json(request)
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected a Realtime session request")
        payload["session"] = _config(payload.get("session", {}))
        session = await gateway.acquire(payload["session"]["model"], owner)
        session.operation = asyncio.create_task(
            gateway.backend.client_secret(session.binding, payload)
        )
        return _response(await asyncio.shield(session.operation))
    except LiveBackendError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    except (httpx.HTTPError, TimeoutError):
        raise HTTPException(502, "Realtime credential request failed") from None
    finally:
        if session is not None:
            await gateway.finish(session)


@router.post("/calls", summary="Create a direct-media Realtime WebRTC call")
async def create_call(request: Request):
    gateway = _gateway(request)
    owner = gateway.authenticate(request.headers.get("authorization"))
    session = None
    retained = False
    try:
        sdp, config = await _offer(request)
        session = await gateway.acquire(config["model"], owner)
        session.operation = asyncio.create_task(
            gateway.create_call(session, sdp, config)
        )
        upstream = await asyncio.shield(session.operation)
        if session.cleanup is not None:
            raise HTTPException(410, "Realtime call closed during creation")
        response = _response(upstream)
        if upstream.is_success:
            path = request.scope.get("root_path", "") + router.prefix
            response.headers["location"] = f"{path}/calls/{session.id}"
            response.headers["link"] = (
                f'<{path}?call_id={session.id}>; rel="realtime-sideband"'
            )
            retained = True
        return response
    except LiveBackendError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    except (httpx.HTTPError, TimeoutError):
        raise HTTPException(502, "Realtime call setup failed") from None
    finally:
        if session is not None and not retained:
            await gateway.finish(session)


@router.post("/calls/{call_id}/hangup", summary="Hang up an owned Realtime call")
@router.delete("/calls/{call_id}", summary="Close an owned Realtime call")
async def hangup(call_id: str, request: Request):
    gateway = _gateway(request)
    owner = gateway.authenticate(request.headers.get("authorization"))
    response = await gateway.finish(gateway.owned(call_id, owner))
    if response is not None and response.status_code not in {200, 204, 404, 410}:
        return _response(response)
    return Response(status_code=204 if request.method == "DELETE" else 200)


async def _relay(socket: WebSocket, upstream):
    async def send():
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("text")
            if data is None:
                data = message.get("bytes")
            if data is None:
                continue
            if len(data.encode("utf-8") if isinstance(data, str) else data) > MAX_BODY:
                await socket.close(code=1009, reason="Realtime message exceeds 1 MiB")
                return
            await upstream.send(data)

    async def receive():
        async for data in upstream:
            if isinstance(data, str):
                await socket.send_text(data)
            else:
                await socket.send_bytes(data)

    tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@router.websocket("")
async def realtime(socket: WebSocket):
    gateway = _gateway(socket)
    session = None
    claimed = False
    try:
        owner = gateway.authenticate(socket.headers.get("authorization"))
        call_id = socket.query_params.get("call_id")
        model = None
        if call_id:
            session = gateway.owned(call_id, owner)
            if session.active:
                raise HTTPException(409, "Realtime sideband is already attached")
        else:
            model = _config({"model": socket.query_params.get("model", DEFAULT_MODEL)})[
                "model"
            ]
            session = await gateway.acquire(model, owner)
        session.active = claimed = True
        session.operation = asyncio.create_task(gateway.connect(session, model))
        upstream = await asyncio.shield(session.operation)
        if session.cleanup is not None:
            raise HTTPException(410, "Realtime session closed during attachment")
        protocols = socket.scope.get("subprotocols", [])
        await socket.accept(subprotocol="realtime" if "realtime" in protocols else None)
        session.downstream = socket
        if session.cleanup is not None:
            return
        await _relay(socket, upstream)
    except (HTTPException, LiveBackendError) as exc:
        await _deny(
            socket,
            exc.status_code,
            json.dumps(
                {
                    "detail": str(exc.detail)
                    if isinstance(exc, HTTPException)
                    else str(exc)
                }
            ).encode(),
            {"content-type": "application/json"},
        )
    except InvalidStatus as exc:
        response = exc.response
        safe = {
            k: v for k, v in response.headers.raw_items() if k.lower() in SAFE_HEADERS
        }
        await _deny(socket, response.status_code, bytes(response.body), safe)
    except ConnectionClosed as exc:
        if socket.application_state == WebSocketState.CONNECTED:
            code = exc.rcvd.code if exc.rcvd else 1011
            if code in {1005, 1006, 1015}:
                code = 1011
            await socket.close(code=code)
    except WebSocketDisconnect:
        pass
    except (OSError, TimeoutError, httpx.HTTPError, WebSocketException):
        if socket.application_state == WebSocketState.CONNECTING:
            await _deny(
                socket,
                502,
                b'{"detail":"Realtime connection failed"}',
                {"content-type": "application/json"},
            )
        elif socket.application_state == WebSocketState.CONNECTED:
            await socket.close(code=1011, reason="Realtime connection failed")
    finally:
        if claimed:
            await gateway.finish(session)
