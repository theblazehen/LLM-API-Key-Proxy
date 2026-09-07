"""Isolated live signaling contracts; no real credentials or network access."""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.datastructures import Headers
from websockets.http11 import Response as WebSocketResponse

SRC = Path(__file__).resolve().parents[1] / "src"
OFFER = {
    "sdp": "v=0\r\ns=offer\r\n",
    "session": {"model": "gpt-live-1-codex", "voice": "future-voice", "initial_items": [{"extension": 7}]},
    "future_extension": {"opaque": [1, None, "é"]},
}
SDP = b"v=0\r\ns=answer\r\na=future-extension:opaque\r\n"


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, SRC / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    # Other existing tests install incomplete package stubs at collection time.
    # Load these two modules explicitly and scope all dependency replacements.
    monkeypatch.syspath_prepend(str(SRC))
    for name, directory in (("rotator_library", "rotator_library"),
                            ("rotator_library.client", "rotator_library/client"),
                            ("rotator_library.providers", "rotator_library/providers")):
        package = types.ModuleType(name)
        package.__path__ = [str(SRC / directory)]
        monkeypatch.setitem(sys.modules, name, package)
    provider = types.ModuleType("rotator_library.providers.codex_provider")
    provider.USE_OPENAI_API = False
    provider.CODEX_API_BASE = "https://controlled.invalid/backend-api/codex"
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    sys.modules["rotator_library.providers"].codex_provider = provider
    errors = types.ModuleType("rotator_library.error_handler")
    errors.NoAvailableKeysError = type("NoAvailableKeysError", (Exception,), {})
    errors.classify_error = lambda error, **kwargs: error
    monkeypatch.setitem(sys.modules, errors.__name__, errors)
    usage = types.ModuleType("rotator_library.usage")
    usage.CredentialContext = object
    monkeypatch.setitem(sys.modules, usage.__name__, usage)
    backend = load_module(monkeypatch, "rotator_library.client.codex_live", "rotator_library/client/codex_live.py")
    gateway = load_module(monkeypatch, "_test_codex_live_gateway", "proxy_app/codex_live.py")
    return backend, gateway


@pytest.fixture
def setup_backend(modules):
    backend_module, _ = modules
    lease = types.SimpleNamespace(credential="fixture-oauth", mark_failure=Mock(),
                                  mark_success=Mock(), __aexit__=AsyncMock())
    provider = types.SimpleNamespace(
        get_live_oauth_identity=AsyncMock(return_value=("fixture-token", "fixture-account")),
        is_credential_available=Mock(return_value=True), update_quota_from_headers=Mock(),
    )
    manager = types.SimpleNamespace(initialized=True,
        acquire_credential=AsyncMock(return_value=lease), get_model_quota_group=Mock(return_value="live"))
    client = types.SimpleNamespace(
        _get_provider_instance=Mock(return_value=provider),
        _model_resolver=types.SimpleNamespace(is_model_allowed=Mock(return_value=True)),
        _usage_managers={"codex": manager}, all_credentials={"codex": ["fixture-oauth", "api-key"]},
        oauth_credentials={"codex": ["fixture-oauth"]},
        _credential_filter=types.SimpleNamespace(filter_by_tier=Mock(return_value=types.SimpleNamespace(
            all_usable=["fixture-oauth", "api-key"], priorities={}, tier_names={}))),
        cooldown_manager=types.SimpleNamespace(get_remaining_cooldown=AsyncMock(return_value=0)),
        http_client=types.SimpleNamespace(post=AsyncMock(return_value=httpx.Response(
            201, content=SDP, headers={"location": "/v1/realtime/calls/rtc_fixture", "content-type": "application/sdp"}))),
    )
    return types.SimpleNamespace(backend=backend_module.CodexLiveBackend(client), client=client,
                                 provider=provider, manager=manager, lease=lease)


@pytest.mark.asyncio
async def test_backend_offer_sdp_and_refreshed_identity_stay_on_one_lease(setup_backend):
    f = setup_backend
    result = await f.backend.create(OFFER, "genuine-fixture-attestation")
    assert (result.status_code, result.body) == (201, SDP)
    post = f.client.http_client.post.call_args
    assert post.kwargs["json"] is OFFER
    assert post.kwargs["follow_redirects"] is False
    assert post.kwargs["params"] == {"intent": "quicksilver", "architecture": "avas"}
    assert post.kwargs["headers"]["x-oai-attestation"] == "genuine-fixture-attestation"
    assert f.manager.acquire_credential.call_args.kwargs["candidates"] == ["fixture-oauth"]
    f.lease.__aexit__.assert_not_awaited()
    f.provider.get_live_oauth_identity.return_value = ("refreshed-token", "fixture-account")
    refreshed = await f.backend.headers(result.binding)
    assert refreshed["Authorization"] == "Bearer refreshed-token"
    for header in ("chatgpt-account-id", "session-id", "thread-id", "x-session-id"):
        assert refreshed[header] == post.kwargs["headers"][header]
    f.manager.acquire_credential.assert_awaited_once()
    assert all(c.args == ("fixture-oauth",) for c in f.provider.get_live_oauth_identity.await_args_list)
    f.provider.get_live_oauth_identity.return_value = ("other-token", "other-account")
    with pytest.raises(Exception) as raised:
        await f.backend.headers(result.binding)
    assert raised.value.status_code == 409
    await f.backend.release(result.binding)
    await f.backend.release(result.binding)
    f.lease.__aexit__.assert_awaited_once()
    with pytest.raises(Exception) as raised:
        await f.backend.headers(result.binding)
    assert raised.value.status_code == 410


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 503])
async def test_backend_rejection_is_opaque_and_releases_without_retry(setup_backend, status):
    f = setup_backend
    body = b'{"error":{"type":"future_error","opaque":[1,2]}}'
    f.client.http_client.post.return_value = httpx.Response(status, content=body, headers={
        "content-type": "application/problem+json", "retry-after": "17", "x-request-id": "req-fixture",
        "openai-request-id": "openai-fixture", "set-cookie": "never-forward", "authorization": "never-forward",
    })
    result = await f.backend.create(OFFER)
    assert (result.status_code, result.body, result.binding) == (status, body, None)
    assert result.headers == {"content-type": "application/problem+json", "retry-after": "17",
                              "x-request-id": "req-fixture", "openai-request-id": "openai-fixture"}
    f.client.http_client.post.assert_awaited_once()
    f.lease.__aexit__.assert_awaited_once()
    f.lease.mark_success.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("location,body", [
    ("", SDP), ("/calls/not-a-call", SDP), ("/rtc_one/rtc_two", SDP),
    ("http://[", SDP), ("/calls/rtc_fixture", b"not SDP"), ("/calls/rtc_fixture", b""),
])
async def test_invalid_success_cannot_retain_lease(setup_backend, location, body):
    f = setup_backend
    f.client.http_client.post.return_value = httpx.Response(201, content=body, headers={"location": location})
    with pytest.raises(Exception) as raised:
        await f.backend.create(OFFER)
    assert raised.value.status_code == 502
    f.lease.__aexit__.assert_awaited_once()
    f.lease.mark_success.assert_not_called()


@pytest.mark.asyncio
async def test_setup_cancellation_releases_acquired_lease(setup_backend):
    f = setup_backend
    entered = asyncio.Event()
    async def blocked_post(*args, **kwargs):
        entered.set()
        await asyncio.Future()
    f.client.http_client.post.side_effect = blocked_post
    task = asyncio.create_task(f.backend.create(OFFER))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    f.lease.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_survives_cancellation_and_is_exactly_once(setup_backend):
    f = setup_backend
    result = await f.backend.create(OFFER)
    entered, proceed = asyncio.Event(), asyncio.Event()
    async def blocked_release(*args):
        entered.set()
        await proceed.wait()
    f.lease.__aexit__.side_effect = blocked_release
    task = asyncio.create_task(f.backend.release(result.binding))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    proceed.set()
    await asyncio.wait_for(f.backend.release(result.binding), 1)
    f.lease.__aexit__.assert_awaited_once()


class Upstream:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, value):
        self.sent.append(value)

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed = True


class Connection:
    def __init__(self, upstream, entered=None, proceed=None):
        self.upstream, self.entered, self.proceed = upstream, entered, proceed

    async def get(self):
        if self.entered is not None:
            self.entered.set()
        if self.proceed is not None:
            await self.proceed.wait()
        return self.upstream

    def __await__(self):
        return self.get().__await__()

    async def __aenter__(self):
        return await self.get()

    async def __aexit__(self, *args):
        await self.upstream.close()


class Socket:
    def __init__(self, gateway, authorization=None, protocols=()):
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(codex_live=gateway))
        self.headers = {"authorization": authorization} if authorization else {}
        self.scope = {"subprotocols": list(protocols), "extensions": {"websocket.http.response": {}}}
        self.application_state = WebSocketState.CONNECTING
        self.incoming = asyncio.Queue()
        self.sent = []
        self.denial = None
        self.closes = []
        self.accepted = asyncio.Event()
        self.accept_gate = None
        self.selected_protocol = None

    async def accept(self, subprotocol=None):
        self.selected_protocol = subprotocol
        self.accepted.set()
        if self.accept_gate is not None:
            await self.accept_gate.wait()
        self.application_state = WebSocketState.CONNECTED

    async def receive(self):
        return await self.incoming.get()

    async def send_text(self, value):
        self.sent.append(value)

    async def send_bytes(self, value):
        self.sent.append(value)

    async def close(self, code=1000, reason=""):
        self.closes.append((code, reason))
        self.application_state = WebSocketState.DISCONNECTED

    async def send_denial_response(self, response):
        self.denial = response


@pytest.fixture
def live(modules, monkeypatch):
    bm, gm = modules
    backend = types.SimpleNamespace(create=AsyncMock(), headers=AsyncMock(return_value={"Authorization": "Bearer fixture"}),
                                    release=AsyncMock())
    binding = bm.LiveBinding("rtc_fixture", "fixture-oauth", "fixture-account", "session", "thread", "x-session")
    backend.create.return_value = bm.LiveCallResult(201, SDP, {"content-type": "application/sdp"}, binding)
    gateway = gm.CodexLiveGateway(backend, {"alice-key": "alice", "alice-second-key": "alice", "bob-key": "bob"})
    upstreams = []
    def connect(*args, **kwargs):
        upstream = Upstream()
        upstreams.append(upstream)
        return Connection(upstream)
    monkeypatch.setattr(gm, "_LiveConnect", connect)
    return types.SimpleNamespace(module=gm, backend=backend, gateway=gateway, upstreams=upstreams)


async def create(live):
    owner, digest = live.gateway.authenticate("Bearer alice-key")
    result, ticket = await live.gateway.create(OFFER, owner, digest, None)
    return live.gateway.calls[result.binding.call_id], ticket


@pytest.mark.asyncio
async def test_http_gateway_preserves_native_offer_response_and_requires_owner(live):
    app = FastAPI()
    app.include_router(live.module.router)
    app.state.codex_live = live.gateway
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        assert (await client.post("/v1/codex/realtime/calls", json=OFFER)).status_code == 401
        response = await client.post("/v1/codex/realtime/calls", json=OFFER, headers={"Authorization": "Bearer alice-key"})
        assert (response.status_code, response.content) == (201, SDP)
        assert live.backend.create.call_args.args == (OFFER, None)
        location = response.headers["location"]
        assert location == "/v1/codex/realtime/calls/rtc_fixture"
        assert response.headers["link"] == f'<{location}/events>; rel="live-sideband"'
        assert response.headers["x-live-websocket-token"]
        assert response.headers["cache-control"] == "no-store"
        assert (await client.delete(location, headers={"Authorization": "Bearer bob-key"})).status_code == 404
        assert (await client.delete(location, headers={"Authorization": "Bearer alice-second-key"})).status_code == 204
        assert not live.gateway.calls
        live.backend.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_ticket_is_one_use_duplicate_sideband_cannot_terminate_owner(live):
    call, ticket = await create(live)
    first = Socket(live.gateway, protocols=["codex-live", "proxy-ticket." + ticket])
    task = asyncio.create_task(live.module.call_events(call.binding.call_id, first))
    await asyncio.wait_for(first.accepted.wait(), 1)
    try:
        assert first.selected_protocol == "codex-live"
        repeat = Socket(live.gateway, protocols=["codex-live", "proxy-ticket." + ticket])
        await live.module.call_events(call.binding.call_id, repeat)
        assert repeat.denial.status_code == 401
        duplicate = Socket(live.gateway, authorization="Bearer alice-key")
        await live.module.call_events(call.binding.call_id, duplicate)
        assert duplicate.denial.status_code == 409
        foreign = Socket(live.gateway, authorization="Bearer bob-key")
        await live.module.call_events(call.binding.call_id, foreign)
        assert foreign.denial.status_code == 404
        live.backend.release.assert_not_awaited()
    finally:
        await first.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(task, 1)
    assert not live.gateway.calls
    assert live.upstreams[0].sent == ['{"type":"session.close"}']


@pytest.mark.asyncio
async def test_revoking_creating_key_invalidates_ticket_even_with_same_owner_key(live):
    call, ticket = await create(live)
    del live.gateway.keys["alice-key"]
    socket = Socket(live.gateway, protocols=["codex-live", "proxy-ticket." + ticket])
    await live.module.call_events(call.binding.call_id, socket)
    assert socket.denial.status_code == 401
    assert not call.active
    await live.gateway.close()


@pytest.mark.asyncio
async def test_native_extensions_and_binary_frames_remain_opaque(modules):
    _, gm = modules
    upstream = Upstream()
    socket = Socket(None)
    call = types.SimpleNamespace(close_sent=False)
    socket.application_state = WebSocketState.CONNECTED
    extension = '{"type":"future.event","extension":{"a":1}, "spacing": true}'
    binary = b"\x00\xff\x01"
    # Block client termination until both upstream frames have been delivered.
    delivered = asyncio.Event()
    original = socket.send_bytes
    async def delivered_bytes(value):
        await original(value)
        delivered.set()
    socket.send_bytes = delivered_bytes
    await upstream.incoming.put(extension)
    await upstream.incoming.put(binary)
    await socket.incoming.put({"type": "websocket.receive", "text": extension})
    await socket.incoming.put({"type": "websocket.receive", "bytes": binary})
    task = asyncio.create_task(gm._relay(socket, upstream, call))
    await asyncio.wait_for(delivered.wait(), 1)
    await socket.incoming.put({"type": "websocket.receive", "text": '{"type":"session.close","extra":true}'})
    await asyncio.wait_for(task, 1)
    assert socket.sent == [extension, binary]
    assert upstream.sent == [extension, binary, '{"type":"session.close","extra":true}']
    assert call.close_sent


@pytest.mark.asyncio
async def test_abnormal_upstream_close_is_visible_and_releases_call(live):
    call, _ = await create(live)
    socket = Socket(live.gateway, authorization="Bearer alice-key")
    task = asyncio.create_task(live.module.call_events(call.binding.call_id, socket))
    await asyncio.wait_for(socket.accepted.wait(), 1)
    await live.upstreams[0].incoming.put(ConnectionClosedError(Close(1011, "fixture failure"), None))
    await asyncio.wait_for(task, 1)
    assert socket.closes[0] == (1011, "fixture failure")
    assert not live.gateway.calls
    live.backend.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_setups_count_toward_capacity_and_cancellation_frees_slot(live):
    live.gateway.max_calls = 1
    entered = asyncio.Event()
    async def blocked(*args):
        entered.set()
        await asyncio.Future()
    live.backend.create.side_effect = blocked
    task = asyncio.create_task(create(live))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        with pytest.raises(HTTPException) as raised:
            await create(live)
        assert raised.value.status_code == 503
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert live.gateway.pending == 0
    live.backend.create.side_effect = None
    await create(live)
    with pytest.raises(HTTPException) as raised:
        await create(live)
    assert raised.value.status_code == 503
    await live.gateway.close()
    assert not live.gateway.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
async def test_expiry_terminates_abandoned_and_attached_calls(live, active):
    call, _ = await create(live)
    call.timer.cancel()
    call.active = active
    if active:
        call.upstream = Upstream()
    await asyncio.wait_for(live.gateway._expire(call, 0), 1)
    assert not live.gateway.calls
    live.backend.release.assert_awaited_once()
    upstream = call.upstream if active else live.upstreams[0]
    assert upstream.sent == ['{"type":"session.close"}']
    assert upstream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connect", "accept"])
async def test_delete_during_attachment_cannot_resurrect_released_call(live, monkeypatch, phase):
    call, _ = await create(live)
    entered, proceed = asyncio.Event(), asyncio.Event()
    upstream = Upstream()
    connections = 0
    def connect(*args, **kwargs):
        nonlocal connections
        connections += 1
        if connections == 1 and phase == "connect":
            return Connection(upstream, entered, proceed)
        return Connection(Upstream() if phase == "connect" else upstream)
    monkeypatch.setattr(live.module, "_LiveConnect", connect)
    socket = Socket(live.gateway, authorization="Bearer alice-key")
    if phase == "accept":
        socket.accept_gate = proceed
        entered = socket.accepted
    task = asyncio.create_task(live.module.call_events(call.binding.call_id, socket))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(live.gateway.finish(call), 1)
    proceed.set()
    await asyncio.wait_for(task, 1)
    assert not live.gateway.calls
    live.backend.release.assert_awaited_once()
    assert upstream.closed
    assert socket.application_state != WebSocketState.CONNECTED
    if phase == "connect":
        assert socket.denial.status_code == 410
        assert not socket.accepted.is_set()
    else:
        assert socket.closes


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_oauth_sideband_redirect_is_rejected_not_followed(modules, status):
    _, gm = modules
    rejection = InvalidStatus(WebSocketResponse(status, "Redirect", Headers({
        "Location": "wss://other-origin.invalid/steal-token",
    }), b"redirect body"))
    connector = gm._LiveConnect("wss://controlled.invalid/v1/live/rtc_fixture")
    assert connector.process_redirect(rejection) is rejection


@pytest.fixture
def oauth_modules(modules, monkeypatch):
    # Load the real identity and refresh methods without provider startup, model
    # catalog fetching, or the singleton's background tasks. Only unrelated
    # imports are substituted; refresh HTTP uses an in-memory MockTransport.
    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class NeedsReauth(Exception):
        def __init__(self, credential_path, message):
            super().__init__(message)

    errors = sys.modules["rotator_library.error_handler"]
    errors.CredentialNeedsReauthError = NeedsReauth
    errors.EmptyResponseError = type("EmptyResponseError", (Exception,), {})
    errors.TransientQuotaError = type("TransientQuotaError", (Exception,), {})
    stub("rotator_library.core.errors", StreamedAPIError=Exception)
    stub("rotator_library.providers.provider_interface",
         ProviderInterface=type("ProviderInterface", (), {}),
         UsageResetConfigDef=lambda **kwargs: kwargs, QuotaGroupMap=dict)
    stub("rotator_library.providers.utilities.codex_quota_tracker",
         CodexQuotaTracker=type("CodexQuotaTracker", (), {}))
    stub("rotator_library.model_definitions", ModelDefinitions=object)
    stub("rotator_library.timeout_config", TimeoutConfig=object)
    stub("rotator_library.utils.headless_detection", is_headless_environment=lambda: True)
    stub("rotator_library.utils.reauth_coordinator", get_reauth_coordinator=Mock())
    stub("rotator_library.utils.resilient_io", safe_write_json=Mock())
    stub("litellm")
    base = load_module(monkeypatch, "rotator_library.providers.openai_oauth_base",
                       "rotator_library/providers/openai_oauth_base.py")
    provider = load_module(monkeypatch, "rotator_library.providers._live_test_provider",
                           "rotator_library/providers/codex_provider.py")
    return base, provider, NeedsReauth


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [(400, b'{"error":"invalid_grant"}'),
                                       (401, b"unauthorized"), (403, b"forbidden")])
async def test_live_refresh_rejection_never_queues_interactive_reauth(oauth_modules, monkeypatch, status, body):
    base, module, needs_reauth = oauth_modules
    provider = object.__new__(module.CodexProvider)
    credentials = {"refresh_token": "fixture-refresh", "access_token": "expired-fixture",
                   "account_id": "fixture-account"}
    provider._credentials_cache = {}
    provider._get_lock = AsyncMock(return_value=asyncio.Lock())
    provider._load_credentials = AsyncMock(return_value=credentials)
    provider._is_token_expired = Mock(return_value=True)
    provider._queue_refresh = AsyncMock()
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(status, content=body)
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(base.httpx, "AsyncClient", lambda: client)
    with pytest.raises(needs_reauth):
        await provider.get_live_oauth_identity("fixture-oauth")
    # A forbidden create_task(_queue_refresh(...)) invokes the mock immediately,
    # so this detects queued recovery without depending on scheduler timing.
    provider._queue_refresh.assert_not_called()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_gateway_handshake_rejection_preserves_body_and_safe_headers(live, monkeypatch, status):
    call, _ = await create(live)
    body = b'{"type":"upstream.future_error","extension":42}'
    rejection = InvalidStatus(WebSocketResponse(status, "Denied", Headers({
        "content-type": "application/json", "retry-after": "9", "x-request-id": "fixture-request",
        "set-cookie": "private-cookie",
    }), body))
    class RejectedConnection(Connection):
        async def get(self):
            raise rejection
    monkeypatch.setattr(live.module, "_LiveConnect", lambda *a, **kw: RejectedConnection(None))
    socket = Socket(live.gateway, authorization="Bearer alice-key")
    await live.module.call_events(call.binding.call_id, socket)
    assert (socket.denial.status_code, socket.denial.body) == (status, body)
    assert socket.denial.headers["retry-after"] == "9"
    assert socket.denial.headers["x-request-id"] == "fixture-request"
    assert "set-cookie" not in socket.denial.headers
    assert not live.gateway.calls
    live.backend.release.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["é" * (512 * 1024 + 1), b"x" * (1024 * 1024 + 1)])
async def test_oversized_frames_close_without_upstream_forwarding(modules, data):
    _, gm = modules
    socket, upstream = Socket(None), Upstream()
    socket.application_state = WebSocketState.CONNECTED
    await socket.incoming.put({"type": "websocket.receive", "text" if isinstance(data, str) else "bytes": data})
    await asyncio.wait_for(gm._relay(socket, upstream, types.SimpleNamespace(close_sent=False)), 1)
    assert socket.closes[0][0] == 1009
    assert not upstream.sent


@pytest.mark.asyncio
async def test_client_session_close_is_not_duplicated_by_cleanup(live):
    call, _ = await create(live)
    socket = Socket(live.gateway, authorization="Bearer alice-key")
    native_close = '{"type":"session.close", "future_extension":true}'
    await socket.incoming.put({"type": "websocket.receive", "text": native_close})
    await asyncio.wait_for(live.module.call_events(call.binding.call_id, socket), 1)
    assert live.upstreams[0].sent == [native_close]
    assert live.upstreams[0].closed
    live.backend.release.assert_awaited_once()
    assert not live.gateway.calls
