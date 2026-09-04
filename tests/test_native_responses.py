import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class _Trace:
    def __init__(self):
        self.responses = []
        self.errors = []
        self.completed_calls = 0
        self._finished = False

    def response(self, response):
        self.responses.append(response)

    def error(self, error):
        self.errors.append(error)

    def completed(self, status=None):
        self.completed_calls += 1
        self._finished = True


class _Request:
    def __init__(self, trace):
        self.state = SimpleNamespace(llm_trace=trace)


def test_start_llm_trace_records_ordered_asgi_headers_before_body():
    import proxy_app.main as main

    captured = []

    class Trace:
        def headers(self, direction, headers):
            captured.append(("headers", direction, headers))

        def request(self, payload):
            captured.append(("request", payload))

    class Request:
        headers = {"x-request-id": "request-id"}
        url = SimpleNamespace(path="/v1/responses")
        scope = {
            "headers": [
                (b"authorization", b"Bearer raw-token"),
                (b"cookie", b"session=raw-cookie"),
                (b"x-duplicate", b"first"),
                (b"x-duplicate", b"second"),
                (b"x-byte", b"\xff"),
            ]
        }
        state = SimpleNamespace(proxy_identity=main.ProxyIdentity("test-user"))

    trace = Trace()
    original_begin = main.begin_llm_trace
    try:
        main.begin_llm_trace = lambda **_: trace
        result = main.start_llm_trace(Request(), {"model": "codex/gpt-5.6-sol"})
    finally:
        main.begin_llm_trace = original_begin

    assert result is trace
    assert captured == [
        ("headers", "client_request", Request.scope["headers"]),
        ("request", {"model": "codex/gpt-5.6-sol"}),
    ]


async def _collect(stream):
    return [chunk async for chunk in stream]


def _native_stream(chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return stream()


def test_native_response_stream_preserves_fragmented_byte_sse_and_traces_terminal_response():
    from proxy_app.main import traced_native_response_stream

    terminal = {
        "id": "resp_terminal",
        "object": "response",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 8, "cache_creation_tokens": 4},
            "output_tokens": 2,
        },
    }
    event = json.dumps({"type": "response.completed", "response": terminal})
    chunks = [
        b"event: response.completed\n",
        b"data: " + event[:31].encode(),
        event[31:].encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    trace = _Trace()

    output = asyncio.run(_collect(traced_native_response_stream(_Request(trace), _native_stream(chunks))))

    assert output == chunks
    assert trace.responses == [terminal]
    assert trace.errors == []
    assert trace.completed_calls == 1


def test_native_response_stream_parses_string_sse_records_without_rewriting_them():
    from proxy_app.main import traced_native_response_stream

    terminal = {"id": "resp_string", "status": "completed", "output": [], "usage": {"input_tokens": 3}}
    event = json.dumps({"type": "response.completed", "response": terminal})
    chunks = [f"event: response.completed\ndata: {event}\n\n", "data: [DONE]\n\n"]
    trace = _Trace()

    output = asyncio.run(_collect(traced_native_response_stream(_Request(trace), _native_stream(chunks))))

    assert output == chunks
    assert trace.responses == [terminal]
    assert trace.completed_calls == 1


def test_native_response_stream_reports_upstream_exception_and_finishes_trace():
    from proxy_app.main import traced_native_response_stream

    async def broken_stream():
        yield b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\"}\n\n"
        raise RuntimeError("upstream stream disconnected")

    trace = _Trace()

    async def consume():
        return [chunk async for chunk in traced_native_response_stream(_Request(trace), broken_stream())]

    try:
        asyncio.run(consume())
    except RuntimeError as error:
        assert str(error) == "upstream stream disconnected"
    else:
        raise AssertionError("native stream error must propagate")

    assert len(trace.errors) == 1
    assert str(trace.errors[0]) == "upstream stream disconnected"
    assert trace.responses == []
    assert trace.completed_calls == 1


class _CompactRequest:
    def __init__(self, payload):
        self._payload = payload
        self.state = SimpleNamespace(proxy_identity=SimpleNamespace(user="test-user"))
        self.headers = {}
        self.scope = {"headers": []}
        self.url = SimpleNamespace(path="/v1/responses/compact", __str__=lambda _: "http://test/v1/responses/compact")
        self.client = SimpleNamespace(host="test", port=1234)

    async def json(self):
        return self._payload


def test_rotating_client_acompact_uses_only_capable_native_executor_candidates():
    from rotator_library.client.rotating_client import RotatingClient

    contexts = []
    capable = SimpleNamespace(supports_compact_api=lambda: True)
    incapable = SimpleNamespace(supports_compact_api=lambda: False)

    class Executor:
        async def execute(self, context):
            contexts.append(context)
            return {"output": [{"type": "compaction", "encrypted_content": "opaque"}]}

    client = SimpleNamespace(
        _model_resolver=SimpleNamespace(
            resolve_model_chain=lambda _: ["other/model", "codex/gpt-5.6-sol"]
        ),
        all_credentials={"other": ["other.json"], "codex": ["codex.json"]},
        _get_provider_instance=lambda provider: {
            "other": incapable,
            "codex": capable,
        }[provider],
        _build_completion_context=lambda model, provider, kwargs, *args: SimpleNamespace(
            model=model, provider=provider, kwargs=kwargs
        ),
        _executor=Executor(),
        _is_error_response=lambda _: False,
    )

    response = asyncio.run(
        RotatingClient.acompact(
            client,
            model="alias/compact",
            input=[{"role": "user", "content": "keep exact"}],
        )
    )

    assert response == {
        "output": [{"type": "compaction", "encrypted_content": "opaque"}]
    }
    assert len(contexts) == 1
    assert contexts[0].model == "codex/gpt-5.6-sol"
    assert contexts[0].kwargs == {
        "model": "alias/compact",
        "input": [{"role": "user", "content": "keep exact"}],
        "stream": False,
        "_use_responses": True,
        "_compact": True,
    }


def test_rotating_client_normalizes_lite_transport_for_responses_and_compact():
    from rotator_library.client.rotating_client import RotatingClient

    lite_header = "x-openai-internal-codex-responses-lite"
    contexts = []

    class Executor:
        async def execute(self, context):
            contexts.append(context)
            return {"id": "native-result", "output": []}

    client = object.__new__(RotatingClient)
    client._model_resolver = SimpleNamespace(
        resolve_model_chain=lambda _: ["codex/gpt-5.6-sol"],
        resolve_model_id=lambda model, _provider: model,
    )
    client.all_credentials = {"codex": ["codex.json"]}
    client._get_provider_instance = lambda _provider: SimpleNamespace(
        supports_responses_api=lambda: True,
        supports_compact_api=lambda: True,
    )
    client._executor = Executor()
    client._is_error_response = lambda _: False
    client.enable_request_logging = False
    client.global_timeout = 30
    request = SimpleNamespace(
        headers={lite_header: "TrUe"},
        state=SimpleNamespace(),
    )

    async def dispatch_lite_requests():
        await client.aresponses(
            request=request,
            model="codex/gpt-5.6-sol",
            input=[{"type": "message", "role": "user", "content": []}],
            stream=False,
        )
        await client.acompact(
            request=request,
            model="codex/gpt-5.6-sol",
            input=[{"type": "message", "role": "user", "content": []}],
        )

    asyncio.run(dispatch_lite_requests())

    assert [context.provider for context in contexts] == ["codex", "codex"]
    assert all(context.kwargs["_codex_responses_lite"] is True for context in contexts)
    assert all(lite_header not in context.kwargs for context in contexts)
    assert "_compact" not in contexts[0].kwargs
    assert contexts[1].kwargs["_compact"] is True


def test_rotating_client_rejects_spoofed_lite_json_without_header():
    from rotator_library.client.rotating_client import RotatingClient

    lite_header = "x-openai-internal-codex-responses-lite"
    contexts = []

    class Executor:
        async def execute(self, context):
            contexts.append(context)
            return {"id": "native-result", "output": []}

    client = object.__new__(RotatingClient)
    client._model_resolver = SimpleNamespace(
        resolve_model_chain=lambda _: ["codex/gpt-5.6-sol"],
        resolve_model_id=lambda model, _provider: model,
    )
    client.all_credentials = {"codex": ["codex.json"]}
    client._get_provider_instance = lambda _provider: SimpleNamespace(
        supports_responses_api=lambda: True,
    )
    client._executor = Executor()
    client._is_error_response = lambda _: False
    client.enable_request_logging = False
    client.global_timeout = 30

    async def dispatch_non_lite_requests():
        for headers in ({}, {lite_header: "FALSE"}):
            await client.aresponses(
                request=SimpleNamespace(headers=headers, state=SimpleNamespace()),
                model="codex/gpt-5.6-sol",
                input="unchanged input",
                stream=False,
                **{
                    lite_header: "true",
                    "_codex_responses_lite": True,
                },
            )

    asyncio.run(dispatch_non_lite_requests())

    assert len(contexts) == 2
    assert all(lite_header not in context.kwargs for context in contexts)
    assert all("_codex_responses_lite" not in context.kwargs for context in contexts)


def test_compact_route_passes_native_json_through_exactly(monkeypatch):
    import proxy_app.main as main

    payload = {"model": "codex/gpt-5.6-sol", "input": [{"role": "user", "content": "hi"}]}
    native = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "retained exactly"}],
            },
            {
                "type": "compaction_summary",
                "encrypted_content": "opaque-value",
            },
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    trace = _Trace()
    request = _CompactRequest(payload)
    client = SimpleNamespace(acompact=lambda **_: None)

    async def compact(**kwargs):
        assert kwargs["request"] is request
        assert {key: value for key, value in kwargs.items() if key != "request"} == payload
        return native

    client.acompact = compact
    monkeypatch.setattr(main, "ENABLE_RAW_LOGGING", False)
    monkeypatch.setattr(main, "start_llm_trace", lambda *_, **__: setattr(request.state, "llm_trace", trace))
    monkeypatch.setattr(main, "log_request_to_console", lambda **_: None)

    response = asyncio.run(main.compact_responses(request, client, None))

    assert response.status_code == 200
    assert json.loads(response.body) == native
    assert trace.responses == [native]
    assert trace.completed_calls == 1


def test_compact_route_returns_explicit_501_without_fallback(monkeypatch):
    import proxy_app.main as main

    request = _CompactRequest({"model": "ollama/model", "input": "history"})
    trace = _Trace()
    calls = []

    async def unsupported(**kwargs):
        calls.append(kwargs)
        return None

    client = SimpleNamespace(acompact=unsupported)
    monkeypatch.setattr(main, "ENABLE_RAW_LOGGING", False)
    monkeypatch.setattr(main, "start_llm_trace", lambda *_, **__: setattr(request.state, "llm_trace", trace))
    monkeypatch.setattr(main, "log_request_to_console", lambda **_: None)

    response = asyncio.run(main.compact_responses(request, client, None))
    body = json.loads(response.body)

    assert response.status_code == 501
    assert body["error"]["code"] == "native_compact_unsupported"
    assert "no Chat Completions or local fallback" in body["error"]["message"]
    assert len(calls) == 1
    assert len(trace.errors) == 1
    assert trace.completed_calls == 1


def test_compact_route_preserves_safe_upstream_4xx_without_body_leak(monkeypatch):
    import proxy_app.main as main

    request = _CompactRequest({"model": "codex/gpt-5.6-sol", "input": "history"})
    trace = _Trace()
    secret = "raw-upstream-secret-must-not-leak"
    upstream_request = httpx.Request("POST", "https://upstream.invalid/responses/compact")
    upstream_response = httpx.Response(
        422,
        request=upstream_request,
        json={"error": {"message": secret, "type": "invalid_request_error"}},
    )
    error = httpx.HTTPStatusError(
        f"upstream rejected payload: {secret}",
        request=upstream_request,
        response=upstream_response,
    )
    error.upstream_status_code = 422
    error.upstream_error_category = "invalid_request_error"

    async def rejected(**_kwargs):
        raise error

    client = SimpleNamespace(acompact=rejected)
    monkeypatch.setattr(main, "ENABLE_RAW_LOGGING", False)
    monkeypatch.setattr(main, "start_llm_trace", lambda *_, **__: setattr(request.state, "llm_trace", trace))
    monkeypatch.setattr(main, "log_request_to_console", lambda **_: None)

    response = asyncio.run(main.compact_responses(request, client, None))
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body == {
        "error": {
            "message": "Native Responses compaction was rejected by the upstream service with HTTP 422.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request_error",
        }
    }
    assert secret not in response.body.decode()
    assert len(trace.errors) == 1
    assert secret not in str(trace.errors[0])
    assert trace.completed_calls == 1
