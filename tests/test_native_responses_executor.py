"""Focused contracts for native Codex Responses streaming.

These tests use no credentials or network. They describe the boundary between the
native Responses provider stream and the existing executor accounting/retry path.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.error_handler import classify_error
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers import codex_provider


async def _collect(stream):
    return [chunk async for chunk in stream]


class _CredentialContext:
    def __init__(self):
        self.successes = []

    def mark_success(self, **usage):
        self.successes.append(usage)


class _TransactionLogger:
    def __init__(self):
        self.responses = []

    def log_response(self, response):
        self.responses.append(response)


class _Trace:
    def __init__(self):
        self.responses = []
        self.completed_calls = 0

    def response(self, response):
        self.responses.append(response)

    def completed(self):
        self.completed_calls += 1


class _HeaderTrace:
    def __init__(self):
        self.header_events = []

    def headers(self, direction, headers, *, status=None, metadata=None):
        pairs = headers.items() if hasattr(headers, "items") else headers
        self.header_events.append((direction, list(pairs), status, metadata))


class _TraceForwardingPlugin:
    def has_custom_logic(self):
        return True

    async def acompletion(self, _client, **kwargs):
        return kwargs.get("_llm_trace")


class _NativeStreamingPlugin:
    def __init__(self):
        self.kwargs = None

    def supports_responses_api(self):
        return True

    def has_custom_logic(self):
        return True

    async def aresponses(self, _client, **kwargs):
        self.kwargs = kwargs

        async def chunks():
            yield b"data: [DONE]\n\n"

        return chunks()


class _NativeUnaryPlugin:
    calculate_api_equivalent_cost = True

    def supports_responses_api(self):
        return True

    def has_custom_logic(self):
        return True

    async def aresponses(self, _client, **_kwargs):
        return {
            "id": "resp_compact",
            "output": [{"type": "compaction", "encrypted_content": "opaque"}],
            "usage": {
                "input_tokens": 20,
                "input_tokens_details": {
                    "cached_tokens": 12,
                    "cache_creation_tokens": 3,
                },
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        }


def test_executor_forwards_request_trace_to_codex_custom_provider():
    trace = SimpleNamespace(
        credential_selected=lambda **_kwargs: None,
        response=lambda _payload: None,
        completed=lambda: None,
    )
    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(llm_trace=trace),
    )
    executor = object.__new__(RequestExecutor)
    executor._transforms = SimpleNamespace(
        apply=lambda *_args, **_kwargs: None,
    )

    async def prepare(_provider, _model, _cred, _context):
        return {"model": "codex/gpt-5.6-sol"}

    executor._prepare_request_kwargs = prepare
    plugin = _TraceForwardingPlugin()
    executor._get_plugin_instance = lambda _provider: plugin
    executor._run_pre_request_callback = lambda *_args: asyncio.sleep(0)
    executor._extract_usage_tokens = lambda _response: (0, 0, 0, 0, 0)
    executor._calculate_cost = lambda *_args: 0.0
    executor._extract_response_headers = lambda _response: None
    executor._max_retries = 1
    executor._http_client = object()

    class CredentialContext:
        credential = "credential.json"
        stable_id = "stable"

        def mark_success(self, **_kwargs):
            pass

    class Acquired:
        async def __aenter__(self):
            return CredentialContext()

        async def __aexit__(self, *_args):
            return False

    class UsageManager:
        states = {}

        async def get_availability_stats(self, *_args):
            return {"available": 1, "total": 1}

        async def acquire_credential(self, **_kwargs):
            return Acquired()

    async def prepare_execution(_context):
        return UsageManager(), SimpleNamespace(priorities={}), ["credential.json"], None, {}

    executor._prepare_execution = prepare_execution
    executor._wait_for_cooldown = lambda *_args: asyncio.sleep(0)
    executor._log_acquiring_credential = lambda *_args: None
    executor._log_acquired_credential = lambda *_args: None

    context = RequestContext(
        model="codex/gpt-5.6-sol",
        provider="codex",
        kwargs={},
        streaming=False,
        credentials=["credential.json"],
        deadline=time.time() + 5,
        request=request,
    )

    assert asyncio.run(executor._execute_non_streaming(context)) is trace


def test_executor_native_compact_dict_records_native_usage_and_cost():
    executor = object.__new__(RequestExecutor)
    executor._transforms = SimpleNamespace(apply=lambda *_args, **_kwargs: None)

    async def prepare(_provider, _model, _cred, _context):
        return {
            "model": "codex/gpt-5.6-sol",
            "input": "history",
            "_use_responses": True,
            "_compact": True,
        }

    executor._prepare_request_kwargs = prepare
    executor._get_plugin_instance = lambda _provider: _NativeUnaryPlugin()
    executor._run_pre_request_callback = lambda *_args: asyncio.sleep(0)
    executor._extract_usage_tokens = lambda _response: (_ for _ in ()).throw(
        AssertionError("generic usage extraction must not handle native dicts")
    )
    executor._calculate_cost = lambda *_args: (_ for _ in ()).throw(
        AssertionError("generic cost calculation must not handle native dicts")
    )
    cost_calls = []

    def native_cost(*args):
        cost_calls.append(args)
        return 1.25

    executor._calculate_native_responses_cost = native_cost
    executor._extract_response_headers = lambda _response: None
    executor._max_retries = 1
    executor._http_client = object()
    executor._wait_for_cooldown = lambda *_args: asyncio.sleep(0)
    executor._log_acquiring_credential = lambda *_args: None
    executor._log_acquired_credential = lambda *_args: None
    credential = _CredentialContext()
    credential.credential = "credential.json"
    credential.stable_id = "stable"

    class Acquired:
        async def __aenter__(self):
            return credential

        async def __aexit__(self, *_args):
            return False

    class UsageManager:
        states = {}

        async def get_availability_stats(self, *_args):
            return {"available": 1, "total": 1}

        async def acquire_credential(self, **_kwargs):
            return Acquired()

    async def prepare_execution(_context):
        return UsageManager(), SimpleNamespace(priorities={}), ["credential.json"], None, {}

    executor._prepare_execution = prepare_execution
    context = RequestContext(
        model="codex/gpt-5.6-sol",
        provider="codex",
        kwargs={"_use_responses": True, "_compact": True},
        streaming=False,
        credentials=["credential.json"],
        deadline=time.time() + 5,
        request=SimpleNamespace(headers={}, state=SimpleNamespace(llm_trace=None)),
    )

    response = asyncio.run(executor._execute_non_streaming(context))

    assert response["id"] == "resp_compact"
    assert cost_calls == [("codex", "codex/gpt-5.6-sol", 8, 3, 12, 3, 2)]
    assert len(credential.successes) == 1
    assert credential.successes[0] == {
        "response": response,
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "thinking_tokens": 2,
        "prompt_tokens_cache_read": 12,
        "prompt_tokens_cache_write": 3,
        "approx_cost": 1.25,
        "response_headers": None,
    }


def test_executor_native_stream_dispatch_does_not_add_chat_stream_options():
    plugin = _NativeStreamingPlugin()
    executor = object.__new__(RequestExecutor)
    executor._transforms = SimpleNamespace(apply=lambda *_args, **_kwargs: None)

    async def prepare(_provider, _model, _cred, _context):
        return {
            "model": "codex/gpt-5.6-sol",
            "stream": True,
            "_use_responses": True,
        }

    executor._prepare_request_kwargs = prepare
    executor._get_plugin_instance = lambda _provider: plugin
    executor._run_pre_request_callback = lambda *_args: asyncio.sleep(0)
    executor._max_retries = 1
    executor._http_client = object()
    executor._wait_for_cooldown = lambda *_args: asyncio.sleep(0)
    executor._log_acquiring_credential = lambda *_args: None
    executor._log_acquired_credential = lambda *_args: None

    async def passthrough(stream, **_kwargs):
        async for chunk in stream:
            yield chunk

    executor._native_responses_stream_wrapper = passthrough

    class CredentialContext:
        credential = "credential.json"
        stable_id = "stable"

    class Acquired:
        async def __aenter__(self):
            return CredentialContext()

        async def __aexit__(self, *_args):
            return False

    class UsageManager:
        states = {}

        async def get_availability_stats(self, *_args):
            return {"available": 1, "total": 1}

        async def acquire_credential(self, **_kwargs):
            return Acquired()

    async def prepare_execution(_context):
        return UsageManager(), SimpleNamespace(priorities={}), ["credential.json"], None, {}

    executor._prepare_execution = prepare_execution
    context = RequestContext(
        model="codex/gpt-5.6-sol",
        provider="codex",
        kwargs={"stream": True, "_use_responses": True},
        streaming=True,
        credentials=["credential.json"],
        deadline=time.time() + 5,
        request=SimpleNamespace(headers={}, state=SimpleNamespace(llm_trace=None)),
    )

    assert asyncio.run(_collect(executor._execute_streaming(context))) == [
        b"data: [DONE]\n\n"
    ]
    assert "stream_options" not in plugin.kwargs


_RESPONSE_RAW_HEADERS = [
    (b"x-codex-primary-used-percent", b"12.5"),
    (b"x-codex-primary-reset-at", b"first-reset"),
    (b"x-codex-primary-reset-at", b"second-reset"),
]


class _ErrorResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.request = httpx.Request("POST", "https://example.invalid/responses")
        self.headers = httpx.Headers(_RESPONSE_RAW_HEADERS)

    @property
    def text(self):
        return self._body.decode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        if False:
            yield ""

    def raise_for_status(self):
        response = httpx.Response(
            self.status_code,
            content=self._body,
            headers=self.headers,
            request=self.request,
        )
        response.raise_for_status()


class _ErrorClient:
    def __init__(self, status_code, body):
        self.response = _ErrorResponse(status_code, body)

    def stream(self, *_args, **_kwargs):
        return self.response


class _NonStreamingErrorClient:
    def __init__(self, status_code, body):
        self.response = httpx.Response(
            status_code,
            content=body,
            headers=_RESPONSE_RAW_HEADERS,
            request=httpx.Request("POST", "https://example.invalid/responses"),
        )

    async def post(self, *_args, **_kwargs):
        return self.response


class _NativeSuccessClient:
    def __init__(self):
        self.headers = None

    async def post(self, *_args, headers, **_kwargs):
        self.headers = headers
        return httpx.Response(
            200,
            json={"id": "resp_success", "output": []},
            headers=_RESPONSE_RAW_HEADERS,
            request=httpx.Request("POST", "https://example.invalid/responses"),
        )


class _CapturingJSONClient:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.url = None
        self.headers = None
        self.payload = None

    async def post(self, url, *, headers, json, **_kwargs):
        self.url = url
        self.headers = headers
        self.payload = json
        return httpx.Response(
            200,
            json=self.response_payload,
            headers=_RESPONSE_RAW_HEADERS,
            request=httpx.Request("POST", url),
        )


class _SSESuccessResponse:
    def __init__(self):
        self.status_code = 200
        self.request = httpx.Request("POST", "https://example.invalid/responses")
        self.headers = httpx.Headers(_RESPONSE_RAW_HEADERS)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        yield 'data: {"type":"response.output_text.delta","delta":"ok"}'
        yield "data: [DONE]"


class _SSEClient:
    def __init__(self):
        self.headers = None

    def stream(self, *_args, headers, **_kwargs):
        self.headers = headers
        return _SSESuccessResponse()


def _fragmented_native_stream(chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return stream()


def test_native_terminal_usage_is_accounted_and_logged_after_fragmented_sse():
    terminal = {
        "id": "resp_terminal",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {
                "cached_tokens": 8,
                "cache_creation_tokens": 4,
            },
            "output_tokens": 2,
        },
    }
    event = json.dumps({"type": "response.completed", "response": terminal})
    chunks = [
        b"event: response.completed\n",
        b"data: " + event[:29].encode("utf-8"),
        event[29:].encode("utf-8") + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    credential = _CredentialContext()
    transaction_logger = _TransactionLogger()
    trace = _Trace()
    executor = RequestExecutor.__new__(RequestExecutor)

    output = asyncio.run(
        _collect(
            executor._native_responses_stream_wrapper(
                _fragmented_native_stream(chunks),
                provider="codex",
                model="gpt-5.6-terra",
                cred_context=credential,
                transaction_logger=transaction_logger,
                llm_trace=trace,
            )
        )
    )

    assert output == chunks
    assert transaction_logger.responses == [terminal]
    assert trace.responses == [terminal]
    assert trace.completed_calls == 1
    assert len(credential.successes) == 1
    usage = credential.successes[0]
    # Cached input is accounted separately, so ordinary prompt usage is uncached input.
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 2
    assert usage["prompt_tokens_cache_read"] == 8
    assert usage["prompt_tokens_cache_write"] == 4


def test_native_stream_wrapper_preserves_compaction_events_and_items_byte_identical():
    chunks = [
        b"event: response.compaction.delta\n",
        b'data: {"type":"response.compaction.delta","delta":"opaque"}\n\n',
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"compaction","encrypted_content":"ciphertext"}}\n\n',
        b"event: response.completed\n",
        b'data: {"type":"response.completed","response":{"id":"resp_compacted","object":"response","status":"completed","output":[{"type":"compaction","encrypted_content":"ciphertext"}],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    executor = RequestExecutor.__new__(RequestExecutor)

    output = asyncio.run(
        _collect(
            executor._native_responses_stream_wrapper(
                _fragmented_native_stream(chunks),
                provider="codex",
                model="gpt-5.6-sol",
                cred_context=_CredentialContext(),
                transaction_logger=None,
                llm_trace=None,
            )
        )
    )

    assert output == chunks


@pytest.mark.parametrize(
    ("status_code", "body", "expected_error_type"),
    [
        (400, b'{"error":{"message":"invalid request"}}', "invalid_request"),
        (429, b'{"error":{"message":"rate limit exceeded"}}', "rate_limit"),
    ],
)
def test_native_responses_http_failures_raise_classifiable_exceptions(
    monkeypatch, status_code, body, expected_error_type
):
    provider = CodexProvider()

    async def no_credential_recovery(*_args, **_kwargs):
        return None

    monkeypatch.setattr(provider, "_recover_unauthorized_credential", no_credential_recovery)

    async def consume():
        stream = provider._stream_native_responses(
            _ErrorClient(status_code, body),
            headers={},
            payload={"stream": True},
            credential_path="",
        )
        return [chunk async for chunk in stream]

    with pytest.raises(httpx.HTTPStatusError) as caught:
        asyncio.run(consume())

    assert caught.value.response.status_code == status_code
    assert classify_error(caught.value, "codex").error_type == expected_error_type


def test_native_non_streaming_error_retains_safe_status_and_category(monkeypatch):
    provider = CodexProvider()
    trace = _HeaderTrace()
    raw_message = "distinctive raw upstream message"
    bearer_secret = "Bearer secret-token-that-must-not-leak"
    account_id = "acct-sensitive-identifier"
    body = json.dumps(
        {
            "error": {
                "type": "invalid_request/error with spaces",
                "message": raw_message,
                "authorization": bearer_secret,
                "account_id": account_id,
            }
        }
    ).encode()

    async def auth_header(_credential):
        return {"Authorization": bearer_secret}

    async def get_account_id(_credential):
        return account_id

    async def no_credential_recovery(*_args, **_kwargs):
        return None

    monkeypatch.setattr(provider, "get_auth_header", auth_header)
    monkeypatch.setattr(provider, "get_account_id", get_account_id)
    monkeypatch.setattr(provider, "_recover_unauthorized_credential", no_credential_recovery)

    with pytest.raises(httpx.HTTPStatusError) as caught:
        asyncio.run(
            provider.aresponses(
                _NonStreamingErrorClient(422, body),
                credential_identifier="credential.json",
                model="codex/gpt-5.6-sol",
                input="safe fixed input",
                stream=False,
                _llm_trace=trace,
            )
        )

    error = caught.value
    rendered = f"{error!s}\n{error!r}"
    assert error.response.status_code == 422
    assert error.upstream_status_code == 422
    assert error.upstream_error_category == "invalid_request_error_with_spaces"
    assert "status=422" in rendered
    assert "category=invalid_request_error_with_spaces" in rendered
    assert raw_message not in rendered
    assert bearer_secret not in rendered
    assert account_id not in rendered
    assert raw_message not in error.args
    assert bearer_secret not in error.args
    assert account_id not in error.args
    assert raw_message not in error.__dict__.values()
    assert bearer_secret not in error.__dict__.values()
    assert account_id not in error.__dict__.values()
    assert trace.header_events[0][0] == "provider_request"
    assert trace.header_events[0][1][0] == ("Authorization", bearer_secret)
    response_event = trace.header_events[1]
    assert response_event[0] == "provider_response"
    assert response_event[1][: len(_RESPONSE_RAW_HEADERS)] == _RESPONSE_RAW_HEADERS
    assert response_event[2:] == (422, {"boundary": "native_responses"})


def test_native_non_streaming_headers_preserve_auth_and_duplicate_response_pairs(monkeypatch):
    provider = CodexProvider()
    trace = _HeaderTrace()
    client = _NativeSuccessClient()

    async def auth_header(_credential):
        return {"Authorization": "Bearer test-token"}

    async def no_account_id(_credential):
        return None

    monkeypatch.setattr(provider, "get_auth_header", auth_header)
    monkeypatch.setattr(provider, "get_account_id", no_account_id)

    result = asyncio.run(
        provider.aresponses(
            client,
            credential_identifier="credential.json",
            model="codex/gpt-5.6-sol",
            input="input",
            stream=False,
            _llm_trace=trace,
        )
    )

    assert result["id"] == "resp_success"
    assert client.headers["Authorization"] == "Bearer test-token"
    assert trace.header_events[0] == (
        "provider_request",
        list(client.headers.items()),
        None,
        {"boundary": "native_responses"},
    )
    response_event = trace.header_events[1]
    assert response_event[0] == "provider_response"
    assert response_event[1][: len(_RESPONSE_RAW_HEADERS)] == _RESPONSE_RAW_HEADERS
    assert response_event[2:] == (200, {"boundary": "native_responses"})


def test_native_provider_filters_chat_only_stream_options(monkeypatch):
    provider = CodexProvider()
    client = _CapturingJSONClient({"id": "resp_success"})

    async def auth_header(_credential):
        return {"Authorization": "Bearer test-token"}

    async def no_account_id(_credential):
        return None

    monkeypatch.setattr(provider, "get_auth_header", auth_header)
    monkeypatch.setattr(provider, "get_account_id", no_account_id)

    result = asyncio.run(
        provider.aresponses(
            client,
            credential_identifier="credential.json",
            model="codex/gpt-5.6-sol",
            input="input",
            stream=False,
            stream_options={"include_usage": True},
            max_output_tokens=23,
            prompt_cache_retention="24h",
            context_management=[{"type": "compaction", "compact_threshold": 99}],
        )
    )

    assert result == {"id": "resp_success"}
    assert "stream_options" not in client.payload
    assert client.payload["max_output_tokens"] == 23
    assert client.payload["prompt_cache_retention"] == "24h"
    assert client.payload["context_management"] == [
        {"type": "compaction", "compact_threshold": 99}
    ]


def test_compact_posts_unary_json_and_passes_response_through(monkeypatch):
    provider = CodexProvider()
    upstream_payload = {
        "id": "cmp_123",
        "output": [
            {
                "type": "compaction",
                "encrypted_content": "opaque",
                "unknown_future_field": {"preserved": True},
            }
        ],
    }
    client = _CapturingJSONClient(upstream_payload)
    trace = _HeaderTrace()
    quota_headers = []

    async def auth_header(_credential):
        return {"Authorization": "Bearer test-token"}

    async def account_id(_credential):
        return "acct-test"

    monkeypatch.setattr(provider, "get_auth_header", auth_header)
    monkeypatch.setattr(provider, "get_account_id", account_id)
    monkeypatch.setattr(
        provider,
        "update_quota_from_headers",
        lambda credential, headers: quota_headers.append((credential, headers)),
    )

    result = asyncio.run(
        provider.aresponses(
            client,
            credential_identifier="credential.json",
            model="codex/gpt-5.6-sol",
            input=[{"role": "user", "content": "compact this"}],
            stream=True,
            stream_options={"include_usage": True},
            _compact=True,
            _llm_trace=trace,
        )
    )

    assert result == upstream_payload
    assert client.url == codex_provider.CODEX_RESPONSES_COMPACT_ENDPOINT
    assert client.payload == {
        "model": "gpt-5.6-sol",
        "input": [{"role": "user", "content": "compact this"}],
    }
    assert client.headers["Authorization"] == "Bearer test-token"
    assert client.headers["ChatGPT-Account-Id"] == "acct-test"
    assert client.headers["Accept"] == "application/json"
    assert quota_headers[0][0] == "credential.json"
    assert quota_headers[0][1]["x-codex-primary-used-percent"] == "12.5"
    assert [event[3] for event in trace.header_events] == [
        {"boundary": "native_responses_compact"},
        {"boundary": "native_responses_compact"},
    ]


def test_native_stream_error_headers_are_recorded_before_status_handling(monkeypatch):
    provider = CodexProvider()
    trace = _HeaderTrace()

    async def no_credential_recovery(*_args, **_kwargs):
        return None

    monkeypatch.setattr(provider, "_recover_unauthorized_credential", no_credential_recovery)

    async def consume():
        stream = provider._stream_native_responses(
            _ErrorClient(429, b'{"error":{"message":"rate limit"}}'),
            headers={"Authorization": "Bearer test-token"},
            payload={"stream": True},
            credential_path="",
            trace=trace,
        )
        return [chunk async for chunk in stream]

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(consume())

    assert trace.header_events == [
        ("provider_request", [("Authorization", "Bearer test-token")], None, {"boundary": "native_responses"}),
        ("provider_response", _RESPONSE_RAW_HEADERS, 429, {"boundary": "native_responses"}),
    ]


def test_native_stream_headers_preserve_auth_and_duplicate_response_pairs():
    trace = _HeaderTrace()
    client = _SSEClient()

    async def consume():
        stream = CodexProvider()._stream_native_responses(
            client,
            headers={"Authorization": "Bearer test-token"},
            payload={"stream": True},
            credential_path="",
            trace=trace,
        )
        return [chunk async for chunk in stream]

    assert asyncio.run(consume()) == [
        b'data: {"type":"response.output_text.delta","delta":"ok"}\n',
        b"data: [DONE]\n",
    ]
    assert trace.header_events == [
        ("provider_request", [("Authorization", "Bearer test-token")], None, {"boundary": "native_responses"}),
        ("provider_response", _RESPONSE_RAW_HEADERS, 200, {"boundary": "native_responses"}),
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_via_responses_headers_preserve_ordered_raw_pairs(monkeypatch, stream):
    provider = CodexProvider()
    trace = _HeaderTrace()
    client = _SSEClient()

    async def auth_header(_credential):
        return {"Authorization": "Bearer test-token"}

    async def no_account_id(_credential):
        return None

    monkeypatch.setattr(provider, "get_auth_header", auth_header)
    monkeypatch.setattr(provider, "get_account_id", no_account_id)

    async def invoke():
        response = await provider.acompletion(
            client,
            credential_identifier="credential.json",
            model="codex/gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            _llm_trace=trace,
        )
        if stream:
            return [chunk async for chunk in response]
        return response

    asyncio.run(invoke())

    assert client.headers["Authorization"] == "Bearer test-token"
    expected_metadata = {"boundary": "chat_via_responses"}
    if not stream:
        expected_metadata["attempt"] = 1
    assert trace.header_events == [
        ("provider_request", list(client.headers.items()), None, expected_metadata),
        ("provider_response", _RESPONSE_RAW_HEADERS, 200, expected_metadata),
    ]
