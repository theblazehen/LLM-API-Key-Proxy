"""Focused contracts for native Codex Responses streaming.

These tests use no credentials or network. They describe the boundary between the
native Responses provider stream and the existing executor accounting/retry path.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.client.executor import RequestExecutor
from rotator_library.error_handler import classify_error
from rotator_library.providers.codex_provider import CodexProvider


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


class _ErrorResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.request = httpx.Request("POST", "https://example.invalid/responses")
        self.headers = {"content-type": "application/json"}

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
