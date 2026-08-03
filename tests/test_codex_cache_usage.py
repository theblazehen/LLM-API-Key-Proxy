import asyncio
import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "rotator_library"
sys.path.insert(0, str(SRC_ROOT))

import pytest

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)
providers_package = types.ModuleType("rotator_library.providers")
providers_package.__path__ = [str(PACKAGE_ROOT / "providers")]
sys.modules.setdefault("rotator_library.providers", providers_package)


class EmptyResponseError(Exception):
    pass


class TransientQuotaError(Exception):
    pass


class StreamedAPIError(Exception):
    pass


class CredentialNeedsReauthError(Exception):
    pass


fake_error_handler = types.ModuleType("rotator_library.error_handler")
fake_error_handler.EmptyResponseError = EmptyResponseError
fake_error_handler.TransientQuotaError = TransientQuotaError
fake_error_handler.CredentialNeedsReauthError = CredentialNeedsReauthError
sys.modules.setdefault("rotator_library.error_handler", fake_error_handler)

fake_core_errors = types.ModuleType("rotator_library.core.errors")
fake_core_errors.StreamedAPIError = StreamedAPIError
sys.modules.setdefault("rotator_library.core.errors", fake_core_errors)


class _FakeUsage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_litellm = types.ModuleType("litellm")
fake_litellm.Usage = _FakeUsage
fake_litellm.ModelResponse = _FakeModelResponse
sys.modules.setdefault("litellm", fake_litellm)

from rotator_library.providers.codex_provider import CodexProvider


class _FakeStreamResponse:
    status_code = 200
    headers = {}

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for event in self._events:
            yield f"data: {json.dumps(event)}"

    async def aread(self):
        return b""


class _FakeStreamClient:
    def __init__(self, events):
        self._events = events

    def stream(self, *args, **kwargs):
        return _FakeStreamResponse(self._events)


def _events(input_details):
    return [
        {"type": "response.created", "response": {"id": "resp-cache"}},
        {"type": "response.output_text.delta", "delta": "ok"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-cache",
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 20,
                    "total_tokens": 140,
                    "input_tokens_details": input_details,
                },
            },
        },
    ]


def _details(usage):
    return getattr(usage, "prompt_tokens_details", None)


@pytest.fixture(autouse=True)
def _avoid_model_refresh(monkeypatch):
    import rotator_library.providers.codex_provider as codex_provider

    monkeypatch.setattr(
        codex_provider,
        "get_available_models",
        lambda: ["gpt-5.6-sol"],
    )


def test_codex_nonstream_preserves_cache_read_and_write_usage():
    async def run():
        return await CodexProvider()._non_stream_response(
            _FakeStreamClient(
                _events({"cached_tokens": 80, "cache_creation_tokens": 16})
            ),
            headers={},
            payload={},
            model="gpt-5.6-sol",
            reasoning_compat="default",
        )

    response = asyncio.run(run())

    assert _details(response.usage) == {
        "cached_tokens": 80,
        "cache_creation_tokens": 16,
    }


def test_codex_stream_preserves_cache_read_and_write_usage():
    async def run():
        stream = CodexProvider()._stream_response(
            _FakeStreamClient(
                _events({"cached_tokens": 80, "cache_creation_tokens": 16})
            ),
            headers={},
            payload={},
            model="gpt-5.6-sol",
            reasoning_compat="default",
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(run())

    assert _details(chunks[-1].usage) == {
        "cached_tokens": 80,
        "cache_creation_tokens": 16,
    }


def test_codex_usage_omits_absent_cache_counters():
    async def run():
        return await CodexProvider()._non_stream_response(
            _FakeStreamClient(_events({})),
            headers={},
            payload={},
            model="gpt-5.6-sol",
            reasoning_compat="default",
        )

    response = asyncio.run(run())

    assert _details(response.usage) is None
