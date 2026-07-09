import json
import asyncio
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "rotator_library"
sys.path.insert(0, str(SRC_ROOT))

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

providers_package = types.ModuleType("rotator_library.providers")
providers_package.__path__ = [str(PACKAGE_ROOT / "providers")]
sys.modules.setdefault("rotator_library.providers", providers_package)

utilities_package = types.ModuleType("rotator_library.providers.utilities")
utilities_package.__path__ = [str(PACKAGE_ROOT / "providers" / "utilities")]
sys.modules.setdefault("rotator_library.providers.utilities", utilities_package)

class EmptyResponseError(Exception):
    def __init__(self, provider, model, message=""):
        self.provider = provider
        self.model = model
        self.message = message or f"Empty response from {provider}/{model}"
        super().__init__(self.message)


class TransientQuotaError(Exception):
    pass


class StreamedAPIError(Exception):
    pass


class CredentialNeedsReauthError(Exception):
    pass


class ClassifiedError:
    pass


def classify_error(error):
    return ClassifiedError()


def mask_credential(credential, style="default"):
    return credential



fake_error_handler = types.ModuleType("rotator_library.error_handler")
fake_error_handler.CredentialNeedsReauthError = CredentialNeedsReauthError
fake_error_handler.EmptyResponseError = EmptyResponseError
fake_error_handler.TransientQuotaError = TransientQuotaError
fake_error_handler.ClassifiedError = ClassifiedError
fake_error_handler.classify_error = classify_error
fake_error_handler.mask_credential = mask_credential
sys.modules.setdefault("rotator_library.error_handler", fake_error_handler)

core_package = types.ModuleType("rotator_library.core")
core_package.__path__ = [str(PACKAGE_ROOT / "core")]
sys.modules.setdefault("rotator_library.core", core_package)

fake_core_errors = types.ModuleType("rotator_library.core.errors")
fake_core_errors.StreamedAPIError = StreamedAPIError
sys.modules.setdefault("rotator_library.core.errors", fake_core_errors)


class _FakeModelResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeMessage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeChoices:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeUsage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_litellm = types.ModuleType("litellm")
fake_litellm.ModelResponse = _FakeModelResponse
fake_litellm.Message = _FakeMessage
fake_litellm.Choices = _FakeChoices
fake_litellm.Usage = _FakeUsage

fake_litellm_exceptions = types.ModuleType("litellm.exceptions")
for _name in (
    "APIConnectionError",
    "RateLimitError",
    "ServiceUnavailableError",
    "AuthenticationError",
    "InvalidRequestError",
    "BadRequestError",
    "OpenAIError",
    "InternalServerError",
    "Timeout",
    "ContextWindowExceededError",
):
    setattr(fake_litellm_exceptions, _name, type(_name, (Exception,), {}))

sys.modules.setdefault("litellm", fake_litellm)
sys.modules.setdefault("litellm.exceptions", fake_litellm_exceptions)



import pytest

from rotator_library.error_handler import EmptyResponseError
from rotator_library.providers import codex_provider
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.ollama_cloud_provider import OllamaCloudProvider
from rotator_library.usage.manager import UsageManager


class FakeStreamResponse:
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


class FakeStreamClient:
    def __init__(self, events):
        self._events = events

    def stream(self, *args, **kwargs):
        return FakeStreamResponse(self._events)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [
                {"type": "text", "text": "Describe this"},
                {"type": "input_text", "text": " in one sentence."},
            ],
            "Describe this\n in one sentence.",
        ),
        (
            [
                {"type": "text", "text": "Look at this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {"type": "input_image", "image_url": "data:image/png;base64,def"},
                {"type": "text", "text": "Done."},
                {"type": "tool_use", "id": "ignored"},
            ],
            "Look at this:\nDone.",
        ),
    ],
)
def test_ollama_cloud_message_array_content_is_converted_to_string(content, expected):
    provider = OllamaCloudProvider()

    converted = provider._to_ollama_messages(
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "already a string"},
        ]
    )

    assert converted[0]["content"] == expected
    assert isinstance(converted[0]["content"], str)
    assert converted[1]["content"] == "already a string"


def test_codex_stream_completed_without_output_raises_empty_response_error():
    async def run_stream():
        provider = CodexProvider()
        client = FakeStreamClient(
            [
                {"type": "response.created", "response": {"id": "resp-empty"}},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-empty",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    },
                },
            ]
        )

        stream = provider._stream_response(
            client=client,
            headers={},
            payload={},
            model="gpt-5",
            reasoning_compat="think-tags",
        )

        async for _ in stream:
            pass

    with pytest.raises(EmptyResponseError):
        asyncio.run(run_stream())


def test_codex_non_stream_completed_without_output_raises_empty_response_error():
    async def run_response():
        provider = CodexProvider()
        client = FakeStreamClient(
            [
                {"type": "response.created", "response": {"id": "resp-empty"}},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-empty",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    },
                },
            ]
        )

        await provider._non_stream_response(
            client=client,
            headers={},
            payload={},
            model="gpt-5",
            reasoning_compat="think-tags",
        )

    with pytest.raises(EmptyResponseError):
        asyncio.run(run_response())


def test_codex_context_window_comes_from_codex_model_metadata():
    codex_provider._models_cache = {
        "base_models": ["gpt-5.5"],
        "reasoning_efforts": {},
        "fast_models": {"gpt-5.5"},
        "model_limits": {
            "gpt-5.5": {
                "context_window": 272000,
                "max_output": 128000,
            }
        },
    }
    codex_provider._models_cache_time = time.time()

    provider = CodexProvider()

    assert provider.get_model_context_window("codex/gpt-5.5") == 272000
    assert provider.get_model_context_window("codex/gpt-5.5-fast") == 272000
    assert provider.get_model_context_window("codex/gpt-5.5:xhigh") == 272000


def test_codex_percent_quota_snapshots_are_exposed_in_group_stats_without_requests():
    credential = "/credentials/codex-account.json"
    primary_reset = 1_800_000_000
    secondary_reset = 1_800_604_800

    async def store_snapshots_and_get_stats():
        manager = UsageManager(provider="codex")
        await manager.initialize([credential])
        provider = CodexProvider()

        stored = await provider._store_baselines_to_usage_manager(
            {
                credential: {
                    "status": "success",
                    "primary": {
                        "used_percent": 37.5,
                        "remaining_percent": 62.5,
                        "remaining_fraction": 0.625,
                        "window_minutes": 300,
                        "reset_at": primary_reset,
                        "is_exhausted": False,
                    },
                    "secondary": {
                        "used_percent": 81.25,
                        "remaining_percent": 18.75,
                        "remaining_fraction": 0.1875,
                        "window_minutes": 10_080,
                        "reset_at": secondary_reset,
                        "is_exhausted": False,
                    },
                }
            },
            manager,
            force=True,
        )
        return stored, await manager.get_stats_for_endpoint()

    stored, stats = asyncio.run(store_snapshots_and_get_stats())

    assert stored == 2
    credential_stats = next(iter(stats["credentials"].values()))
    expected_windows = {
        "5h-limit": {
            "used_percent": 37.5,
            "remaining_percent": 62.5,
            "window_minutes": 300,
            "reset_at": primary_reset,
            "quota_source": "codex",
        },
        "weekly-limit": {
            "used_percent": 81.25,
            "remaining_percent": 18.75,
            "window_minutes": 10_080,
            "reset_at": secondary_reset,
            "quota_source": "codex",
        },
    }
    for group_name, expected in expected_windows.items():
        window = next(
            iter(credential_stats["group_usage"][group_name]["windows"].values())
        )
        assert {field: window[field] for field in expected} == expected
        assert window["request_count"] == 0
