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

client_package = types.ModuleType("rotator_library.client")
client_package.__path__ = [str(PACKAGE_ROOT / "client")]
sys.modules.setdefault("rotator_library.client", client_package)

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
fake_core_errors.CredentialNeedsReauthError = CredentialNeedsReauthError
for _name in ("NoAvailableKeysError", "PreRequestCallbackError", "RequestErrorAccumulator"):
    setattr(fake_core_errors, _name, type(_name, (Exception,), {}))
fake_core_errors.ClassifiedError = ClassifiedError
fake_core_errors.classify_error = classify_error
fake_core_errors.should_rotate_on_error = lambda error: False
fake_core_errors.should_retry_same_key = lambda error: False
fake_core_errors.mask_credential = mask_credential
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
fake_litellm.EmbeddingResponse = type("EmbeddingResponse", (), {})
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
from rotator_library.client.executor import RequestExecutor
from rotator_library.client.streaming import StreamingHandler
from rotator_library.client.models import ModelResolver


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


def test_api_equivalent_cost_opt_in_overrides_provider_skip_and_uses_cache_rates(
    monkeypatch,
):
    class ApiEquivalentProvider:
        skip_cost_calculation = True
        calculate_api_equivalent_cost = True

        @staticmethod
        def get_api_equivalent_model(model):
            return "openai/gpt-priced"

    class PricingRegistry:
        @staticmethod
        def compute_cost(model, input_tokens, output_tokens, cache_read, cache_write):
            assert model == "openai/gpt-priced"
            assert (input_tokens, output_tokens, cache_read, cache_write) == (
                70,
                25,
                30,
                4,
            )
            return input_tokens * 0.01 + cache_read * 0.002 + output_tokens * 0.03

    import rotator_library.model_info_service as model_info_service

    monkeypatch.setattr(
        model_info_service, "get_model_info_service", lambda: PricingRegistry()
    )
    executor = RequestExecutor.__new__(RequestExecutor)
    executor._plugins = {"codex": ApiEquivalentProvider()}
    executor._plugin_instances = {}
    response = _FakeModelResponse(
        usage=_FakeUsage(
            prompt_tokens=100,
            completion_tokens=25,
            cache_read_tokens=30,
            cache_creation_tokens=4,
        )
    )

    assert executor._calculate_cost("codex", "codex/gpt-alias", response) == 1.51


def test_stream_cost_is_zero_when_registry_and_litellm_have_no_pricing(monkeypatch):
    class MissingPricingRegistry:
        @staticmethod
        def compute_cost(*args):
            return None

    import rotator_library.model_info_service as model_info_service

    monkeypatch.setattr(
        model_info_service, "get_model_info_service", lambda: MissingPricingRegistry()
    )
    monkeypatch.setattr(fake_litellm, "get_model_info", lambda model: {}, raising=False)

    assert StreamingHandler()._calculate_stream_cost("ollama_cloud/unknown", 9, 3) == 0.0


def test_completion_context_uses_concrete_model_resolved_from_alias():
    resolver = ModelResolver(provider_plugins={})

    assert resolver.resolve_model_id("alias/glm", "ollama_cloud") == "ollama_cloud/glm-5.2"


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
