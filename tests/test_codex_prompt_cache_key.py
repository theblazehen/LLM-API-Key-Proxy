import asyncio
import sys
import types
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "rotator_library"
sys.path.insert(0, str(SRC_ROOT))

# Match the narrow import stubs used by test_provider_regressions.py.  The
# request-path tests do not need the package's production initialization.
rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

providers_package = types.ModuleType("rotator_library.providers")
providers_package.__path__ = [str(PACKAGE_ROOT / "providers")]
providers_package.PROVIDER_PLUGINS = {}
sys.modules.setdefault("rotator_library.providers", providers_package)

client_package = types.ModuleType("rotator_library.client")
client_package.__path__ = [str(PACKAGE_ROOT / "client")]
sys.modules.setdefault("rotator_library.client", client_package)

utilities_package = types.ModuleType("rotator_library.providers.utilities")
utilities_package.__path__ = [str(PACKAGE_ROOT / "providers" / "utilities")]
sys.modules.setdefault("rotator_library.providers.utilities", utilities_package)


class EmptyResponseError(Exception):
    pass


class TransientQuotaError(Exception):
    pass


class StreamedAPIError(Exception):
    pass


class CredentialNeedsReauthError(Exception):
    pass


class ClassifiedError:
    pass


class NoAvailableKeysError(Exception):
    pass


class PreRequestCallbackError(Exception):
    pass


class RequestErrorAccumulator:
    pass


def classify_error(_error):
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
fake_core_errors.NoAvailableKeysError = NoAvailableKeysError
fake_core_errors.PreRequestCallbackError = PreRequestCallbackError
fake_core_errors.RequestErrorAccumulator = RequestErrorAccumulator
fake_core_errors.ClassifiedError = ClassifiedError
fake_core_errors.classify_error = classify_error
fake_core_errors.should_rotate_on_error = lambda _error: False
fake_core_errors.should_retry_same_key = lambda _error: False
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
fake_litellm.set_verbose = False
fake_litellm.drop_params = False

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


# codex_provider populates its model cache at import time.  Keep this test
# self-contained and guarantee that importing it cannot contact the network.
_real_urlopen = urllib_request.urlopen


def _offline_urlopen(*_args, **_kwargs):
    raise urllib_error.URLError("network disabled in prompt-cache-key tests")


urllib_request.urlopen = _offline_urlopen
try:
    from rotator_library.client import rotating_client
    from rotator_library.providers.codex_provider import CodexProvider
finally:
    urllib_request.urlopen = _real_urlopen


def _derive_codex_prompt_cache_key(kwargs, request):
    """Resolve the production helper at assertion time, not during collection."""
    return rotating_client._derive_codex_prompt_cache_key(kwargs, request)


class _Trace:
    def __init__(self, proxy_user: str, session_id: str):
        self.proxy_user = proxy_user
        self.session_id = session_id


class _Request:
    def __init__(self, proxy_user: str = "omp-user", session_id: str = "session-1"):
        self.state = types.SimpleNamespace(llm_trace=_Trace(proxy_user, session_id))
        self.headers = {}


def test_derive_codex_prompt_cache_key_preserves_caller_value():
    caller_key = "caller-selected-cache-key"

    assert _derive_codex_prompt_cache_key(
        {"prompt_cache_key": caller_key}, _Request()
    ) == caller_key


def test_derive_codex_prompt_cache_key_is_stable_for_trace_session():
    first = _derive_codex_prompt_cache_key({}, _Request("omp-user", "session-a"))
    second = _derive_codex_prompt_cache_key({}, _Request("omp-user", "session-a"))

    assert first == second
    assert first.startswith("omp-")
    assert len(first) == 64


def test_derive_codex_prompt_cache_key_changes_for_different_trace_sessions():
    first = _derive_codex_prompt_cache_key({}, _Request("omp-user", "session-a"))
    second = _derive_codex_prompt_cache_key({}, _Request("omp-user", "session-b"))

    assert first != second


def test_derive_codex_prompt_cache_key_never_exposes_prompt_or_authorization():
    prompt = "private prompt content must not become a cache key"
    bearer = "Bearer secret-access-token"
    request = _Request("omp-user", "session-a")
    request.headers["authorization"] = bearer
    key = _derive_codex_prompt_cache_key(
        {
            "messages": [{"role": "user", "content": prompt}],
            "extra_headers": {"Authorization": bearer},
        },
        request,
    )

    assert key.startswith("omp-")
    assert prompt not in key
    assert bearer not in key
    assert "Bearer" not in key


@pytest.mark.parametrize(
    "key_factory",
    [
        lambda request: _derive_codex_prompt_cache_key({}, request),
        lambda _request: "caller-selected-cache-key",
    ],
    ids=["derived", "caller"],
)
def test_codex_chat_completion_sends_prompt_cache_key_unchanged(key_factory, monkeypatch):
    async def exercise():
        request = _Request("omp-user", "session-a")
        prompt_cache_key = key_factory(request)
        provider = CodexProvider()
        captured = {}

        async def get_auth_header(_credential):
            return {}

        async def get_account_id(_credential):
            return None

        async def capture_non_stream(*args):
            captured["payload"] = args[2]
            return {"id": "resp_test", "output": []}

        monkeypatch.setattr(provider, "get_auth_header", get_auth_header)
        monkeypatch.setattr(provider, "get_account_id", get_account_id)
        monkeypatch.setattr(provider, "_non_stream_with_retry", capture_non_stream)

        await provider.acompletion(
            object(),
            model="codex/gpt-5.6-sol",
            messages=[{"role": "user", "content": "not used for cache identity"}],
            stream=False,
            credential_identifier="credential.json",
            prompt_cache_key=prompt_cache_key,
        )

        assert captured["payload"]["prompt_cache_key"] == prompt_cache_key

    asyncio.run(exercise())
