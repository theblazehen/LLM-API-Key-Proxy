import sys
import types
import uuid
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "rotator_library"
sys.path.insert(0, str(SRC_ROOT))

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
fake_litellm.__path__ = []

fake_litellm_core_utils = types.ModuleType("litellm.litellm_core_utils")
fake_litellm_token_counter = types.ModuleType(
    "litellm.litellm_core_utils.token_counter"
)
fake_litellm_token_counter.token_counter = lambda *args, **kwargs: 0

fake_litellm_exceptions = types.ModuleType("litellm.exceptions")
for name in (
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
    setattr(fake_litellm_exceptions, name, type(name, (Exception,), {}))

sys.modules.setdefault("litellm", fake_litellm)
sys.modules.setdefault("litellm.litellm_core_utils", fake_litellm_core_utils)
sys.modules.setdefault(
    "litellm.litellm_core_utils.token_counter", fake_litellm_token_counter
)
sys.modules.setdefault("litellm.exceptions", fake_litellm_exceptions)


_real_urlopen = urllib_request.urlopen


def _offline_urlopen(*_args, **_kwargs):
    raise urllib_error.URLError("network disabled in client metadata tests")


urllib_request.urlopen = _offline_urlopen
try:
    from rotator_library.providers.codex_provider import _build_codex_request_metadata
finally:
    urllib_request.urlopen = _real_urlopen


_METADATA_ENV_VARS = (
    "CODEX_INSTALLATION_ID",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
    "CODEX_WINDOW_ID",
)
_METADATA_FIELDS = (
    "x-codex-installation-id",
    "session_id",
    "thread_id",
    "x-codex-window-id",
)


def _clear_metadata_env(monkeypatch):
    for name in _METADATA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_seeded_metadata_is_deterministic_and_valid_uuid(monkeypatch):
    _clear_metadata_env(monkeypatch)

    first = _build_codex_request_metadata("conversation-cache-key")
    second = _build_codex_request_metadata("conversation-cache-key")

    assert first == second
    for field in ("x-codex-installation-id", "session_id", "thread_id"):
        assert str(uuid.UUID(first[field])) == first[field]
    # Official-CLI window shape: "{conversation_id}:{generation}".
    conversation, sep, generation = first["x-codex-window-id"].partition(":")
    assert sep == ":" and generation == "1"
    assert str(uuid.UUID(conversation)) == conversation


def test_seeded_metadata_matches_official_conversation_shape(monkeypatch):
    _clear_metadata_env(monkeypatch)

    metadata = _build_codex_request_metadata("conversation-shape")

    # session_id and thread_id both carry the stable conversation id; the
    # window id derives from it; the installation id stays distinct.
    assert metadata["session_id"] == metadata["thread_id"]
    assert metadata["x-codex-window-id"] == f"{metadata['session_id']}:1"
    assert metadata["x-codex-installation-id"] != metadata["session_id"]


def test_seeded_metadata_differs_for_each_conversation(monkeypatch):
    _clear_metadata_env(monkeypatch)

    first = _build_codex_request_metadata("conversation-one")
    second = _build_codex_request_metadata("conversation-two")

    assert all(first[field] != second[field] for field in _METADATA_FIELDS)


def test_metadata_without_seed_remains_random(monkeypatch):
    _clear_metadata_env(monkeypatch)

    assert _build_codex_request_metadata() != _build_codex_request_metadata()


def test_session_environment_override_has_priority_over_seed(monkeypatch):
    _clear_metadata_env(monkeypatch)
    monkeypatch.setenv("CODEX_SESSION_ID", "operator-selected-session")

    metadata = _build_codex_request_metadata("conversation-cache-key")

    assert metadata["session_id"] == "operator-selected-session"
