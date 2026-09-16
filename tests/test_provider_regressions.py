import json
import asyncio
import base64
import sys
import time
import types
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx

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
fake_litellm_types = types.ModuleType("litellm.types")
fake_litellm_utils = types.ModuleType("litellm.types.utils")
fake_litellm_utils.ModelResponseStream = _FakeModelResponse
sys.modules.setdefault("litellm.types", fake_litellm_types)
sys.modules.setdefault("litellm.types.utils", fake_litellm_utils)



import pytest

from rotator_library.error_handler import EmptyResponseError
from rotator_library.providers import codex_provider
from rotator_library.providers import openai_oauth_base
from rotator_library.providers.utilities import codex_quota_tracker
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.ollama_cloud_provider import OllamaCloudProvider
from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.usage.manager import UsageManager
from rotator_library.client.executor import RequestExecutor
from rotator_library.client.streaming import StreamingHandler
from rotator_library.client.models import ModelResolver


class FakeStreamResponse:
    status_code = 200
    headers = {}

    def __init__(self, events):
        self._events = events
        self.headers = httpx.Headers()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for event in self._events:
            yield f"data: {json.dumps(event)}"

    async def aread(self):
        return b""


class FakeHTTPResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("GET", "https://example.invalid"),
                response=httpx.Response(self.status_code),
            )


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):
        return self.responses.pop(0)

    async def post(self, *args, **kwargs):
        return self.responses.pop(0)


class FakeStreamClient:
    def __init__(self, events):
        self._events = events

    def stream(self, *args, **kwargs):
        return FakeStreamResponse(self._events)


def _jwt(claims):
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{encoded}.signature"


async def _capture_codex_completion_payload(monkeypatch, **kwargs):
    provider = CodexProvider()
    requests = []

    async def get_auth_header(_credential):
        return {"Authorization": "Bearer test-token"}

    async def get_account_id(_credential):
        return None

    def handle(request):
        assert request.method == "POST"
        assert str(request.url) == codex_provider.CODEX_RESPONSES_ENDPOINT
        requests.append(json.loads(request.content))
        events = [
            {"type": "response.output_text.delta", "delta": "{}"},
            {"type": "response.completed", "response": {"id": "resp_format"}},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    monkeypatch.setattr(provider, "get_auth_header", get_auth_header)
    monkeypatch.setattr(provider, "get_account_id", get_account_id)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await provider.acompletion(
            client,
            model="codex/gpt-5.6-sol",
            messages=[{"role": "user", "content": "Return JSON."}],
            **kwargs,
        )
        if kwargs.get("stream"):
            async for _chunk in result:
                pass
    assert len(requests) == 1
    return requests[0]


@pytest.mark.parametrize("stream", [False, True])
def test_codex_acompletion_preserves_strict_response_schema(monkeypatch, stream):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "strict_result",
            "description": "A closed result with a referenced child",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Child"}},
                "required": ["child"],
                "additionalProperties": False,
                "$defs": {
                    "Child": {
                        "type": "object",
                        "properties": {"value": {"type": ["string", "null"]}},
                        "required": ["value"],
                        "additionalProperties": False,
                    }
                },
            },
        },
    }
    original = deepcopy(response_format)
    payload = asyncio.run(_capture_codex_completion_payload(
        monkeypatch, response_format=response_format, stream=stream
    ))
    assert payload["text"] == {
        "verbosity": "medium",
        "format": {"type": "json_schema", **original["json_schema"]},
    }
    assert "response_format" not in payload
    assert response_format == original


@pytest.mark.parametrize("kwargs", [{}, {"response_format": None}])
def test_codex_acompletion_without_response_format_is_unchanged(monkeypatch, kwargs):
    payload = asyncio.run(_capture_codex_completion_payload(monkeypatch, **kwargs))
    assert payload["text"] == {"verbosity": "medium"}
    assert "response_format" not in payload


@pytest.mark.parametrize("format_type", ["text", "json_object"])
def test_codex_acompletion_maps_simple_response_formats(monkeypatch, format_type):
    response_format = {"type": format_type}
    payload = asyncio.run(_capture_codex_completion_payload(
        monkeypatch, response_format=response_format
    ))
    assert payload["text"] == {"verbosity": "medium", "format": response_format}
    assert response_format == {"type": format_type}


@pytest.mark.parametrize("optional", [{}, {"strict": False}, {"strict": None}, {"type": "json_schema"}])
def test_codex_acompletion_preserves_optional_schema_fields(monkeypatch, optional):
    specification = {"name": "result", "schema": {}, **optional}
    payload = asyncio.run(_capture_codex_completion_payload(
        monkeypatch, response_format={"type": "json_schema", "json_schema": specification}
    ))
    assert payload["text"]["format"] == {"type": "json_schema", **specification}
    assert "description" not in payload["text"]["format"]


@pytest.mark.parametrize("response_format", [
    "json_object", [], {}, {"type": []}, {"type": "xml"},
    {"type": "text", "schema": {}},
    {"type": "json_schema"},
    {"type": "json_schema", "json_schema": []},
    *[
        {"type": "json_schema", "json_schema": specification}
        for specification in [
            {}, {"name": "result"}, {"name": "result", "schema": []},
            {"name": "", "schema": {}}, {"name": 1, "schema": {}},
            {"name": "bad name", "schema": {}}, {"name": "x" * 65, "schema": {}},
            {"name": "result", "schema": {}, "strict": "true"},
            {"name": "result", "schema": {}, "strict": 1},
            {"name": "result", "schema": {}, "description": []},
            {"name": "result", "schema": {}, "type": "text"},
            {"name": "result", "schema": {}, "unsupported": True},
        ]
    ],
])
def test_codex_acompletion_rejects_invalid_response_formats_before_io(monkeypatch, response_format):
    async def exercise():
        provider = CodexProvider()
        original = deepcopy(response_format)

        async def unexpected_auth(_credential):
            pytest.fail("invalid format reached credential I/O")

        def unexpected_request(_request):
            pytest.fail("invalid format reached upstream")

        monkeypatch.setattr(provider, "get_auth_header", unexpected_auth)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
            with pytest.raises(ValueError, match="response_format"):
                await provider.acompletion(client, response_format=response_format)
        assert response_format == original

    asyncio.run(exercise())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("tool_choice, expected", [
    ("auto", "auto"),
    ("none", "none"),
    ("required", "required"),
    ({"type": "function", "function": {"name": "report_result"}},
     {"type": "function", "name": "report_result"}),
])
def test_codex_acompletion_preserves_tool_choice(
    monkeypatch, stream, with_tools, tool_choice, expected
):
    original = deepcopy(tool_choice)
    kwargs = {"stream": stream, "tool_choice": tool_choice}
    if with_tools:
        kwargs["tools"] = [{"type": "function", "function": {
            "name": "report_result", "parameters": {"type": "object", "properties": {}}
        }}]
    payload = asyncio.run(_capture_codex_completion_payload(monkeypatch, **kwargs))
    assert payload["tool_choice"] == expected
    if with_tools:
        assert payload["tools"][0]["name"] == "report_result"
    assert tool_choice == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_codex_acompletion_default_tool_choice(monkeypatch, stream, with_tools):
    kwargs = {"stream": stream}
    if with_tools:
        kwargs["tools"] = [{"type": "function", "function": {
            "name": "report_result", "parameters": {"type": "object", "properties": {}}
        }}]
    payload = asyncio.run(_capture_codex_completion_payload(monkeypatch, **kwargs))
    if with_tools:
        assert payload["tool_choice"] == "auto"
    else:
        assert "tool_choice" not in payload


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", [
    None, False, 1, [], "", "any", {}, {"type": []}, {"type": "custom"},
    {"type": "function"}, {"type": "function", "function": []},
    {"type": "function", "name": "report_result"},
    {"type": "function", "function": {"name": "report_result"}, "extra": True},
    *[
        {"type": "function", "function": function}
        for function in [
            {}, {"name": None}, {"name": 1}, {"name": []}, {"name": ""},
            {"name": "bad name"}, {"name": "x" * 65},
            {"name": "report_result", "extra": True},
        ]
    ],
])
def test_codex_acompletion_rejects_invalid_tool_choice_before_io(
    monkeypatch, stream, tool_choice
):
    async def exercise():
        provider = CodexProvider()
        original = deepcopy(tool_choice)

        async def unexpected_auth(_credential):
            pytest.fail("invalid tool choice reached credential I/O")

        def unexpected_request(_request):
            pytest.fail("invalid tool choice reached upstream")

        monkeypatch.setattr(provider, "get_auth_header", unexpected_auth)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
            with pytest.raises(ValueError, match="tool_choice"):
                await provider.acompletion(client, tool_choice=tool_choice, stream=stream)
        assert tool_choice == original

    asyncio.run(exercise())


def test_codex_device_login_uses_cli_flow_and_persists_tokens(monkeypatch, tmp_path):
    requests = []
    id_token = _jwt(
        {
            "email": "device@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account-device"
            },
        }
    )
    access_token = _jwt(
        {"https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}}
    )

    class Response:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        @property
        def is_success(self):
            return 200 <= self.status_code < 300

        def json(self):
            return self._data

        def raise_for_status(self):
            if not self.is_success:
                request = httpx.Request("POST", "https://auth.openai.com")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError("failed", request=request, response=response)

    responses = iter(
        [
            Response(
                200,
                {
                    "device_auth_id": "device-id",
                    "user_code": "ABCD-EFGH",
                    "interval": "1",
                },
            ),
            Response(403, {}),
            Response(
                200,
                {
                    "authorization_code": "authorization-code",
                    "code_challenge": "challenge",
                    "code_verifier": "verifier",
                },
            ),
            Response(
                200,
                {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                    "id_token": id_token,
                    "expires_in": 3600,
                },
            ),
        ]
    )


def test_codex_setup_credential_device_login_flow(monkeypatch, tmp_path):
    requests = []
    id_token = _jwt(
        {
            "email": "setup-device@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account-setup-device"
            },
        }
    )
    access_token = _jwt(
        {"https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}}
    )

    class Response:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        @property
        def is_success(self):
            return 200 <= self.status_code < 300

        def json(self):
            return self._data

        def raise_for_status(self):
            if not self.is_success:
                request = httpx.Request("POST", "https://auth.openai.com")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError("failed", request=request, response=response)

    responses = iter(
        [
            Response(
                200,
                {
                    "device_auth_id": "device-id-setup",
                    "user_code": "SETUP-1234",
                    "interval": "1",
                },
            ),
            Response(
                200,
                {
                    "authorization_code": "auth-code-setup",
                    "code_challenge": "challenge-setup",
                    "code_verifier": "verifier-setup",
                },
            ),
            Response(
                200,
                {
                    "access_token": access_token,
                    "refresh_token": "refresh-token-setup",
                    "id_token": id_token,
                    "expires_in": 3600,
                },
            ),
        ]
    )

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return next(responses)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(openai_oauth_base.httpx, "AsyncClient", Client)
    monkeypatch.setattr(openai_oauth_base.asyncio, "sleep", no_sleep)
    provider = CodexProvider()

    result = asyncio.run(
        provider.setup_credential(base_dir=tmp_path, login_method="device")
    )

    assert result.success is True
    assert result.email == "setup-device@example.com"
    assert result.account_id == "account-setup-device"
    assert Path(result.file_path).exists()
    saved_data = json.loads(Path(result.file_path).read_text())
    assert saved_data["refresh_token"] == "refresh-token-setup"
    assert saved_data["account_id"] == "account-setup-device"
    assert saved_data["_proxy_metadata"]["email"] == "setup-device@example.com"


def test_codex_unauthorized_forces_refresh(monkeypatch):
    provider = CodexProvider()
    calls = []

    async def refresh(path):
        calls.append(path)
        return {"Authorization": "Bearer refreshed"}

    monkeypatch.setattr(provider, "refresh_auth_header_after_unauthorized", refresh)

    asyncio.run(provider._recover_unauthorized_credential("codex_oauth_2.json", 401))

    assert calls == ["codex_oauth_2.json"]


def test_codex_non_auth_error_does_not_refresh(monkeypatch):
    provider = CodexProvider()

    async def unexpected_refresh(_):
        raise AssertionError("refresh should not run")

    monkeypatch.setattr(
        provider, "refresh_auth_header_after_unauthorized", unexpected_refresh
    )

    asyncio.run(provider._recover_unauthorized_credential("codex_oauth_2.json", 429))


def _codex_quota(path, remaining, reset_at, fetched_at):
    return codex_quota_tracker.CodexQuotaSnapshot(
        credential_path=path,
        identifier=path,
        plan_type="pro",
        primary=codex_quota_tracker.RateLimitWindow(
            used_percent=100 - remaining,
            remaining_percent=remaining,
            window_minutes=10_080,
            reset_at=reset_at,
        ),
        secondary=None,
        credits=None,
        fetched_at=fetched_at,
        status="success",
        error=None,
        account_id=f"account-{path}",
        source="api",
    )


def _codex_resets(now, expires_in=86400):
    return codex_quota_tracker.ResetCreditsSnapshot(
        available_count=1,
        credits=(
            codex_quota_tracker.ResetCredit(
                id="reset-1",
                reset_type="codex_rate_limits",
                status="available",
                granted_at=now - 100,
                expires_at=now + expires_in,
                title="Reset",
                description=None,
            ),
        ),
        fetched_at=now,
        status="success",
    )


def test_codex_reset_policy_never_redeems_while_quota_remains(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(codex_quota_tracker.time, "time", lambda: now)
    provider = CodexProvider()
    provider._quota_cache["a"] = _codex_quota("a", 20, now + 4 * 86400, now)
    provider._reset_credits_cache["a"] = _codex_resets(now)

    action, reason, _ = provider._reset_policy("a", ["a"])

    assert action == "drain"
    assert "quota remains" in reason


def test_codex_reset_policy_preserves_credit_when_alternative_is_available(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(codex_quota_tracker.time, "time", lambda: now)
    provider = CodexProvider()
    provider._quota_cache["a"] = _codex_quota("a", 0, now + 4 * 86400, now)
    provider._quota_cache["b"] = _codex_quota("b", 50, now + 4 * 86400, now)
    provider._reset_credits_cache["a"] = _codex_resets(now)

    action, reason, _ = provider._reset_policy("a", ["a", "b"])

    assert action == "wait"
    assert "another account" in reason


def test_codex_reset_policy_redeems_only_when_fleet_blocked(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(codex_quota_tracker.time, "time", lambda: now)
    monkeypatch.setattr(codex_quota_tracker, "RESET_AUTO_MODE", "automatic")
    provider = CodexProvider()
    provider._quota_cache["a"] = _codex_quota("a", 0, now + 4 * 86400, now)
    provider._quota_cache["b"] = _codex_quota("b", 0, now + 4 * 86400, now)
    provider._reset_credits_cache["a"] = _codex_resets(now)

    action, reason, credit_id = provider._reset_policy("a", ["a", "b"])

    assert action == "redeem"
    assert "all accounts" in reason
    assert credit_id == "reset-1"


def test_codex_reset_policy_waits_for_imminent_credit_expiry(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(codex_quota_tracker.time, "time", lambda: now)
    monkeypatch.setattr(codex_quota_tracker, "RESET_AUTO_MODE", "automatic")
    provider = CodexProvider()
    provider._quota_cache["a"] = _codex_quota("a", 0, now + 4 * 86400, now)
    provider._reset_credits_cache["a"] = _codex_resets(now, expires_in=60)

    action, reason, _ = provider._reset_policy("a", ["a"])

    assert action == "wait"
    assert reason == "credit expiry is imminent; automatic refill unverified"


def test_codex_fetch_reset_credits_sorts_available_details(monkeypatch):
    provider = CodexProvider()

    async def headers(_):
        return {"Authorization": "Bearer token"}

    responses = [
        FakeHTTPResponse(
            200,
            {
                "available_count": 2,
                "credits": [
                    {
                        "id": "later",
                        "reset_type": "codex_rate_limits",
                        "status": "Available",
                        "expires_at": "2026-07-22T00:00:00Z",
                    },
                    {
                        "id": "earlier",
                        "reset_type": "codex_rate_limits",
                        "status": "Available",
                        "expires_at": "2026-07-21T00:00:00Z",
                    },
                ],
            },
        )
    ]
    monkeypatch.setattr(provider, "_account_headers", headers)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeHTTPClient(responses))

    snapshot = asyncio.run(provider.fetch_reset_credits("credential.json"))

    assert snapshot.status == "success"
    assert snapshot.available_count == 2
    assert [credit.id for credit in snapshot.credits] == ["earlier", "later"]
    assert snapshot.next_expiry_at == pytest.approx(1784592000.0)


def test_codex_redeem_reset_credit_uses_requested_id_and_idempotency(monkeypatch):
    provider = CodexProvider()
    now = time.time()
    provider._reset_credits_cache["credential.json"] = _codex_resets(now)
    requests = []

    async def quota(_):
        return _codex_quota("credential.json", 0, now + 86400, now)

    async def resets(_):
        snapshot = provider._reset_credits_cache["credential.json"]
        return snapshot

    async def headers(_):
        return {"Authorization": "Bearer token"}

    class Client(FakeHTTPClient):
        async def post(self, *args, **kwargs):
            requests.append(kwargs["json"])
            return FakeHTTPResponse(200, {"code": "Reset"})

    monkeypatch.setattr(provider, "fetch_quota_from_api", quota)
    monkeypatch.setattr(provider, "fetch_reset_credits", resets)
    monkeypatch.setattr(provider, "_account_headers", headers)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: Client([]))

    result = asyncio.run(
        provider.redeem_reset_credit(
            "credential.json",
            credit_id="reset-1",
            redeem_request_id="request-1",
        )
    )

    assert requests == [
        {"redeem_request_id": "request-1", "credit_id": "reset-1"}
    ]
    assert result["code"] == "Reset"
    assert result["redeem_request_id"] == "request-1"




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


def test_codex_default_reasoning_is_separate_from_visible_content():
    message = {"role": "assistant", "content": "visible output"}

    result = codex_provider._apply_reasoning_to_message(
        message,
        "internal summary",
        "",
        codex_provider.DEFAULT_REASONING_COMPAT,
    )

    assert result["content"] == "visible output"
    assert result["reasoning_summary"] == "internal summary"
    assert "<think>" not in result["content"]


def test_codex_non_stream_response_keeps_reasoning_out_of_visible_content():
    async def run_response():
        provider = CodexProvider()
        client = FakeStreamClient(
            [
                {"type": "response.created", "response": {"id": "resp-reasoning"}},
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "internal summary",
                },
                {"type": "response.output_text.delta", "delta": "visible output"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-reasoning",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                },
            ]
        )

        return await provider._non_stream_response(
            client=client,
            headers={},
            payload={},
            model="gpt-5.6-sol",
            reasoning_compat=codex_provider.DEFAULT_REASONING_COMPAT,
        )

    response = asyncio.run(run_response())
    message = response.choices[0]["message"]

    assert message["content"] == "visible output"
    assert message["reasoning_summary"] == "internal summary"
    assert "<think>" not in message["content"]


def test_codex_stream_wrapper_yields_before_upstream_completes():
    async def run_stream():
        provider = CodexProvider()
        upstream_can_finish = asyncio.Event()

        async def fake_stream_response(*args, **kwargs):
            yield _FakeModelResponse(
                choices=[
                    _FakeChoices(delta={"content": "first"}, finish_reason=None)
                ]
            )
            await upstream_can_finish.wait()
            yield _FakeModelResponse(
                choices=[_FakeChoices(delta={}, finish_reason="stop")]
            )

        provider._stream_response = fake_stream_response
        stream = provider._stream_with_retry(
            client=None,
            headers={},
            payload={},
            model="gpt-5.6-sol",
            reasoning_compat="think-tags",
        )

        first = await asyncio.wait_for(anext(stream), timeout=0.1)
        assert first.choices[0].delta["content"] == "first"

        upstream_can_finish.set()
        remaining = [chunk async for chunk in stream]
        assert remaining[0].choices[0].finish_reason == "stop"

    asyncio.run(run_stream())


def test_codex_stale_model_cache_returns_immediately_and_refreshes_in_background(
    monkeypatch,
):
    stale = {
        "base_models": ["gpt-5.6-sol"],
        "reasoning_efforts": {},
        "fast_models": set(),
        "model_limits": {},
    }
    refresh_started = []
    codex_provider._models_cache = stale
    codex_provider._models_cache_time = 0
    monkeypatch.setattr(
        codex_provider, "_start_models_refresh", lambda: refresh_started.append(True)
    )

    assert codex_provider._get_model_data() is stale
    assert refresh_started == [True]


def test_codex_failed_background_refresh_backs_off(monkeypatch):
    stale = {
        "base_models": ["gpt-5.6-sol"],
        "reasoning_efforts": {},
        "fast_models": set(),
        "model_limits": {},
    }
    codex_provider._models_cache = stale
    codex_provider._models_cache_time = 0
    codex_provider._models_refresh_in_progress = True
    monkeypatch.setattr(codex_provider, "_fetch_models_from_github", lambda: None)

    before = time.time()
    codex_provider._refresh_models_cache()

    assert codex_provider._models_cache is stale
    assert codex_provider._models_cache_time >= before
    assert codex_provider._models_refresh_in_progress is False


def test_codex_uses_model_specific_upstream_instruction():
    codex_provider._models_cache = {
        "base_models": ["gpt-5.6-sol"],
        "reasoning_efforts": {},
        "fast_models": {"gpt-5.6-sol"},
        "model_limits": {},
        "model_instructions": {
            "gpt-5.6-sol": "You are Codex, an agent based on GPT-5."
        },
    }
    codex_provider._models_cache_time = time.time()

    assert (
        codex_provider._get_model_instruction("gpt-5.6-sol")
        == "You are Codex, an agent based on GPT-5."
    )


def test_codex_56_supports_current_reasoning_levels():
    expected = {"low", "medium", "high", "xhigh", "max", "ultra"}

    assert codex_provider._FALLBACK_REASONING_EFFORTS["gpt-5.6-sol"] == expected
    assert expected <= codex_provider.REASONING_EFFORTS


def test_provider_interface_does_not_advertise_compact_api_by_default():
    assert ProviderInterface.supports_compact_api(object()) is False


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


def test_gpt_alias_routes_only_to_astra_low():
    resolver = ModelResolver(provider_plugins={})

    assert resolver.resolve_model_chain("alias/gpt") == ["codex/gpt-6-astra:low"]


def test_reviewer_alias_exposes_and_selects_both_review_models(monkeypatch):
    import rotator_library.client.models as models_mod
    resolver = ModelResolver(provider_plugins={})
    selections = iter(["alias/gemma-reviewer", "alias/deepseek-flash"])
    monkeypatch.setattr(
        models_mod.random, "choices",
        lambda choices, weights, k: [next(selections)],
    )

    assert "alias/reviewer" in resolver.get_alias_models()
    assert resolver.resolve_model_chain("alias/reviewer") == [
        "ollama_cloud/gemma4:31b-cloud"
    ]
    assert resolver.resolve_model_chain("alias/reviewer") == [
        "ollama_cloud/deepseek-v4-flash",
        "opencode_go/deepseek-v4-flash",
    ]


def test_reviewer_alias_weights_are_configurable(monkeypatch):
    resolver = ModelResolver(provider_plugins={})
    monkeypatch.setenv("ALIAS_REVIEWER_WEIGHTS", "alias/gemma-reviewer=100")

    assert resolver.resolve_model_chain("alias/reviewer") == [
        "ollama_cloud/gemma4:31b-cloud"
    ]


def test_codex_percent_quota_snapshots_are_exposed_in_group_stats_without_requests():
    credential = "/credentials/codex-account.json"
    primary_reset = 1_800_000_000
    secondary_reset = 1_800_604_800

    async def store_snapshots_and_get_stats():
        manager = UsageManager(provider="codex")
        await manager.initialize([credential])
        provider = CodexProvider()
        observed = []
        provider.set_quota_observer(observed.append)
        snapshot = replace(
            _codex_quota(credential, 18.75, secondary_reset, time.time()),
            primary=codex_quota_tracker.RateLimitWindow(
                used_percent=37.5,
                remaining_percent=62.5,
                window_minutes=300,
                reset_at=primary_reset,
            ),
            secondary=codex_quota_tracker.RateLimitWindow(
                used_percent=81.25,
                remaining_percent=18.75,
                window_minutes=10_080,
                reset_at=secondary_reset,
            ),
        )
        provider._publish_quota_snapshot(snapshot)

        stored = await provider._store_baselines_to_usage_manager(
            {credential: {"status": "success"}},
            manager,
            force=True,
        )
        assert observed == [snapshot]
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


def test_codex_weekly_primary_replaces_removed_short_window_and_resets_stale_state():
    credential = "/credentials/codex-account.json"
    weekly_reset = 1_800_604_800

    async def refresh_and_get_stats():
        manager = UsageManager(provider="codex")
        await manager.initialize([credential])
        provider = CodexProvider()

        state = next(iter(manager._states.values()))
        state.get_group_stats("5h-limit")
        state.get_group_stats("weekly-limit")
        from rotator_library.usage.types import FairCycleState

        global_cycle = FairCycleState(model_or_group="codex-global")
        state.fair_cycle["codex-global"] = global_cycle
        global_cycle.exhausted = True
        global_cycle.exhausted_reason = "quota_exceeded"

        observed = []
        provider.set_quota_observer(observed.append)
        snapshot = _codex_quota(credential, 53.0, weekly_reset, time.time())
        provider._publish_quota_snapshot(snapshot)
        stored = await provider._store_baselines_to_usage_manager(
            {credential: {"status": "success"}},
            manager,
            force=True,
        )
        assert observed == [snapshot]
        return stored, await manager.get_stats_for_endpoint()

    stored, stats = asyncio.run(refresh_and_get_stats())

    assert stored == 1
    credential_stats = next(iter(stats["credentials"].values()))
    assert "5h-limit" not in credential_stats["group_usage"]
    weekly = next(
        iter(credential_stats["group_usage"]["weekly-limit"]["windows"].values())
    )
    assert weekly["window_minutes"] == 10_080
    assert weekly["used_percent"] == 47.0
    assert credential_stats["fair_cycle"]["codex-global"]["exhausted"] is False
