import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy_app.proxy_auth import (
    bearer_token,
    is_proxy_key_environment,
    load_proxy_api_keys,
    resolve_proxy_identity,
)


def test_loads_legacy_named_and_json_proxy_keys():
    keys = load_proxy_api_keys(
        {
            "PROXY_API_KEY": "legacy-secret",
            "PROXY_API_KEY_USER_ALICE": "alice-secret",
            "PROXY_API_KEYS": '{"Bob Smith":"bob-secret"}',
            "OPENAI_API_KEY": "upstream-secret",
        }
    )

    assert keys == {
        "legacy-secret": "default",
        "alice-secret": "alice",
        "bob-secret": "bob-smith",
    }
    assert resolve_proxy_identity("alice-secret", keys).user == "alice"
    assert resolve_proxy_identity("wrong", keys) is None


def test_bearer_token_is_strict_about_scheme_and_value():
    assert bearer_token("Bearer secret") == "secret"
    assert bearer_token("bearer secret") == "secret"
    assert bearer_token("Basic secret") is None
    assert bearer_token("Bearer") is None


def test_proxy_key_environment_does_not_capture_provider_keys():
    assert is_proxy_key_environment("PROXY_API_KEY")
    assert is_proxy_key_environment("PROXY_API_KEYS")
    assert is_proxy_key_environment("PROXY_API_KEY_USER_ALICE")
    assert not is_proxy_key_environment("OPENAI_API_KEY")
