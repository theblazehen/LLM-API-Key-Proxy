# SPDX-License-Identifier: MIT
"""Inbound proxy API-key parsing and identity resolution."""

from __future__ import annotations

import hmac
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

_NAMED_KEY_PREFIX = "PROXY_API_KEY_USER_"
_USER_RE = re.compile(r"[^a-z0-9_.-]+")


@dataclass(frozen=True)
class ProxyIdentity:
    """A safe label for an authenticated proxy caller."""

    user: str


def _normalize_user(value: str) -> str:
    normalized = _USER_RE.sub("-", value.strip().lower()).strip("-")
    return normalized or "unknown"


def load_proxy_api_keys(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return ``secret -> user`` without logging or otherwise exposing secrets.

    ``PROXY_API_KEY`` remains supported as user ``default``. Named keys may be
    supplied as ``PROXY_API_KEY_USER_<NAME>`` variables or as a JSON object in
    ``PROXY_API_KEYS`` whose keys are user names and values are secrets.
    """

    source = os.environ if environ is None else environ
    keys: dict[str, str] = {}

    legacy = source.get("PROXY_API_KEY", "").strip()
    if legacy:
        keys[legacy] = "default"

    encoded = source.get("PROXY_API_KEYS", "").strip()
    if encoded:
        try:
            named = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("PROXY_API_KEYS must be a JSON object") from exc
        if not isinstance(named, dict):
            raise ValueError("PROXY_API_KEYS must be a JSON object")
        for user, secret in named.items():
            if not isinstance(user, str) or not isinstance(secret, str):
                raise ValueError("PROXY_API_KEYS names and values must be strings")
            if secret.strip():
                keys[secret.strip()] = _normalize_user(user)

    for name, value in source.items():
        if name.startswith(_NAMED_KEY_PREFIX) and value.strip():
            keys[value.strip()] = _normalize_user(name[len(_NAMED_KEY_PREFIX) :])

    return keys


def resolve_proxy_identity(
    presented_key: Optional[str], configured_keys: Mapping[str, str]
) -> Optional[ProxyIdentity]:
    """Resolve a presented secret using constant-time comparisons."""

    if not presented_key:
        return None
    matched_user: Optional[str] = None
    for secret, user in configured_keys.items():
        if hmac.compare_digest(presented_key, secret):
            matched_user = user
    return ProxyIdentity(matched_user) if matched_user is not None else None


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract a case-insensitive Bearer token, rejecting other schemes."""

    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def is_proxy_key_environment(name: str) -> bool:
    """Return whether an environment variable contains inbound proxy keys."""

    return name in {"PROXY_API_KEY", "PROXY_API_KEYS"} or name.startswith(
        _NAMED_KEY_PREFIX
    )
