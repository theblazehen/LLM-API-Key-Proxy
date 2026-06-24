# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenCode Go provider.

OpenCode Go exposes one API-key-backed model namespace, but models are split
across OpenAI-compatible chat completions and Anthropic-compatible messages
endpoints. Route each model through the matching LiteLLM provider while keeping
the public model prefix compatible with OpenCode's `opencode-go/<model>` IDs.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, List, Union

import httpx
import litellm

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


OPENCODE_GO_API_BASE = os.getenv(
    "OPENCODE_GO_API_BASE", "https://opencode.ai/zen/go/v1"
).rstrip("/")
OPENCODE_GO_ANTHROPIC_API_BASE = os.getenv(
    "OPENCODE_GO_ANTHROPIC_API_BASE", "https://opencode.ai/zen/go"
).rstrip("/")
PUBLIC_PROVIDER_PREFIX = "opencode-go"

OPENAI_COMPATIBLE_MODELS = {
    "glm-5.2",
    "glm-5.1",
    "kimi-k2.7",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "mimo-v2.5",
    "mimo-v2.5-pro",
}

ANTHROPIC_COMPATIBLE_MODELS = {
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
}

FALLBACK_MODELS = sorted(OPENAI_COMPATIBLE_MODELS | ANTHROPIC_COMPATIBLE_MODELS)


class OpencodeGoProvider(ProviderInterface):
    """API-key provider for OpenCode Go models."""

    provider_env_name: str = "opencode_go"
    skip_cost_calculation: bool = True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available OpenCode Go models, with documented models as fallback."""
        try:
            response = await client.get(
                f"{OPENCODE_GO_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [
                f"{PUBLIC_PROVIDER_PREFIX}/{model['id']}"
                for model in data
                if isinstance(model, dict) and model.get("id")
            ]
            if models:
                return models
        except Exception as exc:
            lib_logger.debug(f"Failed to fetch OpenCode Go models: {exc}")

        return [f"{PUBLIC_PROVIDER_PREFIX}/{model}" for model in FALLBACK_MODELS]

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential = kwargs.pop("credential_identifier")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_name = model.split("/", 1)[1] if "/" in model else model

        kwargs = kwargs.copy()
        kwargs["api_key"] = credential
        kwargs["api_base"] = OPENCODE_GO_API_BASE

        if model_name in ANTHROPIC_COMPATIBLE_MODELS:
            kwargs["model"] = f"anthropic/{model_name}"
            kwargs["api_base"] = OPENCODE_GO_ANTHROPIC_API_BASE
            kwargs["custom_llm_provider"] = "anthropic"
        else:
            kwargs["model"] = f"openai/{model_name}"
            kwargs["custom_llm_provider"] = "openai"

        return await litellm.acompletion(**kwargs)
