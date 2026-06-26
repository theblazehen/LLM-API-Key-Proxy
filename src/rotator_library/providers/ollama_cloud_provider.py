# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Ollama Cloud provider using Ollama's native chat API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Union

import httpx
import litellm

from .provider_interface import ProviderInterface

OLLAMA_CLOUD_API_BASE = "https://ollama.com"
OLLAMA_CLOUD_MODELS = ["glm-5.2", "deepseek-v4-pro", "deepseek-v4-flash"]


class OllamaCloudProvider(ProviderInterface):
    """API-key provider for Ollama Cloud's native API."""

    provider_env_name: str = "ollama_cloud"
    skip_cost_calculation: bool = True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(
                f"{OLLAMA_CLOUD_API_BASE}/api/tags",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("models", [])
            models = [
                f"ollama_cloud/{model.get('name') or model.get('model')}"
                for model in data
                if isinstance(model, dict) and (model.get("name") or model.get("model"))
            ]
            if models:
                return models
        except Exception:
            pass

        return [f"ollama_cloud/{model}" for model in OLLAMA_CLOUD_MODELS]

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs: Any
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential = kwargs.pop("credential_identifier")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_name = model.split("/", 1)[1] if "/" in model else model
        messages = kwargs.get("messages", [])
        stream = bool(kwargs.get("stream", False))

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }
        for source_key, target_key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_tokens", "num_predict"),
        ):
            if source_key in kwargs and kwargs[source_key] is not None:
                payload[target_key] = kwargs[source_key]

        headers = {"Authorization": f"Bearer {credential}"}
        if stream:
            return self._stream_chat(client, headers, payload, model)

        response = await client.post(
            f"{OLLAMA_CLOUD_API_BASE}/api/chat",
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            raise ValueError(f"Ollama Cloud error {response.status_code}: {response.text}")
        return self._to_litellm_response(response.json(), model)

    async def _stream_chat(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        async with client.stream(
            "POST",
            f"{OLLAMA_CLOUD_API_BASE}/api/chat",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ValueError(
                    f"Ollama Cloud error {response.status_code}: {body.decode('utf-8', errors='replace')}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message") or {}
                content = message.get("content") or ""
                if content:
                    yield litellm.ModelResponse(
                        id=f"ollama-cloud-{uuid.uuid4()}",
                        created=int(time.time()),
                        model=model,
                        choices=[
                            litellm.utils.StreamingChoices(
                                index=0,
                                delta=litellm.utils.Delta(content=content),
                                finish_reason=None,
                            )
                        ],
                    )
                if data.get("done"):
                    yield litellm.ModelResponse(
                        id=f"ollama-cloud-{uuid.uuid4()}",
                        created=int(time.time()),
                        model=model,
                        choices=[
                            litellm.utils.StreamingChoices(
                                index=0,
                                delta=litellm.utils.Delta(content=""),
                                finish_reason="stop",
                            )
                        ],
                        usage=litellm.Usage(
                            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                            completion_tokens=data.get("eval_count", 0) or 0,
                            total_tokens=(data.get("prompt_eval_count", 0) or 0)
                            + (data.get("eval_count", 0) or 0),
                        ),
                    )

    def _to_litellm_response(self, data: Dict[str, Any], model: str) -> litellm.ModelResponse:
        message = data.get("message") or {}
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        return litellm.ModelResponse(
            id=f"ollama-cloud-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop" if data.get("done", True) else None,
                )
            ],
            usage=litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
