# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Ollama Cloud provider using Ollama's native chat API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm

from .provider_interface import ProviderInterface

OLLAMA_CLOUD_API_BASE = "https://ollama.com"
OLLAMA_CLOUD_MODELS = ["glm-5.2", "deepseek-v4-pro", "deepseek-v4-flash"]
OLLAMA_CLOUD_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)


class OllamaCloudProvider(ProviderInterface):
    """API-key provider for Ollama Cloud's native API."""

    provider_env_name: str = "ollama_cloud"
    skip_cost_calculation: bool = True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(
                f"{OLLAMA_CLOUD_API_BASE}/api/tags",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=OLLAMA_CLOUD_TIMEOUT,
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
            "messages": self._to_ollama_messages(messages),
            "stream": stream,
        }
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
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
            timeout=OLLAMA_CLOUD_TIMEOUT,
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
            timeout=OLLAMA_CLOUD_TIMEOUT,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ValueError(
                    f"Ollama Cloud error {response.status_code}: {body.decode('utf-8', errors='replace')}"
                )

            saw_content = False
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message") or {}
                content = message.get("content") or ""
                tool_calls = self._to_openai_tool_calls(
                    message.get("tool_calls"), include_index=True
                )
                if content:
                    saw_content = True
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
                if tool_calls:
                    saw_content = True
                    yield litellm.ModelResponse(
                        id=f"ollama-cloud-{uuid.uuid4()}",
                        created=int(time.time()),
                        model=model,
                        choices=[
                            litellm.utils.StreamingChoices(
                                index=0,
                                delta=litellm.utils.Delta(tool_calls=tool_calls),
                                finish_reason=None,
                            )
                        ],
                    )
                if data.get("done"):
                    if not saw_content:
                        raise ValueError(f"Ollama Cloud returned empty stream for {model}")
                    yield litellm.ModelResponse(
                        id=f"ollama-cloud-{uuid.uuid4()}",
                        created=int(time.time()),
                        model=model,
                        choices=[
                            litellm.utils.StreamingChoices(
                                index=0,
                                delta=litellm.utils.Delta(content=""),
                                finish_reason="tool_calls" if tool_calls else "stop",
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
        content = message.get("content", "")
        tool_calls = self._to_openai_tool_calls(message.get("tool_calls"))
        if not content.strip() and not tool_calls:
            raise ValueError(f"Ollama Cloud returned empty content for {model}")

        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        finish_reason = (
            "tool_calls" if tool_calls else "stop" if data.get("done", True) else None
        )
        choice = litellm.Choices(
            index=0,
            message=litellm.Message(
                role=message.get("role", "assistant"),
                content=content,
            ),
            finish_reason=finish_reason,
        )
        if tool_calls:
            choice.message.tool_calls = tool_calls

        return litellm.ModelResponse(
            id=f"ollama-cloud-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _to_ollama_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ollama_messages = []
        for message in messages:
            converted = dict(message)
            tool_calls = converted.get("tool_calls")
            if tool_calls:
                converted["tool_calls"] = self._to_ollama_tool_calls(tool_calls)
            ollama_messages.append(converted)
        return ollama_messages

    def _to_ollama_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        converted = []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    arguments = {}
            converted.append(
                {
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    }
                }
            )
        return converted

    def _to_openai_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]], include_index: bool = False
    ) -> List[Dict[str, Any]]:
        if not tool_calls:
            return []

        converted = []
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            converted_call = {
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            }
            if include_index:
                converted_call["index"] = function.get(
                    "index", tool_call.get("index", index)
                )
            converted.append(converted_call)
        return converted
