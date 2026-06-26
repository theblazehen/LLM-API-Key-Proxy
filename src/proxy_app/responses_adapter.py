# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""OpenAI Responses API compatibility helpers.

This module implements a conservative, stateless Responses API adapter backed by
Chat Completions. It is intentionally protocol-focused: routing, credentials,
fallbacks, and provider-specific behavior stay in ``RotatingClient``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional


def responses_to_chat_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a /v1/responses request into /v1/chat/completions shape."""
    messages = _responses_input_to_messages(body.get("input", ""), body.get("instructions"))

    chat: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }

    _copy_if_present(body, chat, "temperature")
    _copy_if_present(body, chat, "top_p")
    _copy_if_present(body, chat, "presence_penalty")
    _copy_if_present(body, chat, "frequency_penalty")
    _copy_if_present(body, chat, "stop")
    _copy_if_present(body, chat, "user")
    _copy_if_present(body, chat, "tool_choice")
    _copy_if_present(body, chat, "parallel_tool_calls")

    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        chat["max_tokens"] = body["max_tokens"]

    response_format = _response_format_from_responses(body)
    if response_format:
        chat["response_format"] = response_format

    tools = _responses_tools_to_chat(body.get("tools"))
    if tools is not None:
        chat["tools"] = tools

    thinking = _reasoning_to_thinking(body.get("reasoning"), body.get("thinking"))
    if thinking is not None:
        chat["thinking"] = thinking

    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}

    return chat


def chat_response_to_responses(body: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a non-streaming Chat Completions response to Responses shape."""
    response_id = body.get("id") or f"resp_{uuid.uuid4().hex}"
    created_at = body.get("created") or int(time.time())
    model = body.get("model") or original_request.get("model")
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    finish_reason = choices[0].get("finish_reason") if choices else None

    output = _message_to_output_items(message)
    output_text = _extract_text(message.get("content"))
    reasoning_text = _extract_reasoning(message)
    usage = _responses_usage(body.get("usage"))

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": _status_from_finish_reason(finish_reason),
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": original_request.get("instructions"),
        "max_output_tokens": original_request.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": bool(message.get("tool_calls")),
        "previous_response_id": None,
        "reasoning": _reasoning_summary(reasoning_text),
        "service_tier": "default",
        "store": bool(original_request.get("store", False)),
        "temperature": original_request.get("temperature"),
        "text": original_request.get("text") or {"format": {"type": "text"}},
        "tool_choice": original_request.get("tool_choice", "auto" if original_request.get("tools") else "none"),
        "tools": original_request.get("tools") or [],
        "top_p": original_request.get("top_p"),
        "truncation": original_request.get("truncation", "disabled"),
        "usage": usage,
        "user": original_request.get("user"),
        "metadata": original_request.get("metadata") or {},
        "output_text": output_text,
    }


async def chat_stream_to_responses_events(
    chat_stream: AsyncGenerator[Any, None],
    original_request: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Convert Chat Completions streaming chunks into Responses SSE events."""
    state = _ResponsesStreamState(original_request)
    yield state.sse("response.created", {"response": state.response("in_progress")})
    yield state.sse("response.in_progress", {"response": state.response("in_progress")})

    try:
        async for raw_chunk in chat_stream:
            chunk = _chunk_to_dict(raw_chunk)
            if not chunk:
                continue
            state.apply_metadata(chunk)
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                state.usage = _responses_usage(usage)

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning_delta:
                for event in state.ensure_reasoning_started():
                    yield event
                state.reasoning_text += reasoning_delta
                yield state.sse(
                    "response.reasoning_text.delta",
                    {
                        "response_id": state.response_id,
                        "item_id": state.reasoning_item_id,
                        "output_index": state.reasoning_output_index,
                        "content_index": 0,
                        "delta": reasoning_delta,
                    },
                )

            text_delta = _extract_text(delta.get("content"))
            if text_delta:
                for event in state.ensure_message_started():
                    yield event
                state.output_text += text_delta
                yield state.sse(
                    "response.output_text.delta",
                    {
                        "response_id": state.response_id,
                        "item_id": state.message_item_id,
                        "output_index": state.message_output_index,
                        "content_index": 0,
                        "delta": text_delta,
                    },
                )

            for tool_delta in delta.get("tool_calls") or []:
                index = state.merge_tool_delta(tool_delta)
                for event in state.ensure_tool_started(index):
                    yield event
                function_delta = (tool_delta.get("function") or {}).get("arguments")
                if function_delta is not None:
                    yield state.sse(
                        "response.function_call_arguments.delta",
                        {
                            "response_id": state.response_id,
                            "item_id": state.tool_item_ids[index],
                            "output_index": state.tool_output_indexes[index],
                            "call_id": state.tool_calls[index].get("id") or f"call_{index}",
                            "delta": function_delta,
                        },
                    )

            if choice.get("finish_reason"):
                state.finish_reason = choice.get("finish_reason")

        for event in state.done_events():
            yield event
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        yield state.sse("error", {"message": str(exc), "code": "proxy_error", "param": None})
        yield state.sse(
            "response.completed",
            {
                "response": state.response(
                    "completed",
                    output=state.completed_output(),
                    error={"message": str(exc), "type": "proxy_error", "code": "proxy_error"},
                )
            },
        )
        yield b"data: [DONE]\n\n"


class _ResponsesStreamState:
    def __init__(self, original_request: Dict[str, Any]) -> None:
        self.original_request = original_request
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.created_at = int(time.time())
        self.model = original_request.get("model")
        self.sequence_number = 0
        self.next_output_index = 0
        self.output_text = ""
        self.reasoning_text = ""
        self.reasoning_item_id = f"rs_{uuid.uuid4().hex}"
        self.reasoning_output_index: Optional[int] = None
        self.reasoning_started = False
        self.message_item_id = f"msg_{uuid.uuid4().hex}"
        self.message_output_index: Optional[int] = None
        self.message_started = False
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.tool_order: List[int] = []
        self.tool_item_ids: Dict[int, str] = {}
        self.tool_output_indexes: Dict[int, int] = {}
        self.finish_reason: Optional[str] = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def apply_metadata(self, chunk: Dict[str, Any]) -> None:
        self.model = chunk.get("model") or self.model
        self.created_at = chunk.get("created") or self.created_at

    def sse(self, event_type: str, payload: Dict[str, Any]) -> bytes:
        payload = dict(payload)
        payload["type"] = event_type
        payload["sequence_number"] = self.sequence_number
        payload["event_id"] = f"event_{uuid.uuid4().hex}"
        self.sequence_number += 1
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")

    def response(
        self,
        status: str,
        output: Optional[List[Dict[str, Any]]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "background": False,
            "error": error,
            "incomplete_details": None,
            "instructions": self.original_request.get("instructions"),
            "max_output_tokens": self.original_request.get("max_output_tokens"),
            "model": self.model,
            "output": output or [],
            "parallel_tool_calls": bool(self.tool_calls),
            "previous_response_id": None,
            "reasoning": _reasoning_summary(self.reasoning_text),
            "service_tier": "default",
            "store": bool(self.original_request.get("store", False)),
            "temperature": self.original_request.get("temperature"),
            "text": self.original_request.get("text") or {"format": {"type": "text"}},
            "tool_choice": self.original_request.get("tool_choice", "auto" if self.original_request.get("tools") else "none"),
            "tools": self.original_request.get("tools") or [],
            "top_p": self.original_request.get("top_p"),
            "truncation": self.original_request.get("truncation", "disabled"),
            "usage": self.usage,
            "user": self.original_request.get("user"),
            "metadata": self.original_request.get("metadata") or {},
            "output_text": self.output_text,
        }

    def ensure_reasoning_started(self) -> List[bytes]:
        if self.reasoning_started:
            return []
        self.reasoning_started = True
        self.reasoning_output_index = self.next_output_index
        self.next_output_index += 1
        return [
            self.sse(
                "response.output_item.added",
                {
                    "response_id": self.response_id,
                    "output_index": self.reasoning_output_index,
                    "item": {"id": self.reasoning_item_id, "type": "reasoning", "summary": [], "content": []},
                },
            ),
            self.sse(
                "response.content_part.added",
                {
                    "response_id": self.response_id,
                    "item_id": self.reasoning_item_id,
                    "output_index": self.reasoning_output_index,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": ""},
                },
            ),
        ]

    def ensure_message_started(self) -> List[bytes]:
        if self.message_started:
            return []
        self.message_started = True
        self.message_output_index = self.next_output_index
        self.next_output_index += 1
        return [
            self.sse(
                "response.output_item.added",
                {
                    "response_id": self.response_id,
                    "output_index": self.message_output_index,
                    "item": _message_item("", self.message_item_id, "in_progress"),
                },
            ),
            self.sse(
                "response.content_part.added",
                {
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": self.message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                },
            ),
        ]

    def merge_tool_delta(self, tool_delta: Dict[str, Any]) -> int:
        index = int(tool_delta.get("index", 0) or 0)
        current = self.tool_calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tool_delta.get("id"):
            current["id"] = tool_delta["id"]
        if tool_delta.get("type"):
            current["type"] = tool_delta["type"]
        function_delta = tool_delta.get("function") or {}
        if function_delta.get("name"):
            current["function"]["name"] = function_delta["name"]
        if "arguments" in function_delta:
            current["function"]["arguments"] += function_delta.get("arguments") or ""
        return index

    def ensure_tool_started(self, index: int) -> List[bytes]:
        if index in self.tool_output_indexes:
            return []
        self.tool_order.append(index)
        self.tool_output_indexes[index] = self.next_output_index
        self.next_output_index += 1
        self.tool_item_ids[index] = f"fc_{uuid.uuid4().hex}"
        tool_call = self.tool_calls[index]
        return [
            self.sse(
                "response.output_item.added",
                {
                    "response_id": self.response_id,
                    "output_index": self.tool_output_indexes[index],
                    "item": {
                        "id": self.tool_item_ids[index],
                        "type": "function_call",
                        "call_id": tool_call.get("id") or f"call_{index}",
                        "name": (tool_call.get("function") or {}).get("name") or "",
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        ]

    def done_events(self) -> List[bytes]:
        events: List[bytes] = []
        output_by_index: Dict[int, Dict[str, Any]] = {}

        if self.reasoning_started and self.reasoning_output_index is not None:
            item = _reasoning_item(self.reasoning_text, self.reasoning_item_id)
            output_by_index[self.reasoning_output_index] = item
            events.extend([
                self.sse(
                    "response.reasoning_text.done",
                    {
                        "response_id": self.response_id,
                        "item_id": self.reasoning_item_id,
                        "output_index": self.reasoning_output_index,
                        "content_index": 0,
                        "text": self.reasoning_text,
                    },
                ),
                self.sse(
                    "response.output_item.done",
                    {"response_id": self.response_id, "output_index": self.reasoning_output_index, "item": item},
                ),
            ])

        if self.message_started and self.message_output_index is not None:
            item = _message_item(self.output_text, self.message_item_id, "completed")
            output_by_index[self.message_output_index] = item
            events.extend([
                self.sse(
                    "response.output_text.done",
                    {
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": 0,
                        "text": self.output_text,
                        "logprobs": [],
                    },
                ),
                self.sse(
                    "response.content_part.done",
                    {
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": self.output_text, "annotations": [], "logprobs": []},
                    },
                ),
                self.sse(
                    "response.output_item.done",
                    {"response_id": self.response_id, "output_index": self.message_output_index, "item": item},
                ),
            ])

        for index in self.tool_order:
            item = _function_call_item(self.tool_calls[index], self.tool_item_ids[index])
            output_by_index[self.tool_output_indexes[index]] = item
            events.extend([
                self.sse(
                    "response.function_call_arguments.done",
                    {
                        "response_id": self.response_id,
                        "item_id": item["id"],
                        "output_index": self.tool_output_indexes[index],
                        "call_id": item["call_id"],
                        "arguments": item["arguments"],
                        "name": item["name"],
                    },
                ),
                self.sse(
                    "response.output_item.done",
                    {"response_id": self.response_id, "output_index": self.tool_output_indexes[index], "item": item},
                ),
            ])

        output = [output_by_index[index] for index in sorted(output_by_index)]
        events.append(self.sse("response.completed", {"response": self.response("completed", output=output)}))
        return events

    def completed_output(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        if self.reasoning_text:
            output.append(_reasoning_item(self.reasoning_text, self.reasoning_item_id))
        if self.output_text:
            output.append(_message_item(self.output_text, self.message_item_id, "completed"))
        for index in self.tool_order:
            output.append(_function_call_item(self.tool_calls[index], self.tool_item_ids[index]))
        return output


def _responses_input_to_messages(input_data: Any, instructions: Optional[Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": _extract_text(instructions)})

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    if not isinstance(input_data, list):
        messages.append({"role": "user", "content": _extract_text(input_data)})
        return messages

    for item in input_data:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in (None, "message"):
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            messages.append({"role": role, "content": _responses_content_to_chat(item.get("content", ""))})
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": _extract_text(item.get("output", "")),
                }
            )
    return messages


def _responses_content_to_chat(content: Any) -> Any:
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        return _extract_text(content)

    chat_parts: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"} or "text" in part:
            text_parts.append(part.get("text", ""))
        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url") or part.get("url")
            if image_url:
                if text_parts:
                    chat_parts.append({"type": "text", "text": "\n".join(p for p in text_parts if p)})
                    text_parts = []
                chat_parts.append({"type": "image_url", "image_url": image_url if isinstance(image_url, dict) else {"url": image_url}})

    if chat_parts:
        if text_parts:
            chat_parts.append({"type": "text", "text": "\n".join(p for p in text_parts if p)})
        return chat_parts
    return "\n".join(p for p in text_parts if p)


def _responses_tools_to_chat(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(tools, list):
        return None
    mapped: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            mapped.append(
                {
                    "type": "function",
                    "function": {
                        "name": function.get("name") or tool.get("name") or "",
                        "description": function.get("description") or tool.get("description") or "",
                        "parameters": function.get("parameters") or tool.get("parameters") or {"type": "object", "properties": {}},
                        **({"strict": function.get("strict")} if "strict" in function else {}),
                    },
                }
            )
        elif tool_type == "custom":
            mapped.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name") or "custom_tool",
                        "description": tool.get("description") or "Custom tool. Pass raw input in the input field.",
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string", "description": "Raw tool input."}},
                            "required": ["input"],
                        },
                        "strict": False,
                    },
                }
            )
    return mapped


def _message_to_output_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    reasoning = _extract_reasoning(message)
    if reasoning:
        items.append(_reasoning_item(reasoning))
    text = _extract_text(message.get("content"))
    if text:
        items.append(_message_item(text))
    for tool_call in message.get("tool_calls") or []:
        items.append(_function_call_item(tool_call))
    return items


def _function_call_item(tool_call: Dict[str, Any], item_id: Optional[str] = None) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
    return {
        "id": item_id or f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "call_id": call_id,
        "name": function.get("name") or "",
        "arguments": function.get("arguments") or "{}",
        "status": "completed",
    }


def _message_item(text: str, item_id: Optional[str] = None, status: str = "completed") -> Dict[str, Any]:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        ] if text or status == "completed" else [],
    }


def _reasoning_item(text: str, item_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": item_id or f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _reasoning_summary(text: str) -> Dict[str, Any]:
    return {"effort": None, "summary": [{"type": "summary_text", "text": text}] if text else []}


def _extract_reasoning(message: Dict[str, Any]) -> str:
    return _extract_text(message.get("reasoning_content") or message.get("reasoning") or "")


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item.get("text") or ""))
                elif item.get("content") is not None:
                    parts.append(_extract_text(item.get("content")))
        return "\n".join(part for part in parts if part)
    if isinstance(content, (dict, int, float, bool)):
        return json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)
    return str(content)


def _response_format_from_responses(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(body.get("response_format"), dict):
        return body["response_format"]
    text = body.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema") or fmt
        return {"type": "json_schema", "json_schema": schema}
    if fmt.get("type") in {"json_object", "text"}:
        return {"type": fmt["type"]}
    return None


def _reasoning_to_thinking(reasoning: Any, thinking: Any) -> Optional[Dict[str, Any]]:
    if isinstance(thinking, dict):
        return thinking
    if reasoning is None:
        return None
    if reasoning is False:
        return {"type": "disabled"}
    if isinstance(reasoning, dict) and reasoning.get("effort") == "none":
        return {"type": "disabled"}
    return {"type": "enabled"}


def _responses_usage(usage: Any) -> Dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def _status_from_finish_reason(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "incomplete"
    return "completed"


def _copy_if_present(source: Dict[str, Any], target: Dict[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        target[key] = source[key]


def _chunk_to_dict(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    return {}


def sse_bytes_to_text(chunks: Iterable[bytes]) -> str:
    """Test helper for inspecting generated SSE output."""
    return b"".join(chunks).decode("utf-8")
