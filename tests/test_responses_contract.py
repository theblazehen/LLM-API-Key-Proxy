import asyncio
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy_app.responses_adapter import (
    chat_response_to_responses,
    chat_stream_to_responses_events,
    responses_to_chat_request,
)


async def _collect(stream):
    return [chunk async for chunk in stream]


def _sse_events(chunks):
    events = []
    for chunk in chunks:
        text = chunk.decode("utf-8")
        if text == "data: [DONE]\n\n":
            continue
        event_name, data = text.split("\ndata: ", 1)
        events.append((event_name.removeprefix("event: "), json.loads(data)))
    return events


def test_non_streaming_text_response_is_a_public_responses_object():
    response = chat_response_to_responses(
        {
            "id": "chatcmpl_contract",
            "created": 1,
            "model": "codex/gpt-5.6-sol",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "harmless reply"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        {"model": "codex/gpt-5.6-sol", "store": False},
    )

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert len(response["output"]) == 1
    output = response["output"][0]
    assert output["type"] == "message"
    assert output["status"] == "completed"
    assert output["role"] == "assistant"
    assert output["content"][0]["type"] == "output_text"
    assert output["content"][0]["text"] == "harmless reply"
    assert response["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    assert response["previous_response_id"] is None


def test_adapted_streaming_text_emits_valid_terminal_responses_object():
    async def chat_stream():
        yield {
            "id": "chatcmpl_stream_contract",
            "model": "codex/gpt-5.6-sol",
            "created": 1,
            "choices": [{"delta": {"content": "harmless "}, "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl_stream_contract",
            "model": "codex/gpt-5.6-sol",
            "created": 1,
            "choices": [{"delta": {"content": "stream"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    events = _sse_events(asyncio.run(_collect(chat_stream_to_responses_events(chat_stream(), {}))))
    event_types = [event[0] for event in events]
    completed = next(payload["response"] for name, payload in events if name == "response.completed")

    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert "response.output_text.delta" in event_types
    assert completed["object"] == "response"
    assert completed["status"] == "completed"
    assert completed["output"][-1]["type"] == "message"
    assert completed["output"][-1]["content"][0]["text"] == "harmless stream"
    assert completed["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


def test_full_history_function_output_becomes_valid_chat_tool_continuation():
    chat = responses_to_chat_request(
        {
            "model": "codex/gpt-5.6-sol",
            "input": [
                {"role": "user", "content": "use harmless_tool"},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "harmless_tool",
                    "arguments": '{"value":"ok"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "harmless result"},
                {"role": "user", "content": "continue"},
            ],
        }
    )

    assert chat["messages"] == [
        {"role": "user", "content": "use harmless_tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "harmless_tool", "arguments": '{"value":"ok"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "harmless result"},
        {"role": "user", "content": "continue"},
    ]


def test_prompt_cache_key_is_preserved_by_responses_fallback_conversion():
    chat = responses_to_chat_request(
        {
            "model": "codex/gpt-5.6-sol",
            "input": "harmless input",
            "prompt_cache_key": "opaque-cache-key",
        }
    )

    assert chat["prompt_cache_key"] == "opaque-cache-key"


def test_fallback_responses_are_explicitly_stateless_for_previous_response_id():
    chat = responses_to_chat_request(
        {
            "model": "codex/gpt-5.6-sol",
            "previous_response_id": "resp_prior",
            "input": "harmless input",
        }
    )
    response = chat_response_to_responses(
        {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
        {"model": "codex/gpt-5.6-sol", "previous_response_id": "resp_prior"},
    )

    assert "previous_response_id" not in chat
    assert response["previous_response_id"] is None
