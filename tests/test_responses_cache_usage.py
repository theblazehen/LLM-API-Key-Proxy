import asyncio
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy_app.responses_adapter import (
    chat_response_to_responses,
    chat_stream_to_responses_events,
)


async def _collect(stream):
    return [chunk async for chunk in stream]


def _completed_response(events):
    for event in events:
        text = event.decode("utf-8")
        if not text.startswith("event: response.completed\n"):
            continue
        return json.loads(text.split("data: ", 1)[1])
    raise AssertionError("response.completed event was not emitted")


def test_chat_response_preserves_cache_read_and_write_usage():
    response = chat_response_to_responses(
        {
            "id": "chatcmpl-cache",
            "model": "codex/gpt-5.6-sol",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 20,
                "total_tokens": 140,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_tokens": 16,
                },
            },
        },
        {"model": "codex/gpt-5.6-sol"},
    )

    assert response["usage"] == {
        "input_tokens": 120,
        "output_tokens": 20,
        "total_tokens": 140,
        "input_tokens_details": {
            "cached_tokens": 80,
            "cache_creation_tokens": 16,
        },
    }


def test_chat_response_omits_absent_cache_usage():
    response = chat_response_to_responses(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "total_tokens": 14,
            },
        },
        {},
    )

    assert response["usage"] == {
        "input_tokens": 12,
        "output_tokens": 2,
        "total_tokens": 14,
    }


def test_chat_stream_preserves_cache_read_and_write_usage():
    async def chat_stream():
        yield {
            "id": "chatcmpl-cache-stream",
            "model": "codex/gpt-5.6-sol",
            "created": 1,
            "choices": [{"delta": {"content": "ok"}, "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl-cache-stream",
            "model": "codex/gpt-5.6-sol",
            "created": 1,
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 20,
                "total_tokens": 140,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_tokens": 16,
                },
            },
        }

    payload = _completed_response(
        asyncio.run(_collect(chat_stream_to_responses_events(chat_stream(), {})))
    )

    assert payload["response"]["usage"] == {
        "input_tokens": 120,
        "output_tokens": 20,
        "total_tokens": 140,
        "input_tokens_details": {
            "cached_tokens": 80,
            "cache_creation_tokens": 16,
        },
    }
