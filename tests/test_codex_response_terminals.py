"""Exercise real response conversion, isolated from legacy tests' module stubs."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_SCENARIOS = r'''
import asyncio
import json
import sys

import httpx

from rotator_library.core.errors import StreamedAPIError
from rotator_library.error_handler import EmptyResponseError
from rotator_library.providers.codex_provider import CodexProvider

streaming = sys.argv[1] == "stream"
provider = CodexProvider.__new__(CodexProvider)
reasoning = {"type": "response.reasoning_summary_text.delta", "delta": "Still thinking"}
text = {"type": "response.output_text.delta", "delta": "Final answer"}
tool = {"type": "response.output_item.done", "output_index": 0, "item": {
    "type": "function_call", "call_id": "call_real", "name": "inspect", "arguments": '{"path":"x"}'}}
added = {"type": "response.output_item.added", "output_index": 0, "item": tool["item"]}
completed = {"type": "response.completed", "response": {
    "id": "resp_real", "status": "completed", "usage": {
        "input_tokens": 3, "output_tokens": 7, "total_tokens": 10}}}

async def invoke(events, chunks):
    body = "".join("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n"
                   for event in events)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body, request=request))
    async with httpx.AsyncClient(transport=transport) as client:
        args = dict(client=client, headers={}, payload={}, model="test-model", reasoning_compat="current")
        if streaming:
            async for chunk in provider._stream_response(**args):
                wire_chunk = chunk.model_dump()
                assert wire_chunk["object"] == "chat.completion.chunk", wire_chunk
                assert "message" not in wire_chunk["choices"][0], wire_chunk
                chunks.append(wire_chunk)
            return chunks[-1]
        result = await provider._non_stream_response(**args)
        return result.model_dump()

async def main():
    # EOF, malformed/truncated JSON, and [DONE] cannot substitute for completion.
    for prefix in ([], [reasoning], [text], [added, tool]):
        for suffix in ([], ['{"type":"response.completed"'], ["[DONE]"]):
            chunks = []
            try:
                await invoke(prefix + suffix, chunks)
            except StreamedAPIError as error:
                assert "without response.completed" in str(error), str(error)
            else:
                raise AssertionError(("missing terminal accepted", prefix, suffix))
            assert all(c["choices"][0]["finish_reason"] is None for c in chunks), chunks

    for terminal, detail in [
        ({"type": "response.incomplete", "response": {"status": "incomplete",
          "incomplete_details": {"reason": "max_output_tokens"}}}, "max_output_tokens"),
        ({"type": "response.failed", "response": {"status": "failed",
          "error": {"code": "server_error", "message": "provider exploded"}}}, "provider exploded"),
        ({"type": "error", "code": "upstream_error", "message": "event failure"}, "event failure"),
        ({"type": "response.completed", "response": {"status": "incomplete",
          "incomplete_details": {"reason": "content_filter"}}}, "content_filter"),
    ]:
        chunks = []
        try:
            await invoke([reasoning, terminal, completed], chunks)
        except StreamedAPIError as error:
            assert error.data == terminal, error.data
            assert detail in str(error), str(error)
        else:
            raise AssertionError(("unsuccessful terminal accepted", terminal))
        assert all(c["choices"][0]["finish_reason"] is None for c in chunks), chunks

    for prefix, finish in [([text], "stop"), ([added, tool], "tool_calls"), ([reasoning], "stop")]:
        chunks = []
        result = await invoke(prefix + [completed, "[DONE]"], chunks)
        assert result["choices"][0]["finish_reason"] == finish, result
        assert result["usage"]["total_tokens"] == 10, result
        messages = [c["choices"][0]["delta"] for c in chunks] if streaming else [result["choices"][0]["message"]]
        if prefix == [text]:
            assert "".join(m.get("content") or "" for m in messages) == "Final answer", messages
        elif prefix == [reasoning]:
            field = "reasoning_content" if streaming else "reasoning_summary"
            assert "".join(m.get(field) or "" for m in messages) == "Still thinking", messages
            assert not any(m.get("tool_calls") for m in messages), messages
        else:
            calls = [call for m in messages for call in (m.get("tool_calls") or [])]
            assert len(calls) == 1, calls
            assert calls[0]["id"] == "call_real", calls
            assert calls[0]["function"] == {"name": "inspect", "arguments": '{"path":"x"}'}, calls

    try:
        await invoke([completed], [])
    except EmptyResponseError:
        pass
    else:
        raise AssertionError("empty completed response no longer uses existing error policy")

asyncio.run(main())
'''


@pytest.mark.parametrize("mode", ["stream", "nonstream"])
def test_codex_response_terminal_contract(mode):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", _SCENARIOS, mode],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
