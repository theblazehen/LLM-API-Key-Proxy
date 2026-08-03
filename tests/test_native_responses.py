import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class _Trace:
    def __init__(self):
        self.responses = []
        self.errors = []
        self.completed_calls = 0
        self._finished = False

    def response(self, response):
        self.responses.append(response)

    def error(self, error):
        self.errors.append(error)

    def completed(self):
        self.completed_calls += 1
        self._finished = True


class _Request:
    def __init__(self, trace):
        self.state = SimpleNamespace(llm_trace=trace)


async def _collect(stream):
    return [chunk async for chunk in stream]


def _native_stream(chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return stream()


def test_native_response_stream_preserves_fragmented_byte_sse_and_traces_terminal_response():
    from proxy_app.main import traced_native_response_stream

    terminal = {
        "id": "resp_terminal",
        "object": "response",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 8, "cache_creation_tokens": 4},
            "output_tokens": 2,
        },
    }
    event = json.dumps({"type": "response.completed", "response": terminal})
    chunks = [
        b"event: response.completed\n",
        b"data: " + event[:31].encode(),
        event[31:].encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    trace = _Trace()

    output = asyncio.run(_collect(traced_native_response_stream(_Request(trace), _native_stream(chunks))))

    assert output == chunks
    assert trace.responses == [terminal]
    assert trace.errors == []
    assert trace.completed_calls == 1


def test_native_response_stream_parses_string_sse_records_without_rewriting_them():
    from proxy_app.main import traced_native_response_stream

    terminal = {"id": "resp_string", "status": "completed", "output": [], "usage": {"input_tokens": 3}}
    event = json.dumps({"type": "response.completed", "response": terminal})
    chunks = [f"event: response.completed\ndata: {event}\n\n", "data: [DONE]\n\n"]
    trace = _Trace()

    output = asyncio.run(_collect(traced_native_response_stream(_Request(trace), _native_stream(chunks))))

    assert output == chunks
    assert trace.responses == [terminal]
    assert trace.completed_calls == 1


def test_native_response_stream_reports_upstream_exception_and_finishes_trace():
    from proxy_app.main import traced_native_response_stream

    async def broken_stream():
        yield b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\"}\n\n"
        raise RuntimeError("upstream stream disconnected")

    trace = _Trace()

    async def consume():
        return [chunk async for chunk in traced_native_response_stream(_Request(trace), broken_stream())]

    try:
        asyncio.run(consume())
    except RuntimeError as error:
        assert str(error) == "upstream stream disconnected"
    else:
        raise AssertionError("native stream error must propagate")

    assert len(trace.errors) == 1
    assert str(trace.errors[0]) == "upstream stream disconnected"
    assert trace.responses == []
    assert trace.completed_calls == 1
