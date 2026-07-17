import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from proxy_app.llm_trace import LLMTraceRecorder
from llm_tail import iter_rows, render_human


def test_records_routing_user_messages_and_response(tmp_path):
    db = tmp_path / "llm.sqlite3"
    recorder = LLMTraceRecorder(db, enabled=True)
    trace = recorder.begin_request(
        request_id="req-1",
        session_id="session-1",
        proxy_user="alice",
        requested_model="alias/gpt",
    )
    trace.request(
        {
            "model": "alias/gpt",
            "messages": [
                {"role": "user", "content": "hello\nworld"},
                {"role": "tool", "content": "tool output"},
            ],
        }
    )
    trace.credential_selected(
        resolved_model="codex/gpt-5.6-sol",
        provider="codex",
        credential_id="codex_oauth_1.json",
    )
    trace.response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    trace.completed()
    assert recorder.flush()
    recorder.close()

    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    rows = list(iter_rows(connection))
    connection.close()

    assert [row["role"] for row in rows] == [
        "request",
        "user",
        "tool",
        "proxy",
        "assistant",
        "assistant",
        "proxy",
    ]
    assistant = next(row for row in rows if row["content_text"] == "done")
    assert assistant["proxy_user"] == "alice"
    assert assistant["credential_id"] == "codex_oauth_1.json"
    assert assistant["resolved_model"] == "codex/gpt-5.6-sol"
    rendered = render_human(dict(assistant))
    assert rendered == (
        "ses [session-1] [codex/gpt-5.6-sol] "
        "[codex_oauth_1.json] [alice] assistant: done"
    )
    assert all("\n" not in render_human(dict(row)) for row in rows)


def test_recorder_is_enabled_by_default_under_usage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_TRACE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_TRACE_DB", raising=False)
    recorder = LLMTraceRecorder()
    trace = recorder.begin_request(proxy_user="default", requested_model="m")
    trace.request({"messages": [{"role": "user", "content": "hi"}]})
    assert recorder.flush()
    recorder.close()

    assert (tmp_path / "usage" / "llm-requests.sqlite3").is_file()
