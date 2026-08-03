import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from proxy_app.llm_trace import LLMTraceRecorder, normalize_request
from llm_tail import iter_rows, render_human


_DIAGNOSTIC_COLUMNS = {
    "request_id",
    "requested_model",
    "resolved_model",
    "provider",
    "credential_id",
    "prompt_cache_key_hash",
    "system_instructions_hash",
    "tools_schema_hash",
    "reasoning_effort",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "cache_breakpoint_mode",
    "cache_breakpoint_location",
    "previous_response_id_present",
    "response_id_hash",
    "transport_mode",
}


def _diagnostic_row(connection, request_id):
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM llm_request_diagnostics WHERE request_id = ?", (request_id,)
    ).fetchone()
    assert row is not None
    return row


def _assert_diagnostics_do_not_expose(connection, request_id, *raw_values):
    """Legacy event content may be retained; diagnostic fields and metadata may not."""
    row = _diagnostic_row(connection, request_id)
    diagnostic_text = "\n".join("" if value is None else str(value) for value in row)
    metadata_text = "\n".join(
        row[0] or ""
        for row in connection.execute(
            "SELECT metadata_json FROM llm_events WHERE request_id = ?", (request_id,)
        )
    )
    for raw_value in raw_values:
        assert raw_value not in diagnostic_text
        assert raw_value not in metadata_text


def _create_legacy_trace_database(path):
    """Create the schema used before request diagnostics were added."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE llm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            session_id TEXT,
            timestamp TEXT NOT NULL,
            proxy_user TEXT,
            requested_model TEXT,
            resolved_model TEXT,
            provider TEXT,
            credential_id TEXT,
            role TEXT NOT NULL,
            event_type TEXT NOT NULL,
            content_text TEXT,
            content_json TEXT,
            status TEXT,
            metadata_json TEXT
        );
        CREATE INDEX llm_events_request_id ON llm_events(request_id, id);
        CREATE INDEX llm_events_session_id ON llm_events(session_id, id);
        CREATE INDEX llm_events_user_id ON llm_events(proxy_user, id);
        CREATE INDEX llm_events_model_id ON llm_events(resolved_model, requested_model, id);
        CREATE TABLE llm_session_prefixes (
            proxy_user TEXT NOT NULL,
            requested_model TEXT NOT NULL,
            prefix_hash TEXT NOT NULL,
            prefix_length INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (proxy_user, requested_model, prefix_hash)
        );
        CREATE INDEX llm_session_prefixes_updated
            ON llm_session_prefixes(updated_at DESC);
        """
    )
    connection.execute(
        "INSERT INTO llm_events (request_id, timestamp, role, event_type) VALUES (?, ?, ?, ?)",
        ("legacy-request", "2026-08-03T00:00:00.000+00:00", "request", "begin"),
    )
    connection.commit()
    connection.close()


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


def test_emits_only_latest_turn():
    first = {
        "model": "alias/gpt",
        "messages": [{"role": "user", "content": "first question"}],
    }
    next_turn = {
        "model": "alias/gpt",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "tool", "content": "fresh tool result"},
            {"role": "user", "content": "follow up"},
        ],
    }

    assert [event["content_text"] for event in normalize_request(next_turn)] == [
        "fresh tool result",
        "follow up",
    ]


def test_anthropic_system_prompt_is_not_replayed_after_first_turn():
    first_turn = {
        "system": "large private instructions",
        "messages": [{"role": "user", "content": "first question"}],
    }
    next_turn = {
        "system": "large private instructions",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ],
    }

    assert [event["content_text"] for event in normalize_request(first_turn)] == [
        "large private instructions",
        "first question",
    ]
    assert [event["content_text"] for event in normalize_request(next_turn)] == [
        "follow up"
    ]


def test_longest_prefix_links_forks_and_persists_across_restart(tmp_path):
    model = "prefix-test-model"
    db = tmp_path / "prefixes.sqlite3"
    recorder = LLMTraceRecorder(db, enabled=True)
    first = {
        "model": model,
        "messages": [
            {"role": "system", "content": "shared coding-agent prompt"},
            {"role": "user", "content": "original anchor"},
        ],
    }
    second = {
        "model": model,
        "messages": [
            {"role": "user", "content": "original anchor"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second user turn"},
        ],
    }
    fork = {
        "model": model,
        "messages": [
            {"role": "system", "content": "shared coding-agent prompt"},
            {"role": "user", "content": "original anchor"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "forked user turn"},
        ],
    }

    session_id = recorder.infer_session_id(first, "prefix-user")
    assert recorder.infer_session_id(second, "prefix-user") == session_id
    assert recorder.infer_session_id(fork, "prefix-user") == session_id
    assert recorder.flush()
    recorder.close()

    restarted = LLMTraceRecorder(db, enabled=True)
    deadline = __import__("time").monotonic() + 2
    while not restarted._prefix_sessions and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert restarted.infer_session_id(second, "prefix-user") == session_id

    unrelated = {
        "model": model,
        "messages": [
            {"role": "system", "content": "shared coding-agent prompt"},
            {"role": "user", "content": "different chat"},
        ],
    }
    assert restarted.infer_session_id(unrelated, "prefix-user") != session_id
    restarted.close()


def test_chat_completions_diagnostics_capture_usage_without_sensitive_values(tmp_path):
    db = tmp_path / "chat-diagnostics.sqlite3"
    raw_cache_key = "chat-cache-key-private-value"
    raw_instructions = "chat-system-instructions-private-value"
    raw_tool_schema = "chat-tool-schema-private-value"
    raw_response_id = "chatcmpl-private-response-id"
    bearer_value = "Bearer chat-private-bearer-value"
    recorder = LLMTraceRecorder(db, enabled=True)
    trace = recorder.begin_request(
        request_id="chat-diagnostics",
        proxy_user="alice",
        requested_model="alias/gpt",
    )
    trace.request(
        {
            "model": "alias/gpt",
            "messages": [
                {"role": "system", "content": raw_instructions},
                {"role": "user", "content": "diagnose the cache"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cache_probe",
                        "description": raw_tool_schema,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "reasoning_effort": "medium",
            "prompt_cache_key": raw_cache_key,
            "prompt_cache_options": {"mode": "implicit"},
        },
        metadata={"authorization": bearer_value, "safe_label": "chat"},
    )
    trace.credential_selected(
        resolved_model="codex/gpt-5.6-sol",
        provider="codex",
        credential_id="codex_oauth_1.json",
        metadata={"nested": {"access_token": bearer_value}},
    )
    trace.response(
        {
            "id": raw_response_id,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 144,
                "completion_tokens": 32,
                "total_tokens": 176,
                "prompt_tokens_details": {
                    "cached_tokens": 112,
                    "cache_creation_tokens": 8,
                },
            },
        },
        metadata={"api_key": bearer_value},
    )
    trace.completed(metadata={"token": bearer_value})
    assert recorder.flush()
    recorder.close()

    connection = sqlite3.connect(db)
    diagnostic = _diagnostic_row(connection, "chat-diagnostics")
    assert diagnostic["requested_model"] == "alias/gpt"
    assert diagnostic["resolved_model"] == "codex/gpt-5.6-sol"
    assert diagnostic["provider"] == "codex"
    assert diagnostic["credential_id"] == "codex_oauth_1.json"
    assert diagnostic["transport_mode"] == "chat_completions"
    assert diagnostic["reasoning_effort"] == "medium"
    assert diagnostic["prompt_cache_key_hash"] == hashlib.sha256(
        raw_cache_key.encode()
    ).hexdigest()
    assert diagnostic["system_instructions_hash"]
    assert diagnostic["tools_schema_hash"]
    assert diagnostic["cache_breakpoint_mode"] == "implicit"
    assert diagnostic["cache_breakpoint_location"] is None
    assert diagnostic["previous_response_id_present"] == 0
    assert diagnostic["response_id_hash"] == hashlib.sha256(
        raw_response_id.encode()
    ).hexdigest()
    assert diagnostic["input_tokens"] == 144
    assert diagnostic["cache_read_tokens"] == 112
    assert diagnostic["cache_write_tokens"] == 8
    assert diagnostic["output_tokens"] == 32
    assert connection.execute(
        "SELECT COUNT(*) FROM llm_request_diagnostics WHERE request_id = ?",
        ("chat-diagnostics",),
    ).fetchone()[0] == 1
    _assert_diagnostics_do_not_expose(
        connection,
        "chat-diagnostics",
        raw_cache_key,
        raw_instructions,
        raw_tool_schema,
        raw_response_id,
        bearer_value,
    )
    connection.close()


def test_native_responses_diagnostics_capture_request_shape_and_usage(tmp_path):
    db = tmp_path / "responses-diagnostics.sqlite3"
    raw_cache_key = "responses-cache-key-private-value"
    raw_instructions = "responses-instructions-private-value"
    raw_tool_schema = "responses-tool-schema-private-value"
    raw_previous_response_id = "resp-private-previous-response-id"
    raw_response_id = "resp-private-response-id"
    bearer_value = "Bearer responses-private-bearer-value"
    recorder = LLMTraceRecorder(db, enabled=True)
    trace = recorder.begin_request(
        request_id="responses-diagnostics",
        proxy_user="bob",
        requested_model="codex/gpt-5.6-terra",
    )
    trace.request(
        {
            "model": "codex/gpt-5.6-terra",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "continue the implementation"}
                    ],
                }
            ],
            "instructions": raw_instructions,
            "tools": [
                {
                    "type": "function",
                    "name": "cache_probe",
                    "description": raw_tool_schema,
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "reasoning": {"effort": "high"},
            "prompt_cache_key": raw_cache_key,
            "prompt_cache_breakpoint": {"mode": "explicit", "index": 2},
            "previous_response_id": raw_previous_response_id,
        },
        metadata={"proxy-authorization": bearer_value},
    )
    trace.credential_selected(
        resolved_model="gpt-5.6-terra",
        provider="codex",
        credential_id="codex_oauth_2.json",
    )
    trace.response(
        {
            "id": raw_response_id,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "native done"}],
                }
            ],
            "usage": {
                "input_tokens": 128,
                "output_tokens": 24,
                "input_tokens_details": {
                    "cached_tokens": 96,
                    "cache_creation_tokens": 16,
                },
            },
        },
        metadata={"refresh_token": bearer_value},
    )
    trace.completed()
    assert recorder.flush()
    recorder.close()

    connection = sqlite3.connect(db)
    diagnostic = _diagnostic_row(connection, "responses-diagnostics")
    assert diagnostic["requested_model"] == "codex/gpt-5.6-terra"
    assert diagnostic["resolved_model"] == "gpt-5.6-terra"
    assert diagnostic["provider"] == "codex"
    assert diagnostic["credential_id"] == "codex_oauth_2.json"
    assert diagnostic["transport_mode"] == "responses"
    assert diagnostic["reasoning_effort"] == "high"
    assert diagnostic["prompt_cache_key_hash"] == hashlib.sha256(
        raw_cache_key.encode()
    ).hexdigest()
    assert diagnostic["system_instructions_hash"]
    assert diagnostic["tools_schema_hash"]
    assert diagnostic["cache_breakpoint_mode"] == "explicit"
    assert diagnostic["cache_breakpoint_location"] == "2"
    assert diagnostic["previous_response_id_present"] == 1
    assert diagnostic["response_id_hash"] == hashlib.sha256(
        raw_response_id.encode()
    ).hexdigest()
    assert diagnostic["input_tokens"] == 128
    assert diagnostic["cache_read_tokens"] == 96
    assert diagnostic["cache_write_tokens"] == 16
    assert diagnostic["output_tokens"] == 24
    _assert_diagnostics_do_not_expose(
        connection,
        "responses-diagnostics",
        raw_cache_key,
        raw_instructions,
        raw_tool_schema,
        raw_previous_response_id,
        raw_response_id,
        bearer_value,
    )
    connection.close()


def test_migrates_legacy_trace_database_and_captures_diagnostics(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    _create_legacy_trace_database(db)
    recorder = LLMTraceRecorder(db, enabled=True)
    trace = recorder.begin_request(request_id="migrated-request", requested_model="m")
    trace.request({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    trace.response(
        {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
    )
    trace.completed()
    assert recorder.flush()
    recorder.close()

    connection = sqlite3.connect(db)
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(llm_request_diagnostics)")
    }
    assert _DIAGNOSTIC_COLUMNS <= columns
    assert connection.execute(
        "SELECT COUNT(*) FROM llm_events WHERE request_id = ?", ("legacy-request",)
    ).fetchone()[0] == 1
    diagnostic = _diagnostic_row(connection, "migrated-request")
    assert diagnostic["transport_mode"] == "chat_completions"
    assert diagnostic["input_tokens"] == 2
    assert diagnostic["cache_read_tokens"] in (None, 0)
    assert diagnostic["cache_write_tokens"] in (None, 0)
    assert diagnostic["output_tokens"] == 1
    connection.close()
