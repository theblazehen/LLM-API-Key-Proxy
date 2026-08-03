# SPDX-License-Identifier: MIT
"""Low-overhead, process-wide SQLite tracing for LLM request lifecycles.

Integration is intentionally small::

    trace = get_recorder().begin_request(...)
    request.state.llm_trace = trace
    trace.request(payload)
    trace.credential_selected(...)
    trace.response(final_response)  # call once, after stream assembly
    trace.completed()

Only safe identity labels belong in ``proxy_user`` and ``credential_id``.  The
recorder never accepts headers and recursively removes secret-looking metadata
keys. Message bodies and tool arguments are retained verbatim by design.
"""

from __future__ import annotations

import atexit
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Any, Mapping
import uuid

_DEFAULT_DB = "usage/llm-requests.sqlite3"
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "token",
        "secret",
        "password",
        "cookie",
        "set-cookie",
        "x-api-key",
    }
)
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "prompt_cache_key",
        "previous_response_id",
        "response_id",
        "instructions",
        "system",
        "messages",
        "input",
        "output",
        "tools",
    }
)
_TRANSPORT_MODES = frozenset(
    {"chat_completions", "responses", "anthropic_messages"}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_events (
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
CREATE INDEX IF NOT EXISTS llm_events_request_id ON llm_events(request_id, id);
CREATE INDEX IF NOT EXISTS llm_events_session_id ON llm_events(session_id, id);
CREATE INDEX IF NOT EXISTS llm_events_user_id ON llm_events(proxy_user, id);
CREATE INDEX IF NOT EXISTS llm_events_model_id ON llm_events(resolved_model, requested_model, id);
CREATE TABLE IF NOT EXISTS llm_request_diagnostics (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,
    requested_model TEXT,
    resolved_model TEXT,
    provider TEXT,
    credential_id TEXT,
    prompt_cache_key_hash TEXT,
    system_instructions_hash TEXT,
    tools_schema_hash TEXT,
    reasoning_effort TEXT,
    input_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    output_tokens INTEGER,
    cache_breakpoint_mode TEXT,
    cache_breakpoint_location TEXT,
    previous_response_id_present INTEGER NOT NULL DEFAULT 0,
    response_id_hash TEXT,
    transport_mode TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_session_prefixes (
    proxy_user TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    prefix_hash TEXT NOT NULL,
    prefix_length INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (proxy_user, requested_model, prefix_hash)
);
CREATE INDEX IF NOT EXISTS llm_session_prefixes_updated
    ON llm_session_prefixes(updated_at DESC);
"""
_COLUMNS = (
    "request_id", "session_id", "timestamp", "proxy_user", "requested_model",
    "resolved_model", "provider", "credential_id", "role", "event_type",
    "content_text", "content_json", "status", "metadata_json",
)
_INSERT = f"INSERT INTO llm_events ({','.join(_COLUMNS)}) VALUES ({','.join('?' for _ in _COLUMNS)})"
_DIAGNOSTIC_COLUMNS = (
    "request_id", "session_id", "requested_model", "resolved_model", "provider",
    "credential_id", "prompt_cache_key_hash", "system_instructions_hash",
    "tools_schema_hash", "reasoning_effort", "input_tokens", "cache_read_tokens",
    "cache_write_tokens", "output_tokens", "cache_breakpoint_mode",
    "cache_breakpoint_location", "previous_response_id_present", "response_id_hash",
    "transport_mode", "created_at", "updated_at",
)
_DIAGNOSTIC_MIGRATIONS = {
    "session_id": "TEXT",
    "requested_model": "TEXT",
    "resolved_model": "TEXT",
    "provider": "TEXT",
    "credential_id": "TEXT",
    "prompt_cache_key_hash": "TEXT",
    "system_instructions_hash": "TEXT",
    "tools_schema_hash": "TEXT",
    "reasoning_effort": "TEXT",
    "input_tokens": "INTEGER",
    "cache_read_tokens": "INTEGER",
    "cache_write_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cache_breakpoint_mode": "TEXT",
    "cache_breakpoint_location": "TEXT",
    "previous_response_id_present": "INTEGER NOT NULL DEFAULT 0",
    "response_id_hash": "TEXT",
    "transport_mode": "TEXT",
    "created_at": "TEXT",
    "updated_at": "TEXT",
}
_UPSERT_DIAGNOSTICS = f"""
INSERT INTO llm_request_diagnostics ({','.join(_DIAGNOSTIC_COLUMNS)})
VALUES ({','.join('?' for _ in _DIAGNOSTIC_COLUMNS)})
ON CONFLICT(request_id) DO UPDATE SET
    session_id=excluded.session_id,
    requested_model=excluded.requested_model,
    resolved_model=excluded.resolved_model,
    provider=excluded.provider,
    credential_id=excluded.credential_id,
    prompt_cache_key_hash=excluded.prompt_cache_key_hash,
    system_instructions_hash=excluded.system_instructions_hash,
    tools_schema_hash=excluded.tools_schema_hash,
    reasoning_effort=excluded.reasoning_effort,
    input_tokens=excluded.input_tokens,
    cache_read_tokens=excluded.cache_read_tokens,
    cache_write_tokens=excluded.cache_write_tokens,
    output_tokens=excluded.output_tokens,
    cache_breakpoint_mode=excluded.cache_breakpoint_mode,
    cache_breakpoint_location=excluded.cache_breakpoint_location,
    previous_response_id_present=excluded.previous_response_id_present,
    response_id_hash=excluded.response_id_hash,
    transport_mode=excluded.transport_mode,
    updated_at=excluded.updated_at
"""
_UPSERT_PREFIX = """
INSERT INTO llm_session_prefixes
    (proxy_user, requested_model, prefix_hash, prefix_length, session_id, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(proxy_user, requested_model, prefix_hash) DO UPDATE SET
    session_id=excluded.session_id,
    prefix_length=excluded.prefix_length,
    updated_at=excluded.updated_at
"""
_STOP = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _enabled_from_env() -> bool:
    return os.getenv("LLM_TRACE_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off", "disabled"
    }


def _sanitize(value: Any, *, drop_private_fields: bool = False) -> Any:
    """Make values JSON-safe while removing secrets and optional private fields."""
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, drop_private_fields=drop_private_fields)
            for key, item in value.items()
            if str(key).strip().lower() not in _SECRET_KEYS
            and (
                not drop_private_fields
                or str(key).strip().lower() not in _PRIVATE_METADATA_KEYS
            )
            and not any(
                marker in str(key).strip().lower()
                for marker in ("authorization", "authentication", "password", "secret", "token", "cookie")
            )
            and not str(key).strip().lower().endswith(("-key", "_key"))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, drop_private_fields=drop_private_fields) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash_text(value: Any) -> str | None:
    """Return a deterministic digest without retaining an opaque identifier."""
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_structure(value: Any) -> str | None:
    """Return a stable digest of safe structured request material."""
    if value is None:
        return None
    encoded = json.dumps(
        _sanitize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transport_mode(
    payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> str | None:
    """Identify the protocol without retaining request routing metadata."""
    explicit = metadata.get("transport_mode") if metadata else None
    if isinstance(explicit, str) and explicit in _TRANSPORT_MODES:
        return explicit
    path = metadata.get("path") if metadata else None
    if path == "/v1/chat/completions":
        return "chat_completions"
    if path == "/v1/responses":
        return "responses"
    if path == "/v1/messages":
        return "anthropic_messages"
    if "input" in payload and "messages" not in payload:
        return "responses"
    if "messages" in payload:
        return "chat_completions"
    return None


def _system_instruction_material(payload: Mapping[str, Any]) -> list[Any]:
    """Extract only system/developer instruction structure for a one-way digest."""
    material: list[Any] = []
    for key in ("instructions", "system"):
        if payload.get(key) is not None:
            material.append({key: payload[key]})

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") in {"system", "developer"}:
                material.append(
                    {
                        "role": message.get("role"),
                        "content": message.get("content"),
                    }
                )

    input_items = payload.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if isinstance(item, Mapping) and item.get("role") in {"system", "developer"}:
                material.append(
                    {
                        "type": item.get("type"),
                        "role": item.get("role"),
                        "content": item.get("content"),
                    }
                )
    return material


def _reasoning_effort(payload: Mapping[str, Any]) -> str | None:
    """Normalize the common Chat Completions and Responses effort forms."""
    effort = payload.get("reasoning_effort")
    if effort is None:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, Mapping):
            effort = reasoning.get("effort")
    if effort is None:
        generation_config = payload.get("generation_config")
        if isinstance(generation_config, Mapping):
            effort = generation_config.get("reasoning_effort")
    if isinstance(effort, str):
        normalized = effort.strip()
        return normalized or None
    return None


def _breakpoint_diagnostics(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Extract only a public cache-control mode and scalar location."""
    breakpoint = payload.get("prompt_cache_breakpoint")
    if isinstance(breakpoint, Mapping):
        mode = breakpoint.get("mode", breakpoint.get("type"))
        location = breakpoint.get(
            "location", breakpoint.get("index", breakpoint.get("position"))
        )
        return (
            str(mode) if isinstance(mode, (str, int, float)) else None,
            str(location) if isinstance(location, (str, int, float)) else None,
        )
    if isinstance(breakpoint, (str, int, float)):
        return str(breakpoint), None

    options = payload.get("prompt_cache_options")
    if isinstance(options, Mapping):
        mode = options.get("mode")
        if isinstance(mode, (str, int, float)):
            return str(mode), None
    return None, None


def _token_count(value: Any) -> int | None:
    """Normalize a non-negative token counter while preserving explicit zero."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _first_token(usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in usage:
            return _token_count(usage.get(key))
    return None


def _usage_diagnostics(payload: Mapping[str, Any]) -> dict[str, int | None]:
    """Read OpenAI Chat/Responses and Anthropic usage without inventing values."""
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}

    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    details = prompt_details if isinstance(prompt_details, Mapping) else input_details
    if not isinstance(details, Mapping):
        details = {}

    return {
        "input_tokens": _first_token(usage, "input_tokens", "prompt_tokens"),
        "output_tokens": _first_token(usage, "output_tokens", "completion_tokens"),
        "cache_read_tokens": _first_token(
            usage, "cache_read_tokens", "cached_tokens", "cache_read_input_tokens"
        )
        if any(
            key in usage
            for key in ("cache_read_tokens", "cached_tokens", "cache_read_input_tokens")
        )
        else _first_token(details, "cached_tokens", "cache_read_tokens"),
        "cache_write_tokens": _first_token(
            usage,
            "cache_write_tokens",
            "cache_creation_tokens",
            "cache_creation_input_tokens",
        )
        if any(
            key in usage
            for key in (
                "cache_write_tokens",
                "cache_creation_tokens",
                "cache_creation_input_tokens",
            )
        )
        else _first_token(details, "cache_creation_tokens", "cache_write_tokens"),
    }


def _request_diagnostics(
    payload: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    breakpoint_mode, breakpoint_location = _breakpoint_diagnostics(payload)
    previous_response_id = payload.get("previous_response_id")
    system_instruction_material = _system_instruction_material(payload)
    return {
        "requested_model": str(payload["model"]) if payload.get("model") is not None else None,
        "prompt_cache_key_hash": _hash_text(payload.get("prompt_cache_key")),
        "system_instructions_hash": _hash_structure(system_instruction_material)
        if system_instruction_material
        else None,
        "tools_schema_hash": _hash_structure(payload["tools"]) if "tools" in payload else None,
        "reasoning_effort": _reasoning_effort(payload),
        "cache_breakpoint_mode": breakpoint_mode,
        "cache_breakpoint_location": breakpoint_location,
        "previous_response_id_present": int(
            isinstance(previous_response_id, str) and bool(previous_response_id.strip())
        ),
        "transport_mode": _transport_mode(payload, metadata),
    }


def _response_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "response_id_hash": _hash_text(payload.get("id")),
    }
    model = payload.get("model")
    if model is not None:
        result["response_model"] = str(model)
    result.update(
        {
            key: value
            for key, value in _usage_diagnostics(payload).items()
            if value is not None
        }
    )
    return result


def _text_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                elif part.get("type") in {"tool_use", "tool_result", "function_call"}:
                    pieces.append(_json(part) or "")
                else:
                    pieces.append(_json(part) or "")
            else:
                pieces.append(str(part))
        return "".join(pieces)
    return _json(content)


def _event(role: str, event_type: str, content: Any, *, status: str | None = None,
           metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    structured = content if isinstance(content, (Mapping, list)) else None
    return {
        "role": role,
        "event_type": event_type,
        "content_text": _text_content(content),
        "content_json": _json(structured),
        "status": status,
        "metadata_json": _json(_sanitize(metadata, drop_private_fields=True)) if metadata else None,
    }


def normalize_request(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize only the newly submitted tail of a chat request.

    Chat clients normally resend the complete history on every turn. Emitting
    only messages after the last assistant message avoids replaying old turns
    in the live tail while retaining new user and tool results.
    """
    events: list[dict[str, Any]] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        last_assistant = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, Mapping) and message.get("role") == "assistant"
            ),
            default=-1,
        )
        # Anthropic sends its top-level system prompt again with every turn.
        # Emit it only before any assistant history exists to avoid replaying
        # large or sensitive instructions in the live tail.
        system = payload.get("system")
        if system is not None and last_assistant < 0:
            events.append(_event("system", "message", system))
        new_messages = messages[last_assistant + 1 :]
        # On a fresh request, system instructions are context rather than a turn.
        if last_assistant < 0:
            new_messages = [
                message
                for message in new_messages
                if not isinstance(message, Mapping) or message.get("role") != "system"
            ]
        for message in new_messages:
            if not isinstance(message, Mapping):
                events.append(_event("unknown", "message", message))
                continue
            role = str(message.get("role") or "unknown")
            events.append(_event(role, "message", message.get("content")))
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    events.append(_event(role, "tool_call", call))
            # Anthropic tool_use blocks are also emitted separately for easy filtering.
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") in {"tool_use", "tool_result"}:
                        events.append(_event(role, str(block.get("type")), block))
    elif "input" in payload:
        input_value = payload.get("input")
        if isinstance(input_value, list):
            for item in input_value:
                if isinstance(item, Mapping) and item.get("type") == "message":
                    events.append(_event(str(item.get("role") or "user"), "message", item.get("content")))
                else:
                    events.append(_event("user", "input", item))
        else:
            events.append(_event("user", "input", input_value))
    return events


def _conversation_prefixes(payload: Mapping[str, Any]) -> list[str]:
    """Return cumulative hashes of canonical ordered conversation messages."""
    canonical_messages: list[Any] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping):
                if message.get("role") == "system":
                    continue
                canonical_messages.append(
                    {
                        key: _sanitize(message.get(key))
                        for key in ("role", "content", "tool_calls", "tool_call_id", "name")
                        if message.get(key) is not None
                    }
                )
            else:
                canonical_messages.append(_sanitize(message))
    elif "input" in payload:
        input_value = payload.get("input")
        if isinstance(input_value, list):
            canonical_messages.extend(_sanitize(input_value))
        else:
            canonical_messages.append(_sanitize(input_value))

    digest = hashlib.sha256()
    prefixes: list[str] = []
    for message in canonical_messages:
        encoded = (_json(message) or "null").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        prefixes.append(digest.hexdigest())
    return prefixes


def normalize_response(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize assembled OpenAI Chat/Responses (and Anthropic) responses.

    Streaming integrations should pass their assembled final response here, not
    individual deltas, so every logical response/tool call is persisted once.
    """
    events: list[dict[str, Any]] = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message") or choice.get("delta")
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "assistant")
            if message.get("content") is not None:
                events.append(_event(role, "response", message.get("content"), status=choice.get("finish_reason")))
            for call in message.get("tool_calls") or ():
                events.append(_event(role, "tool_call", call, status=choice.get("finish_reason")))
            function_call = message.get("function_call")
            if function_call is not None:
                events.append(_event(role, "tool_call", function_call, status=choice.get("finish_reason")))
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                events.append(_event("assistant", "response", item))
                continue
            item_type = str(item.get("type") or "response")
            role = str(item.get("role") or "assistant")
            if item_type in {"function_call", "tool_call", "computer_call", "custom_tool_call"}:
                events.append(_event(role, "tool_call", item, status=item.get("status")))
                continue
            content = item.get("content", item.get("text"))
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") in {"output_text", "text"}:
                        events.append(_event(role, "response", block.get("text"), status=item.get("status")))
                    else:
                        events.append(_event(role, item_type, block, status=item.get("status")))
            elif content is not None:
                events.append(_event(role, "response", content, status=item.get("status")))
    # Anthropic final Message shape.
    if not events and isinstance(payload.get("content"), list):
        for block in payload["content"]:
            kind = str(block.get("type") or "response") if isinstance(block, Mapping) else "response"
            role = str(payload.get("role") or "assistant")
            if isinstance(block, Mapping) and kind == "text":
                events.append(_event(role, "response", block.get("text"), status=payload.get("stop_reason")))
            elif kind == "tool_use":
                events.append(_event(role, "tool_call", block, status=payload.get("stop_reason")))
            else:
                events.append(_event(role, kind, block, status=payload.get("stop_reason")))
    return events


class LLMTraceRecorder:
    """Bounded, non-blocking producer with one SQLite writer thread."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None, *,
                 enabled: bool | None = None, queue_size: int = 4096) -> None:
        self.db_path = Path(db_path or os.getenv("LLM_TRACE_DB", _DEFAULT_DB))
        self.enabled = _enabled_from_env() if enabled is None else enabled
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, queue_size))
        self._closed = False
        self._dropped = 0
        self._prefix_lock = threading.Lock()
        self._prefix_sessions: dict[tuple[str, str, str], str] = {}
        self._thread: threading.Thread | None = None
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._writer, name="llm-trace-writer", daemon=True)
            self._thread.start()

    def infer_session_id(
        self, payload: Mapping[str, Any], proxy_user: str | None = None
    ) -> str:
        """Resolve the longest known non-system prefix using memory only."""
        user = proxy_user or "anonymous"
        model = str(payload.get("model") or "unknown")
        prefixes = _conversation_prefixes(payload)
        session_id = None
        with self._prefix_lock:
            for prefix_hash in reversed(prefixes):
                session_id = self._prefix_sessions.get((user, model, prefix_hash))
                if session_id is not None:
                    break
        if session_id is None:
            seed = prefixes[0] if prefixes else uuid.uuid4().hex
            material = f"{user}\0{model}\0{seed}".encode("utf-8")
            session_id = "auto-" + hashlib.sha256(material).hexdigest()[:16]

        if prefixes:
            with self._prefix_lock:
                for prefix_hash in prefixes:
                    self._prefix_sessions[(user, model, prefix_hash)] = session_id
            self._put(("prefixes", user, model, session_id, tuple(prefixes)))
        return session_id

    @property
    def dropped_events(self) -> int:
        """Number of events discarded because the bounded queue was full."""
        return self._dropped

    def begin_request(self, *, request_id: str | None = None, session_id: str | None = None,
                      proxy_user: str | None = None, requested_model: str | None = None,
                      metadata: Mapping[str, Any] | None = None) -> "LLMTraceContext":
        """Create a per-request context suitable for ``request.state.llm_trace``."""
        context = LLMTraceContext(
            recorder=self,
            request_id=request_id or uuid.uuid4().hex,
            session_id=session_id,
            proxy_user=proxy_user,
            requested_model=requested_model,
        )
        context._request_metadata = (
            _sanitize(metadata, drop_private_fields=True) if metadata else None
        )
        context._emit("request", "begin", None, metadata=metadata)
        return context

    def _put(self, row: tuple[Any, ...]) -> None:
        if not self.enabled or self._closed:
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued writes; return false on timeout (not for request paths)."""
        if not self.enabled:
            return True
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully drain and stop the writer; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            deadline = time.monotonic() + timeout
            try:
                self._queue.put(_STOP, timeout=max(0.0, deadline - time.monotonic()))
            except queue.Full:
                # The writer could not drain within the caller's deadline. It is
                # a daemon, so do not turn shutdown into an unbounded wait.
                return
            self._thread.join(max(0.0, deadline - time.monotonic()))

    def _writer(self) -> None:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.executescript(_SCHEMA)
            self._migrate_diagnostics_schema(connection)
            rows = connection.execute(
                "SELECT proxy_user, requested_model, prefix_hash, session_id "
                "FROM llm_session_prefixes ORDER BY prefix_length"
            )
            with self._prefix_lock:
                for user, model, prefix_hash, session_id in rows:
                    self._prefix_sessions[(user, model, prefix_hash)] = session_id
            connection.commit()
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        break
                    self._write_item(connection, item)
                    # Drain a batch to reduce fsync/lock churn.
                    while True:
                        try:
                            extra = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if extra is _STOP:
                            self._queue.task_done()
                            connection.commit()
                            return
                        self._write_item(connection, extra)
                        self._queue.task_done()
                    connection.commit()
                finally:
                    self._queue.task_done()
        finally:
            connection.commit()
            connection.close()

    @staticmethod
    def _migrate_diagnostics_schema(connection: sqlite3.Connection) -> None:
        """Bring an early or manually-created diagnostics table forward safely."""
        existing = {
            row[1]
            for row in connection.execute("PRAGMA table_info(llm_request_diagnostics)")
        }
        for name, declaration in _DIAGNOSTIC_MIGRATIONS.items():
            if name not in existing:
                connection.execute(
                    f"ALTER TABLE llm_request_diagnostics ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS llm_request_diagnostics_session "
            "ON llm_request_diagnostics(session_id, updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS llm_request_diagnostics_model "
            "ON llm_request_diagnostics(resolved_model, requested_model, updated_at DESC)"
        )

    @staticmethod
    def _write_item(connection: sqlite3.Connection, item: object) -> None:
        if isinstance(item, tuple) and item and item[0] == "prefixes":
            _, user, model, session_id, prefixes = item
            timestamp = _utc_now()
            connection.executemany(
                _UPSERT_PREFIX,
                (
                    (user, model, prefix_hash, length, session_id, timestamp)
                    for length, prefix_hash in enumerate(prefixes, start=1)
                ),
            )
            return
        if isinstance(item, tuple) and item and item[0] == "diagnostics":
            connection.execute(_UPSERT_DIAGNOSTICS, item[1])
            return
        connection.execute(_INSERT, item)  # type: ignore[arg-type]


@dataclass(slots=True)
class LLMTraceContext:
    """Mutable request-local trace state; hook methods never block on SQLite."""

    recorder: LLMTraceRecorder
    request_id: str
    session_id: str | None = None
    proxy_user: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    provider: str | None = None
    credential_id: str | None = None
    _request_metadata: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _diagnostic_created_at: str = field(default_factory=_utc_now, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def _queue_diagnostics(self) -> None:
        """Queue one complete, privacy-safe request diagnostic snapshot."""
        updated_at = _utc_now()
        diagnostic = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "provider": self.provider,
            "credential_id": self.credential_id,
            "prompt_cache_key_hash": None,
            "system_instructions_hash": None,
            "tools_schema_hash": None,
            "reasoning_effort": None,
            "input_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "output_tokens": None,
            "cache_breakpoint_mode": None,
            "cache_breakpoint_location": None,
            "previous_response_id_present": 0,
            "response_id_hash": None,
            "transport_mode": None,
            "created_at": self._diagnostic_created_at,
            "updated_at": updated_at,
        }
        diagnostic.update(self._diagnostics)
        diagnostic.update(
            {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "requested_model": self.requested_model,
                "resolved_model": self.resolved_model,
                "provider": self.provider,
                "credential_id": self.credential_id,
                "created_at": self._diagnostic_created_at,
                "updated_at": updated_at,
            }
        )
        self.recorder._put(
            ("diagnostics", tuple(diagnostic[column] for column in _DIAGNOSTIC_COLUMNS))
        )

    def _emit(self, role: str, event_type: str, content: Any, *, status: str | None = None,
              metadata: Mapping[str, Any] | None = None) -> None:
        event = _event(role, event_type, content, status=status, metadata=metadata)
        self.recorder._put((
            self.request_id, self.session_id, _utc_now(), self.proxy_user,
            self.requested_model, self.resolved_model, self.provider, self.credential_id,
            event["role"], event["event_type"], event["content_text"],
            event["content_json"], event["status"], event["metadata_json"],
        ))

    def request(self, payload: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> None:
        """Record the raw body and normalized input messages (never headers)."""
        if self.requested_model is None and payload.get("model") is not None:
            self.requested_model = str(payload["model"])
        if metadata:
            self._request_metadata = _sanitize(metadata, drop_private_fields=True)
        request_metadata = self._request_metadata
        self._emit("request", "request_payload", payload, metadata=metadata)
        self._diagnostics.update(_request_diagnostics(payload, request_metadata))
        self._queue_diagnostics()
        for event in normalize_request(payload):
            self._emit(event["role"], event["event_type"],
                       json.loads(event["content_json"]) if event["content_json"] else event["content_text"],
                       status=event["status"], metadata=metadata)

    def cache_key_selected(self, cache_key: str | None) -> None:
        """Record only the digest of a cache key derived after request tracing."""
        cache_key_hash = _hash_text(cache_key)
        if cache_key_hash is None:
            return
        self._diagnostics["prompt_cache_key_hash"] = cache_key_hash
        self._queue_diagnostics()

    def credential_selected(self, *, resolved_model: str | None = None,
                            provider: str | None = None, credential_id: str | None = None,
                            metadata: Mapping[str, Any] | None = None) -> None:
        """Attach safe routing labels and record selection; never pass a raw key."""
        self.resolved_model = resolved_model or self.resolved_model
        self.provider = provider or self.provider
        self.credential_id = credential_id or self.credential_id
        self._queue_diagnostics()
        self._emit("proxy", "credential_selected", None, metadata=metadata)

    def response(self, payload: Any, *, status: str | None = None,
                 metadata: Mapping[str, Any] | None = None) -> None:
        """Record the raw body and normalized final response once."""
        if isinstance(payload, Mapping):
            diagnostics = _response_diagnostics(payload)
            response_model = diagnostics.pop("response_model", None)
            if self.resolved_model is None and response_model is not None:
                self.resolved_model = response_model
            self._diagnostics.update(diagnostics)
            self._queue_diagnostics()
        self._emit("proxy", "response_payload", payload, status=status, metadata=metadata)
        events = normalize_response(payload) if isinstance(payload, Mapping) else [_event("assistant", "response", payload)]
        for event in events:
            content = json.loads(event["content_json"]) if event["content_json"] else event["content_text"]
            self._emit(event["role"], event["event_type"], content,
                       status=status or event["status"], metadata=metadata)

    def error(self, error: BaseException | str, *, status: str = "error",
              metadata: Mapping[str, Any] | None = None) -> None:
        """Record a sanitized error description without exception internals."""
        self._emit("proxy", "error", str(error), status=status, metadata=metadata)

    def completed(self, *, status: str = "completed",
                  metadata: Mapping[str, Any] | None = None) -> None:
        """Record request completion once."""
        if not self._finished:
            self._finished = True
            self._queue_diagnostics()
            self._emit("proxy", "completed", None, status=status, metadata=metadata)


_global_lock = threading.Lock()
_global_recorder: LLMTraceRecorder | None = None


def get_recorder() -> LLMTraceRecorder:
    """Return the lazily-created process-wide recorder."""
    global _global_recorder
    if _global_recorder is None:
        with _global_lock:
            if _global_recorder is None:
                _global_recorder = LLMTraceRecorder()
    return _global_recorder


def begin_request(**kwargs: Any) -> LLMTraceContext:
    """Shortcut for ``get_recorder().begin_request(**kwargs)``."""
    return get_recorder().begin_request(**kwargs)


def close_recorder(timeout: float = 5.0) -> None:
    """Close and clear the process-wide recorder (also registered with atexit)."""
    global _global_recorder
    with _global_lock:
        recorder, _global_recorder = _global_recorder, None
    if recorder is not None:
        recorder.close(timeout)


atexit.register(close_recorder)

__all__ = [
    "LLMTraceContext", "LLMTraceRecorder", "begin_request", "close_recorder",
    "get_recorder", "normalize_request", "normalize_response",
]
