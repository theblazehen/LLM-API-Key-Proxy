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


def _sanitize(value: Any) -> Any:
    """Make metadata JSON-safe while removing keys likely to contain secrets."""
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).strip().lower() not in _SECRET_KEYS
            and not any(
                marker in str(key).strip().lower()
                for marker in ("authorization", "authentication", "password", "secret", "token", "cookie")
            )
            and not str(key).strip().lower().endswith(("-key", "_key"))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


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
        "metadata_json": _json(_sanitize(metadata)) if metadata else None,
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
    _finished: bool = field(default=False, init=False, repr=False)

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
        """Record normalized OpenAI/Anthropic input messages (never headers)."""
        if self.requested_model is None and payload.get("model") is not None:
            self.requested_model = str(payload["model"])
        for event in normalize_request(payload):
            self._emit(event["role"], event["event_type"],
                       json.loads(event["content_json"]) if event["content_json"] else event["content_text"],
                       status=event["status"], metadata=metadata)

    def credential_selected(self, *, resolved_model: str | None = None,
                            provider: str | None = None, credential_id: str | None = None,
                            metadata: Mapping[str, Any] | None = None) -> None:
        """Attach safe routing labels and record selection; never pass a raw key."""
        self.resolved_model = resolved_model or self.resolved_model
        self.provider = provider or self.provider
        self.credential_id = credential_id or self.credential_id
        self._emit("proxy", "credential_selected", None, metadata=metadata)

    def response(self, payload: Mapping[str, Any] | str, *, status: str | None = None,
                 metadata: Mapping[str, Any] | None = None) -> None:
        """Record a final response; streaming callers pass the assembled result once."""
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
