#!/usr/bin/env python3
"""Poll and render the SQLite LLM event trace.

Human output is deliberately stable and pipe-friendly::

    ses [session-or-request] [model] [credential-or--] [user] role: content

Every event occupies exactly one physical line. Use ``--json`` for one compact
JSON object per line. SQLite readers coexist with the recorder's WAL writer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
from typing import Any, Iterable

_DEFAULT_DB = "usage/llm-requests.sqlite3"
_STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def _content(row: dict[str, Any]) -> str:
    text = row.get("content_text")
    if text is not None:
        return _one_line(text)
    raw = row.get("content_json")
    if raw:
        try:
            return json.dumps(json.loads(raw), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return _one_line(raw)
    status = row.get("status")
    if status:
        return _one_line(status)
    return ""


def render_human(row: dict[str, Any]) -> str:
    """Render exactly one physical line for an event row."""
    session = row.get("session_id") or row.get("request_id") or "-"
    model = row.get("resolved_model") or row.get("requested_model") or "-"
    credential = row.get("credential_id") or "-"
    user = row.get("proxy_user") or "-"
    role = row.get("role") or row.get("event_type") or "event"
    return (
        f"ses [{_one_line(session)}] [{_one_line(model)}] "
        f"[{_one_line(credential)}] [{_one_line(user)}] {_one_line(role)}: {_content(row)}"
    )


def render_json(row: dict[str, Any]) -> str:
    """Render a compact JSON object, decoding stored JSON columns."""
    result = dict(row)
    for source, target in (("content_json", "content"), ("metadata_json", "metadata")):
        raw = result.pop(source, None)
        if raw is not None:
            try:
                result[target] = json.loads(raw)
            except (TypeError, ValueError):
                result[target] = raw
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def iter_rows(connection: sqlite3.Connection, *, since_id: int = 0,
              user: str | None = None, model: str | None = None,
              session: str | None = None) -> Iterable[dict[str, Any]]:
    """Return events newer than ``since_id`` matching exact optional labels."""
    clauses = ["id > ?"]
    values: list[Any] = [since_id]
    if user is not None:
        clauses.append("proxy_user = ?")
        values.append(user)
    if model is not None:
        clauses.append("(resolved_model = ? OR requested_model = ?)")
        values.extend((model, model))
    if session is not None:
        clauses.append("(session_id = ? OR (session_id IS NULL AND request_id = ?))")
        values.extend((session, session))
    sql = "SELECT * FROM llm_events WHERE " + " AND ".join(clauses) + " ORDER BY id"
    for row in connection.execute(sql, values):
        yield dict(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tail SQLite LLM trace events")
    parser.add_argument("--db", default=os.getenv("LLM_TRACE_DB", _DEFAULT_DB),
                        help="trace database (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="emit compact JSON lines")
    parser.add_argument("--no-follow", action="store_true", help="print available rows and exit")
    parser.add_argument("--since-id", type=int, default=0, metavar="ID",
                        help="only rows with id greater than ID")
    parser.add_argument("--user", help="exact proxy user filter")
    parser.add_argument("--model", help="exact resolved/requested model filter")
    parser.add_argument("--session", help="exact session or request id filter")
    parser.add_argument("--poll-interval", type=float, default=0.25, metavar="SECONDS",
                        help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.since_id < 0:
        raise SystemExit("--since-id must be non-negative")
    if args.poll_interval <= 0:
        raise SystemExit("poll interval must be positive")

    path = Path(args.db)
    last_id = args.since_id
    connection: sqlite3.Connection | None = None
    render = render_json if args.json else render_human
    while not _STOP:
        if connection is None:
            try:
                connection = _connect(path)
            except sqlite3.OperationalError as error:
                if args.no_follow:
                    print(f"llm_tail: {error}", file=sys.stderr)
                    return 1
                time.sleep(args.poll_interval)
                continue
        try:
            rows = list(iter_rows(connection, since_id=last_id, user=args.user,
                                  model=args.model, session=args.session))
        except sqlite3.OperationalError as error:
            connection.close()
            connection = None
            if args.no_follow:
                print(f"llm_tail: {error}", file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
            continue
        for row in rows:
            last_id = max(last_id, int(row["id"]))
            try:
                print(render(row), flush=True)
            except BrokenPipeError:
                if connection is not None:
                    connection.close()
                return 0
        if args.no_follow:
            break
        time.sleep(args.poll_interval)
    if connection is not None:
        connection.close()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Avoid Python's shutdown flush reporting another broken-pipe exception.
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
