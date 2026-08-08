"""Change-only persistence for Codex weekly quota observations."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CodexWeeklyObservation:
    """One distinct weekly quota state for a stable Codex account."""

    id: int
    stable_account_id: str
    email: str | None
    used_percent_text: str
    remaining_percent_text: str
    used_percent: float
    remaining_percent: float
    reset_at: float
    reset_credit_metadata: Any
    source_timestamp: float
    observed_at: float
    fingerprint: str


@dataclass(frozen=True)
class BoundaryObservations:
    """The trusted boundary baseline and changes after that boundary."""

    baseline: CodexWeeklyObservation | None
    changes: tuple[CodexWeeklyObservation, ...]


class QuotaObservationStore:
    """Persist distinct Codex weekly quota states in a caller-owned SQLite DB."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record(
        self,
        *,
        stable_account_id: str,
        email: str | None,
        used_percent: str | int | float | Decimal,
        remaining_percent: str | int | float | Decimal,
        reset_at: int | float,
        reset_credit_metadata: Mapping[str, Any] | Sequence[Any] | None,
        source_timestamp: int | float,
        observed_at: int | float,
    ) -> bool:
        """Insert an observation only when quota/reset/credit state changed."""
        if not stable_account_id:
            raise ValueError("stable_account_id must not be empty")

        used_text = _decimal_text(used_percent)
        remaining_text = _decimal_text(remaining_percent)
        credit_json = _canonical_json(reset_credit_metadata)
        reset_value = float(reset_at)
        fingerprint = _fingerprint(
            used_text,
            remaining_text,
            reset_value,
            credit_json,
        )

        with closing(self._connect()) as connection:
            with connection:
                # Acquire the SQLite write reservation before checking the latest
                # fingerprint.  A deferred transaction lets two connections read
                # the same latest state and then both insert it.
                connection.execute("BEGIN IMMEDIATE")
                latest = connection.execute(
                """
                SELECT fingerprint
                FROM codex_weekly_observations
                WHERE stable_account_id = ?
                ORDER BY observed_at DESC, id DESC
                LIMIT 1
                """,
                (stable_account_id,),
                ).fetchone()
                if latest is not None and latest["fingerprint"] == fingerprint:
                    return False
                cursor = connection.execute(
                """
                INSERT INTO codex_weekly_observations (
                    stable_account_id,
                    email,
                    used_percent_text,
                    remaining_percent_text,
                    used_percent,
                    remaining_percent,
                    reset_at,
                    reset_credit_json,
                    source_timestamp,
                    observed_at,
                    fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_account_id,
                    email,
                    used_text,
                    remaining_text,
                    float(Decimal(used_text)),
                    float(Decimal(remaining_text)),
                    reset_value,
                    credit_json,
                    float(source_timestamp),
                    float(observed_at),
                    fingerprint,
                ),
                )
                return cursor.rowcount == 1

    def latest(self, stable_account_id: str) -> CodexWeeklyObservation | None:
        """Return the most recently observed distinct state for an account."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM codex_weekly_observations
                WHERE stable_account_id = ?
                ORDER BY observed_at DESC, id DESC
                LIMIT 1
                """,
                (stable_account_id,),
            ).fetchone()
        return _observation(row) if row is not None else None

    def history(
        self,
        stable_account_id: str,
        *,
        start_at: int | float | None = None,
        end_at: int | float | None = None,
    ) -> list[CodexWeeklyObservation]:
        """Return chronological distinct states in an inclusive time range."""
        clauses = ["stable_account_id = ?"]
        parameters: list[Any] = [stable_account_id]
        if start_at is not None:
            clauses.append("observed_at >= ?")
            parameters.append(float(start_at))
        if end_at is not None:
            clauses.append("observed_at <= ?")
            parameters.append(float(end_at))

        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM codex_weekly_observations
                WHERE {' AND '.join(clauses)}
                ORDER BY observed_at ASC, id ASC
                """,
                parameters,
            ).fetchall()
        return [_observation(row) for row in rows]

    def latest_at_or_before(
        self,
        stable_account_id: str,
        timestamp: int | float,
    ) -> CodexWeeklyObservation | None:
        """Return the last distinct state observed no later than a timestamp."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM codex_weekly_observations
                WHERE stable_account_id = ? AND observed_at <= ?
                ORDER BY observed_at DESC, id DESC
                LIMIT 1
                """,
                (stable_account_id, float(timestamp)),
            ).fetchone()
        return _observation(row) if row is not None else None

    def observations_around_boundary(
        self,
        stable_account_id: str,
        boundary: int | float,
        *,
        end_at: int | float | None = None,
    ) -> BoundaryObservations:
        """Return the baseline at/before a boundary and later state changes."""
        boundary_value = float(boundary)
        baseline = self.latest_at_or_before(stable_account_id, boundary_value)
        clauses = ["stable_account_id = ?", "observed_at > ?"]
        parameters: list[Any] = [stable_account_id, boundary_value]
        if end_at is not None:
            clauses.append("observed_at <= ?")
            parameters.append(float(end_at))

        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM codex_weekly_observations
                WHERE {' AND '.join(clauses)}
                ORDER BY observed_at ASC, id ASC
                """,
                parameters,
            ).fetchall()
        return BoundaryObservations(
            baseline=baseline,
            changes=tuple(_observation(row) for row in rows),
        )

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS codex_weekly_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stable_account_id TEXT NOT NULL,
                        email TEXT,
                        used_percent_text TEXT NOT NULL,
                        remaining_percent_text TEXT NOT NULL,
                        used_percent REAL NOT NULL,
                        remaining_percent REAL NOT NULL,
                        reset_at REAL NOT NULL,
                        reset_credit_json TEXT NOT NULL,
                        source_timestamp REAL NOT NULL,
                        observed_at REAL NOT NULL,
                        fingerprint TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_codex_weekly_account_observed
                    ON codex_weekly_observations (
                        stable_account_id,
                        observed_at,
                        id
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection


def _decimal_text(value: str | int | float | Decimal) -> str:
    if isinstance(value, bool):
        raise TypeError("quota percentage must be numeric")
    try:
        decimal_value = Decimal(value) if not isinstance(value, float) else Decimal(str(value))
    except Exception as error:
        raise ValueError(f"invalid quota percentage: {value!r}") from error
    if not decimal_value.is_finite():
        raise ValueError("quota percentage must be finite")
    return str(value) if isinstance(value, str) else str(decimal_value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    used_percent_text: str,
    remaining_percent_text: str,
    reset_at: float,
    reset_credit_json: str,
) -> str:
    state = _canonical_json(
        {
            "remaining_percent": remaining_percent_text,
            "reset_at": reset_at,
            "reset_credits": json.loads(reset_credit_json),
            "used_percent": used_percent_text,
        }
    )
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _observation(row: sqlite3.Row) -> CodexWeeklyObservation:
    return CodexWeeklyObservation(
        id=int(row["id"]),
        stable_account_id=str(row["stable_account_id"]),
        email=row["email"],
        used_percent_text=str(row["used_percent_text"]),
        remaining_percent_text=str(row["remaining_percent_text"]),
        used_percent=float(row["used_percent"]),
        remaining_percent=float(row["remaining_percent"]),
        reset_at=float(row["reset_at"]),
        reset_credit_metadata=json.loads(row["reset_credit_json"]),
        source_timestamp=float(row["source_timestamp"]),
        observed_at=float(row["observed_at"]),
        fingerprint=str(row["fingerprint"]),
    )
