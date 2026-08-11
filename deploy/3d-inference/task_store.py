"""SQLite-backed task state for the 3D inference queue."""

from __future__ import annotations

# Vendored with the queue deployment snapshot for reproducible installation.

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional


TERMINAL_STATUSES = ("completed", "error", "cancelled")
RECOVERABLE_STATUSES = ("queued", "dispatching", "processing")


def compute_fingerprint(task_id: str, model: str, params: dict, image_sha256: str) -> str:
    """Deterministic fingerprint for idempotency checking.

    Two submissions with identical task_id, model, quality params, and image
    bytes produce the same fingerprint → safe to return the existing task.
    """
    raw = f"{task_id}|{model}|{json.dumps(params, sort_keys=True)}|{image_sha256}"
    return hashlib.sha256(raw.encode()).hexdigest()


class TaskStore:
    """Small durable store with one SQLite connection per operation."""

    _UPDATABLE_FIELDS = {
        "status",
        "backend_uid",
        "result_path",
        "error",
        "started_at",
        "completed_at",
        "updated_at",
    }

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        self._schema_lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    uid TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    backend_uid TEXT,
                    result_path TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    request_fingerprint TEXT,
                    image_sha256 TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status_created
                ON tasks(status, created_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_terminal_updated
                ON tasks(status, updated_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_created_desc
                ON tasks(created_at DESC);
                """
            )
            # Migration: add columns and indexes if missing on older databases
            mig_ok = True
            for col_sql in [
                "ALTER TABLE tasks ADD COLUMN request_fingerprint TEXT",
                "ALTER TABLE tasks ADD COLUMN image_sha256 TEXT",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass  # column already exists
                except Exception:
                    mig_ok = False
            if mig_ok:
                # Only create indexes referencing new columns after migration
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_tasks_fingerprint ON tasks(request_fingerprint)",
                ]:
                    try:
                        conn.execute(idx_sql)
                    except sqlite3.OperationalError:
                        pass

    @staticmethod
    def _row_to_task(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        task = dict(row)
        task["params"] = json.loads(task.pop("params_json"))
        return task

    def create_task(
        self,
        uid: str,
        model: str,
        params: dict[str, Any],
        input_path: str,
        request_fingerprint: Optional[str] = None,
        image_sha256: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    uid, model, status, params_json, input_path,
                    request_fingerprint, image_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    model,
                    json.dumps(params, separators=(",", ":"), sort_keys=True),
                    os.path.abspath(input_path),
                    request_fingerprint,
                    image_sha256,
                    now,
                    now,
                ),
            )

    def find_by_fingerprint(self, fingerprint: str) -> Optional[dict[str, Any]]:
        """Return existing task with matching fingerprint, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE request_fingerprint = ? ORDER BY created_at DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
        return self._row_to_task(row)

    def get(self, uid: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE uid = ?", (uid,)).fetchone()
        return self._row_to_task(row)

    def update(self, uid: str, **fields: Any) -> bool:
        unknown = set(fields) - self._UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported task fields: {sorted(unknown)}")
        if not fields:
            return False
        fields.setdefault("updated_at", time.time())
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [uid]
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE tasks SET {assignments} WHERE uid = ?", values
            )
        return cursor.rowcount == 1

    def claim(self, uid: str) -> Optional[dict[str, Any]]:
        """Move a queued task to dispatching and increment its attempt count."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM tasks WHERE uid = ?", (uid,)).fetchone()
            if row is None or row["status"] in TERMINAL_STATUSES:
                return None
            if row["status"] == "queued":
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'dispatching', attempts = attempts + 1,
                        started_at = COALESCE(started_at, ?), updated_at = ?
                    WHERE uid = ?
                    """,
                    (now, now, uid),
                )
                row = conn.execute(
                    "SELECT * FROM tasks WHERE uid = ?", (uid,)
                ).fetchone()
        return self._row_to_task(row)

    def cancel(self, uid: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = 'cancelled', completed_at = ?, updated_at = ?
                WHERE uid = ? AND status = 'queued'
                """,
                (now, now, uid),
            )
        return cursor.rowcount == 1

    def retry(self, uid: str) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE tasks
                SET status = 'dispatching', backend_uid = NULL,
                    attempts = attempts + 1, error = NULL, updated_at = ?
                WHERE uid = ? AND status IN ('dispatching', 'processing')
                """,
                (now, uid),
            )
            row = conn.execute("SELECT * FROM tasks WHERE uid = ?", (uid,)).fetchone()
        return self._row_to_task(row)

    def list_by_statuses(self, statuses: Iterable[str]) -> list[dict[str, Any]]:
        statuses = tuple(statuses)
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                """,
                statuses,
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def recoverable(self) -> list[dict[str, Any]]:
        return self.list_by_statuses(RECOVERABLE_STATUSES)

    def queued(self) -> list[dict[str, Any]]:
        return self.list_by_statuses(("queued",))

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def list_recent(self, limit: int = 50, model: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
        """Return the most recent tasks, newest first."""
        conditions = []
        params_list = []
        if model:
            conditions.append("model = ?")
            params_list.append(model)
        if status:
            conditions.append("status = ?")
            params_list.append(status)
        where = " AND ".join(conditions) if conditions else "1"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE {where} ORDER BY created_at DESC LIMIT ?",
                (*params_list, limit),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def cleanup_candidates(self, cutoff: float) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                WHERE status IN ({placeholders}) AND updated_at < ?
                ORDER BY updated_at ASC
                """,
                (*TERMINAL_STATUSES, cutoff),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def delete(self, uid: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE uid = ?", (uid,))
        return cursor.rowcount == 1

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
