from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from .models import Decision, ExperienceSeed


class SQLiteSeedStore:
    def __init__(self, path: str | Path = ".alaya/seeds.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS seeds (id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), check_same_thread=False)

    def save(self, seed: ExperienceSeed) -> None:
        payload = json.dumps(seed.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO seeds(id,payload,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (seed.id, payload, seed.updated_at.isoformat()),
            )

    def get(self, seed_id: str) -> ExperienceSeed | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM seeds WHERE id=?", (seed_id,)).fetchone()
        return ExperienceSeed.from_dict(json.loads(row[0])) if row else None

    def list(self) -> list[ExperienceSeed]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM seeds ORDER BY updated_at DESC, id").fetchall()
        return [ExperienceSeed.from_dict(json.loads(row[0])) for row in rows]

    def delete(self, seed_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM seeds WHERE id=?", (seed_id,))
        return cursor.rowcount > 0

    def list_active(self, since_days: int = 180) -> list[ExperienceSeed]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM seeds WHERE json_extract(payload, '$.status') = 'active' "
                "AND json_extract(payload, '$.updated_at') >= date('now', ? || ' days') "
                "ORDER BY updated_at DESC, id",
                (str(-since_days),),
            ).fetchall()
        return [ExperienceSeed.from_dict(json.loads(row[0])) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM seeds").fetchone()
        return row[0] if row else 0

    def iter_all(self, batch_size: int = 100) -> Iterator[list[ExperienceSeed]]:
        with self._connect() as connection:
            cursor = connection.execute("SELECT payload FROM seeds ORDER BY updated_at DESC, id")
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [ExperienceSeed.from_dict(json.loads(r[0])) for r in rows]

    def close(self) -> None:
        """Release any lingering resources. Safe to call multiple times."""
        pass

    def export_json(self) -> str:
        result: list[dict[str, object]] = []
        for batch in self.iter_all(200):
            result.extend(seed.to_dict() for seed in batch)
        return json.dumps(result, ensure_ascii=False, indent=2)


class DecisionStore:
    """Lightweight decision log backed by the same SQLite database."""

    def __init__(self, path: str | Path = ".alaya/seeds.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), check_same_thread=False)

    def save(self, decision: Decision) -> None:
        payload = json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO decisions(id,payload,created_at) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at",
                (decision.id, payload, decision.created_at.isoformat()),
            )

    def get(self, decision_id: str) -> Decision | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        from datetime import datetime as dt
        return Decision(
            id=data["id"], context=data["context"],
            chosen_seeds=tuple(data["chosen_seeds"]),
            excluded_seeds=tuple(data["excluded_seeds"]),
            action=data["action"],
            created_at=dt.fromisoformat(data["created_at"]),
        )

    def close(self) -> None:
        pass

