from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ExperienceSeed


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

    def close(self) -> None:
        """Release any lingering resources. Safe to call multiple times."""
        pass

    def export_json(self) -> str:
        return json.dumps([seed.to_dict() for seed in self.list()], ensure_ascii=False, indent=2)

