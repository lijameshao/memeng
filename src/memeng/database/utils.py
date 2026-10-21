import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from memeng.config import CONFIG, Config
from memeng.database.migrations import MIGRATIONS


class DbManager:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> None:
        self._conn.executemany(sql, rows)

    def fetchall(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: Sequence = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def meta_get(self, key: str) -> str | None:
        row = self.fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def close(self) -> None:
        self._conn.close()

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    @contextmanager
    def transaction(self):
        self.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            self.execute("ROLLBACK")
            raise
        self.execute("COMMIT")


class Db:
    def __init__(self):
        self._conn: DbManager | None = None

    @property
    def conn(self) -> DbManager:
        if self._conn is None:
            raise RuntimeError("no open database handle; call DB.open(config)")
        return self._conn

    def open(self, config: Config) -> None:
        CONFIG.bind(config)
        self._conn = connect()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None


DB = Db()


def connect() -> DbManager:
    path = CONFIG.db_path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = _open(path, busy_timeout=5000)
    # WAL is a property of the file, so only the writer ever sets it: the
    # read-only connection cannot, and does not need to.
    db.execute("PRAGMA journal_mode=WAL")
    _migrate(db)
    db.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('embed_model', ?)",
        (CONFIG.embed_model,),
    )
    return db



def placeholders(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


def _open(path: Path, busy_timeout: int) -> DbManager:
    conn = sqlite3.connect(path, isolation_level=None, timeout=0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
    conn.execute("PRAGMA foreign_keys=ON")
    return DbManager(conn)


def _migrate(db: DbManager) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    if int(db.meta_get("schema_version") or 0) >= max(MIGRATIONS):
        return

    with db.transaction():
        # Detached captures can open a new store at the same moment, so the version
        # is read again under the write lock: whoever loses the race applies nothing.
        current = int(db.meta_get("schema_version") or 0)
        for version in sorted(v for v in MIGRATIONS if v > current):
            for statement in MIGRATIONS[version]:
                db.execute(statement)
            db.meta_set("schema_version", str(version))
