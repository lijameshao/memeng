from memeng.database.utils import DB
from memeng.utils.common import now_iso


def record_injection(run_id: str, memory_ids: list[str]) -> None:
    DB.conn.executemany(
        "INSERT OR IGNORE INTO injections(run_id, memory_id, created_at) VALUES (?, ?, ?)",
        [(run_id, memory_id, now_iso()) for memory_id in memory_ids],
    )


def injected_ids(run_id: str) -> set[str]:
    rows = DB.conn.fetchall("SELECT memory_id FROM injections WHERE run_id = ?", (run_id,))
    return {row["memory_id"] for row in rows}
