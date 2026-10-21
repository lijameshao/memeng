from memeng.config import CONFIG
from memeng.database.utils import DB, placeholders
from memeng.models.messages import Message
from memeng.utils.common import now_iso


def insert_messages(msgs: list[Message]) -> list[str]:
    DB.conn.executemany(
        """INSERT OR IGNORE INTO messages(
             id, role, content, created_at, processed_at,
             user_id, project_id, agent_id, run_id, gate_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                m.id,
                m.role,
                m.content,
                m.created_at,
                m.processed_at,
                m.user_id,
                m.project_id,
                m.agent_id,
                m.run_id,
                m.gate_count,
            )
            for m in msgs
        ],
    )
    return [m.id for m in msgs]


def get_unprocessed_messages(project_id: str, user_id: str) -> list[Message]:
    rows = DB.conn.fetchall(
        """SELECT id, role, content, created_at, processed_at,
                  user_id, project_id, agent_id, run_id, gate_count
             FROM messages
            WHERE project_id = ? AND user_id = ? AND processed_at IS NULL
            ORDER BY created_at, rowid""",
        (project_id, user_id),
    )
    return [Message(**dict(row)) for row in rows]


def mark_processed(message_ids: list[str]) -> None:
    at = now_iso()
    DB.conn.executemany(
        "UPDATE messages SET processed_at = ? WHERE id = ?",
        [(at, message_id) for message_id in message_ids],
    )


def clear_processed(message_ids: list[str]) -> None:
    DB.conn.executemany(
        "UPDATE messages SET processed_at = NULL WHERE id = ?",
        [(message_id,) for message_id in message_ids],
    )


def release_gated(message_ids: list[str]) -> int:
    """A gate no means "not enough yet", so the turns return; the count stops
    a window that never earns a yes from returning forever.
    """
    if not message_ids:
        return 0
    marks = placeholders(message_ids)
    DB.conn.execute(
        f"UPDATE messages SET processed_at = NULL, gate_count = gate_count + 1"
        f" WHERE id IN ({marks})",
        message_ids,
    )
    return DB.conn.execute(
        f"UPDATE messages SET processed_at = ? WHERE gate_count >= ? AND id IN ({marks})",
        [now_iso(), CONFIG.gate_retire_after, *message_ids],
    ).rowcount
