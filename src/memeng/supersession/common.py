import uuid

from memeng.database.utils import DB, placeholders
from memeng.enums.common import ConflictSource, SupersessionStatus
from memeng.models.supersession import SupersessionRow
from memeng.utils.common import now_iso


def insert_supersession(
    memory_id: str, supersedes: str, proposed_by: ConflictSource
) -> None:
    # Both the extractor and the read-time cosine pass can raise the same pair,
    # and a re-proposal must not reset a verdict somebody already recorded.
    DB.conn.execute(
        """INSERT OR IGNORE INTO supersession(
             id, memory_id, supersedes, status, proposed_by, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            memory_id,
            supersedes,
            SupersessionStatus.UNRESOLVED,
            proposed_by,
            now_iso(),
        ),
    )


def set_supersession_status(
    id: str, status: SupersessionStatus, resolved_by: str
) -> int:
    if status not in (SupersessionStatus.SUPERSEDED, SupersessionStatus.RETAINED):
        raise ValueError(f"status must be superseded or retained, not {status!r}")
    return DB.conn.execute(
        "UPDATE supersession SET status = ?, resolved_by = ?, resolved_at = ? WHERE id = ?",
        (status, resolved_by, now_iso(), id),
    ).rowcount


def supersessions_for(ids: list[str]) -> list[SupersessionRow]:
    if not ids:
        return []
    rows = DB.conn.fetchall(
        f"""SELECT id, memory_id, supersedes, status, proposed_by,
                   resolved_by, resolved_at, created_at
              FROM supersession
             WHERE memory_id IN ({placeholders(ids)})
                OR supersedes IN ({placeholders(ids)})
             ORDER BY created_at, rowid""",
        [*ids, *ids],
    )
    return [
        SupersessionRow(
            **dict(row)
            | {
                "status": SupersessionStatus(row["status"]),
                "proposed_by": ConflictSource(row["proposed_by"]),
            }
        )
        for row in rows
    ]
