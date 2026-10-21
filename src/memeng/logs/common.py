import uuid

from memeng.database.utils import DB
from memeng.models.logs import CaptureLog, RetrievalHit, RetrievalLog, Verdict


def insert_capture_log(log: CaptureLog) -> str:
    DB.conn.execute(
        """INSERT INTO logs_captures(
             id, run_id, user_id, project_id, window_text, truncated,
             message_count, gate_answer, outcome, stored_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            log.id,
            log.run_id,
            log.user_id,
            log.project_id,
            log.window_text,
            int(log.truncated),
            log.message_count,
            log.gate_answer,
            log.outcome,
            log.stored_count,
            log.created_at,
        ),
    )
    return log.id


def insert_retrieval_log(log: RetrievalLog, hits: list[RetrievalHit]) -> str:
    with DB.conn.transaction() as db:
        db.execute(
            """INSERT INTO logs_retrieval(
                 id, run_id, user_id, project_id, prompt, k,
                 bm25_ran, entity_ran, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log.id,
                log.run_id,
                log.user_id,
                log.project_id,
                log.prompt,
                log.k,
                int(log.bm25_ran),
                int(log.entity_ran),
                log.created_at,
            ),
        )
        db.executemany(
            """INSERT INTO logs_retrieval_hits(
                 id, retrieval_id, memory_id, rank, cosine, bm25,
                 entity_boost, final_score, injected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    log.id,
                    hit.memory_id,
                    hit.rank,
                    hit.cosine,
                    hit.bm25,
                    hit.entity_boost,
                    hit.final_score,
                    int(hit.injected),
                )
                for hit in hits
            ],
        )
    return log.id


def insert_verdict(verdict: Verdict) -> None:
    DB.conn.execute(
        """INSERT INTO logs_verdicts(
             target, target_id, verdict, judged_by, judged_at, note)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(target, target_id, judged_by) DO UPDATE SET
             verdict = excluded.verdict,
             judged_at = excluded.judged_at,
             note = excluded.note""",
        (
            verdict.target,
            verdict.target_id,
            verdict.verdict,
            verdict.judged_by,
            verdict.judged_at,
            verdict.note,
        ),
    )
