import struct

import sqlite_vec

from memeng.config import CONFIG
from memeng.database.utils import DB, placeholders
from memeng.enums.common import Category
from memeng.models.facts import StoredFact


def insert_fact(
    fact: StoredFact,
    vector: list[float],
    entities: list[str],
    source_message_ids: list[str],
) -> None:
    with DB.conn.transaction() as db:
        _ensure_vector_table(len(vector))
        rowid = db.execute(
            """INSERT INTO memories(
                 id, fact, rationale, category, source, text_lemmatized, hash,
                 user_id, project_id, agent_id, run_id, capture_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id,
                fact.fact,
                fact.rationale,
                fact.category,
                fact.source,
                fact.text_lemmatized,
                fact.hash,
                fact.user_id,
                fact.project_id,
                fact.agent_id,
                fact.run_id,
                fact.capture_id,
                fact.created_at,
            ),
        ).lastrowid
        db.execute(
            "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(vector)),
        )
        db.execute(
            "INSERT INTO memories_fts(rowid, text_lemmatized) VALUES (?, ?)",
            (rowid, fact.text_lemmatized),
        )
        db.executemany(
            "INSERT INTO entities(entity_norm, memory_id) VALUES (?, ?)",
            [(entity_norm, fact.id) for entity_norm in entities],
        )
        db.executemany(
            "INSERT INTO fact_sources(memory_id, message_id) VALUES (?, ?)",
            [(fact.id, message_id) for message_id in source_message_ids],
        )


def facts_by_ids(ids: list[str]) -> list[StoredFact]:
    if not ids:
        return []
    rows = DB.conn.fetchall(
        f"""SELECT id, fact, rationale, category, source, text_lemmatized, hash,
                   user_id, project_id, agent_id, run_id, capture_id, created_at
              FROM memories
             WHERE id IN ({placeholders(ids)})""",
        ids,
    )
    by_id = {
        row["id"]: StoredFact(**dict(row) | {"category": Category(row["category"])})
        for row in rows
    }
    return [by_id[memory_id] for memory_id in ids if memory_id in by_id]


def get_facts(
    project_id: str, user_id: str, category: str | None = None
) -> list[StoredFact]:
    sql = """SELECT id, fact, rationale, category, source, text_lemmatized, hash,
                    user_id, project_id, agent_id, run_id, capture_id, created_at
               FROM memories
              WHERE project_id = ? AND user_id = ?"""
    params = [project_id, user_id]
    if category is not None:
        sql += " AND category = ?"
        params.append(category)
    rows = DB.conn.fetchall(sql + " ORDER BY created_at, id", params)
    return [
        StoredFact(**dict(row) | {"category": Category(row["category"])})
        for row in rows
    ]


def get_all_facts() -> list[StoredFact]:
    rows = DB.conn.fetchall(
        """SELECT id, fact, rationale, category, source, text_lemmatized, hash,
                  user_id, project_id, agent_id, run_id, capture_id, created_at
             FROM memories
            ORDER BY created_at, id"""
    )
    return [
        StoredFact(**dict(row) | {"category": Category(row["category"])})
        for row in rows
    ]


def get_project_ids(user_id: str) -> list[str]:
    rows = DB.conn.fetchall(
        "SELECT DISTINCT project_id FROM memories WHERE user_id = ? ORDER BY project_id",
        (user_id,),
    )
    return [row["project_id"] for row in rows]


def replace_all_vectors(vectors: dict[str, list[float]]) -> None:
    """The vec0 column width is fixed at create time, so a changed dimension
    recreates the table instead of updating rows.
    """
    rows = DB.conn.fetchall("SELECT rowid, id FROM memories ORDER BY rowid")
    payload = [
        (row["rowid"], sqlite_vec.serialize_float32(vectors[row["id"]])) for row in rows
    ]
    dim = len(next(iter(vectors.values())))
    with DB.conn.transaction() as db:
        if db.meta_get("embed_dim") != str(dim):
            db.execute("DROP TABLE IF EXISTS memories_vec")
            db.execute(
                f"CREATE VIRTUAL TABLE memories_vec"
                f" USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )
            db.meta_set("embed_dim", str(dim))
        else:
            db.execute("DELETE FROM memories_vec")
        db.executemany(
            "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)", payload
        )
        db.meta_set("embed_model", CONFIG.embed_model)


def get_hashes(hashes: list[str], project_id: str, user_id: str) -> set[str]:
    if not hashes:
        return set()
    rows = DB.conn.fetchall(
        f"""SELECT hash
              FROM memories
             WHERE project_id = ? AND user_id = ?
               AND hash IN ({placeholders(hashes)})""",
        [project_id, user_id, *hashes],
    )
    return {row["hash"] for row in rows}


def vectors_for(ids: list[str]) -> dict[str, list[float]]:
    dim = DB.conn.meta_get("embed_dim")
    if dim is None or not ids:
        return {}
    rows = DB.conn.fetchall(
        f"""SELECT m.id, v.embedding
              FROM memories_vec v JOIN memories m ON m.rowid = v.rowid
             WHERE m.id IN ({placeholders(ids)})""",
        ids,
    )
    return {row["id"]: list(struct.unpack(f"{dim}f", row["embedding"])) for row in rows}


def knn(
    vector: list[float],
    project_id: str,
    user_id: str,
    n: int,
) -> list[tuple[str, float]]:
    if DB.conn.meta_get("embed_dim") is None:
        return []
    # sqlite-vec 0.1.9 accepts a rowid subquery alongside the match, so the scope
    # filter runs inside the index and k counts rows already in scope.
    rows = DB.conn.fetchall(
        """SELECT m.id, 1 - v.distance AS cosine
             FROM memories_vec v JOIN memories m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
              AND v.rowid IN (SELECT rowid
                                FROM memories
                               WHERE project_id = ? AND user_id = ?)
            ORDER BY v.distance""",
        (sqlite_vec.serialize_float32(vector), n, project_id, user_id),
    )
    return [(row["id"], row["cosine"]) for row in rows]


def bm25(terms: list[str], project_id: str, user_id: str) -> dict[str, float]:
    if not terms:
        return {}
    quoted = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    rows = DB.conn.fetchall(
        """SELECT m.id, -bm25(memories_fts) AS score
             FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? AND m.project_id = ? AND m.user_id = ?""",
        (quoted, project_id, user_id),
    )
    return {row["id"]: row["score"] for row in rows}


def get_entity_vocab(project_id: str, user_id: str) -> list[str]:
    rows = DB.conn.fetchall(
        """SELECT DISTINCT entity_norm
             FROM entities JOIN memories ON memories.id = entities.memory_id
            WHERE memories.project_id = ? AND memories.user_id = ?""",
        (project_id, user_id),
    )
    return [row["entity_norm"] for row in rows]


def get_entity_memory_ids(
    entity_norms: list[str], project_id: str, user_id: str
) -> dict[str, set[str]]:
    if not entity_norms:
        return {}
    rows = DB.conn.fetchall(
        f"""SELECT entity_norm, memory_id
              FROM entities JOIN memories ON memories.id = entities.memory_id
             WHERE memories.project_id = ? AND memories.user_id = ?
               AND entity_norm IN ({placeholders(entity_norms)})""",
        [project_id, user_id, *entity_norms],
    )
    linked: dict[str, set[str]] = {}
    for row in rows:
        linked.setdefault(row["entity_norm"], set()).add(row["memory_id"])
    return linked


def _ensure_vector_table(dim: int) -> None:
    if DB.conn.meta_get("embed_dim") is not None:
        return
    DB.conn.execute(
        f"CREATE VIRTUAL TABLE memories_vec"
        f" USING vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    DB.conn.meta_set("embed_dim", str(dim))
