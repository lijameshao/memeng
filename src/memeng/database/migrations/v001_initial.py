STATEMENTS = (
    """CREATE TABLE memories(
         rowid            INTEGER PRIMARY KEY,
         id               TEXT NOT NULL UNIQUE,
         fact             TEXT NOT NULL,
         rationale        TEXT NOT NULL,
         category         TEXT NOT NULL,
         source           TEXT,
         text_lemmatized  TEXT NOT NULL,
         hash             TEXT NOT NULL,
         user_id          TEXT NOT NULL,
         project_id       TEXT NOT NULL,
         agent_id         TEXT,
         run_id           TEXT,
         capture_id       TEXT,
         created_at       TEXT NOT NULL
       )""",
    "CREATE VIRTUAL TABLE memories_fts USING fts5(text_lemmatized, tokenize='unicode61')",
    "CREATE TABLE entities(entity_norm TEXT, memory_id TEXT REFERENCES memories(id))",
    "CREATE INDEX entities_by_norm ON entities(entity_norm)",
    """CREATE TABLE supersession(
         id           TEXT PRIMARY KEY,
         memory_id    TEXT NOT NULL REFERENCES memories(id),
         supersedes   TEXT NOT NULL REFERENCES memories(id),
         status       TEXT NOT NULL,
         proposed_by  TEXT NOT NULL,
         resolved_by  TEXT,
         resolved_at  TEXT,
         created_at   TEXT NOT NULL,
         UNIQUE (memory_id, supersedes)
       )""",
    """CREATE TABLE fact_sources(
         memory_id  TEXT REFERENCES memories(id),
         message_id TEXT REFERENCES messages(id),
         PRIMARY KEY (memory_id, message_id)
       )""",
    "CREATE INDEX fact_sources_by_message ON fact_sources(message_id)",
    """CREATE TABLE injections(
         run_id     TEXT NOT NULL,
         memory_id  TEXT NOT NULL REFERENCES memories(id),
         created_at TEXT NOT NULL,
         PRIMARY KEY (run_id, memory_id)
       )""",
    """CREATE TABLE messages(
         id           TEXT PRIMARY KEY,
         role         TEXT NOT NULL,
         content      TEXT NOT NULL,
         created_at   TEXT NOT NULL,
         processed_at TEXT,
         user_id      TEXT NOT NULL,
         project_id   TEXT NOT NULL,
         agent_id     TEXT,
         run_id       TEXT,
         gate_count   INTEGER NOT NULL DEFAULT 0
       )""",
    """CREATE TABLE logs_captures(
         id            TEXT PRIMARY KEY,
         run_id        TEXT,
         user_id       TEXT NOT NULL,
         project_id    TEXT NOT NULL,
         window_text   TEXT NOT NULL,
         truncated     INTEGER NOT NULL,
         message_count INTEGER NOT NULL,
         gate_answer   TEXT,
         outcome       TEXT NOT NULL,
         stored_count  INTEGER NOT NULL,
         created_at    TEXT NOT NULL
       )""",
    """CREATE TABLE logs_retrieval(
         id         TEXT PRIMARY KEY,
         run_id     TEXT NOT NULL,
         user_id    TEXT NOT NULL,
         project_id TEXT NOT NULL,
         prompt     TEXT NOT NULL,
         k          INTEGER NOT NULL,
         bm25_ran   INTEGER NOT NULL,
         entity_ran INTEGER NOT NULL,
         created_at TEXT NOT NULL
       )""",
    """CREATE TABLE logs_retrieval_hits(
         id           TEXT PRIMARY KEY,
         retrieval_id TEXT NOT NULL REFERENCES logs_retrieval(id) ON DELETE CASCADE,
         memory_id    TEXT NOT NULL,
         rank         INTEGER NOT NULL,
         cosine       REAL NOT NULL,
         bm25         REAL NOT NULL,
         entity_boost REAL NOT NULL,
         final_score  REAL NOT NULL,
         injected     INTEGER NOT NULL,
         UNIQUE (retrieval_id, memory_id)
       )""",
    "CREATE INDEX logs_retrieval_hits_by_memory ON logs_retrieval_hits(memory_id)",
    """CREATE TABLE logs_verdicts(
         target    TEXT NOT NULL,
         target_id TEXT NOT NULL,
         verdict   TEXT NOT NULL,
         judged_by TEXT NOT NULL,
         judged_at TEXT NOT NULL,
         note      TEXT,
         PRIMARY KEY (target, target_id, judged_by)
       )""",
)
