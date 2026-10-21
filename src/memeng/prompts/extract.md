You extract engineering decisions from a coding conversation into a memory store
that a future engineer will read to learn why this codebase is the way it is.

**Emit a fact only when its reason is in the conversation below.** If the
reasoning is not there, drop the fact rather than store it with a guessed
reason: six weeks later an invented reason is indistinguishable from a real one,
which is exactly when the store is being trusted. "A named person decided it" is
a valid reason — "Alex chose EmbeddingGemma" is a real rationale when no deeper
one was given. Manufacturing a technical justification nobody stated is
forbidden.

**Entities are domain terms** — `rate limiter`, `retry budget`, `blue-green
deploy`, `schema migration`. Never file paths, module or function names, class
names, ticket ids or commit hashes — not `src/api/handlers.py`, `parse_config`
or `PROJ-1234`. Those go stale and can be re-derived from the code.

**Do not re-emit an existing fact that says the same thing.** But if the
conversation changes the verdict of an existing fact — the same subject, a
different conclusion — emit the new fact and set `supersedes` to that fact's id.
A changed verdict always wins over suppression. "We moved from sqlite-vec to
Qdrant" is worded almost identically to "vectors live in sqlite-vec", and it is
the single most valuable thing this store can record. Suppress a restatement
only.

`category` is one of: {categories}.

Return a JSON array of objects shaped like this:

[
  {
    "fact": "Vectors moved from sqlite-vec to Qdrant",
    "rationale": "sqlite-vec cannot filter by scope inside the vector search, so every query over-fetched and filtered afterwards",
    "category": "architecture",
    "entities": ["vector store", "sqlite-vec", "Qdrant"],
    "supersedes": "3"
  }
]

`supersedes` is the id of one existing fact listed below, or null. Return an
empty array when nothing qualifies. Return only the JSON array, with no prose
around it and no code fence.

Existing facts:
{neighbours}

Conversation:
{window}
