import uuid
from dataclasses import dataclass, field

from memeng.enums.common import Category
from memeng.utils.common import now_iso


@dataclass(frozen=True)
class StoredFact:
    fact: str
    rationale: str
    category: Category
    text_lemmatized: str
    hash: str
    user_id: str
    project_id: str
    source: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    capture_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class ExtractedFact:
    fact: str
    rationale: str
    category: Category
    entities: list[str]
    supersedes: str | None
