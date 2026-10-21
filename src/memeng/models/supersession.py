from dataclasses import dataclass

from memeng.enums.common import ConflictSource, SupersessionStatus


@dataclass(frozen=True)
class SupersessionRow:
    id: str
    memory_id: str
    supersedes: str
    status: SupersessionStatus
    proposed_by: ConflictSource
    created_at: str
    resolved_by: str | None = None
    resolved_at: str | None = None
