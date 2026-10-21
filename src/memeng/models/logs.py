import uuid
from dataclasses import dataclass, field

from memeng.enums.common import CaptureOutcome, VerdictTarget
from memeng.utils.common import now_iso


@dataclass(frozen=True)
class CaptureLog:
    user_id: str
    project_id: str
    window_text: str
    truncated: bool
    message_count: int
    outcome: CaptureOutcome
    gate_answer: str | None = None
    stored_count: int = 0
    run_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class RetrievalLog:
    user_id: str
    project_id: str
    run_id: str
    prompt: str
    k: int
    bm25_ran: bool
    entity_ran: bool
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class RetrievalHit:
    memory_id: str
    rank: int
    cosine: float
    bm25: float
    entity_boost: float
    final_score: float
    injected: bool


@dataclass(frozen=True)
class Verdict:
    target: VerdictTarget
    target_id: str
    verdict: str
    judged_by: str
    note: str | None = None
    judged_at: str = field(default_factory=now_iso)
