import uuid
from dataclasses import dataclass, field

from memeng.utils.common import now_iso


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    user_id: str
    project_id: str
    agent_id: str | None = None
    run_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=now_iso)
    processed_at: str | None = None
    gate_count: int = 0
