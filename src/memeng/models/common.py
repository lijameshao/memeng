from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    user_id: str
    project_id: str
    agent_id: str | None = None
    run_id: str | None = None
