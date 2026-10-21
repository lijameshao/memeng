import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectId:
    value: str
    path: Path


def resolve_project_id(repo_root: Path) -> ProjectId | None:
    alias_file = repo_root / ".memeng.toml"
    if not alias_file.exists():
        return None
    return ProjectId(tomllib.loads(alias_file.read_text())["project_id"], alias_file)
