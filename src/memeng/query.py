from pathlib import Path

from memeng.enums.common import Category, SupersessionStatus
from memeng.facts.common import get_facts
from memeng.models.common import Scope
from memeng.models.facts import StoredFact
from memeng.supersession.common import set_supersession_status, supersessions_for


def browse(scope: Scope, category: str | None = None) -> list[StoredFact]:
    return get_facts(scope.project_id, scope.user_id, category)


def dump(scope: Scope, out_dir: Path) -> None:
    facts = browse(scope)
    if not facts:
        return

    fact_text = {fact.id: fact.fact for fact in facts}
    supersedes: dict[str, list[str]] = {}
    superseded_by: dict[str, list[str]] = {}
    for row in supersessions_for(list(fact_text)):
        newer = fact_text.get(row.memory_id, row.memory_id)
        older = fact_text.get(row.supersedes, row.supersedes)
        supersedes.setdefault(row.memory_id, []).append(
            f"- supersedes: {older} ({row.status})"
        )
        superseded_by.setdefault(row.supersedes, []).append(
            f"- superseded_by: {newer} ({row.status})"
        )

    sections: dict[Category, list[str]] = {}
    for fact in facts:
        lines = [
            f"## {fact.fact}",
            "",
            fact.rationale,
            "",
            f"- source: {fact.source or '-'}",
            f"- decided_at: {fact.created_at[:10]}",
            *supersedes.get(fact.id, []),
            *superseded_by.get(fact.id, []),
        ]
        sections.setdefault(fact.category, []).append("\n".join(lines) + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    for category, rendered in sections.items():
        (out_dir / f"{category}.md").write_text("\n".join(rendered), encoding="utf-8")


def resolve(
    supersession_id: str, status: SupersessionStatus, resolved_by: str
) -> None:
    if set_supersession_status(supersession_id, status, resolved_by) == 0:
        raise ValueError(f"unknown supersession id {supersession_id!r}")
