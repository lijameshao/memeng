import json
import logging
import re
import string

from memeng.config import CONFIG
from memeng.enums.common import Category
from memeng.llm import LLMMalformed
from memeng.models.facts import ExtractedFact, StoredFact

logger = logging.getLogger(__name__)


def _supersedes_id(raw: object, neighbour_count: int) -> str | None:
    if not isinstance(raw, (int, str)):
        return None

    index = str(raw).strip()
    if not index.isdigit():
        return None
    if int(index) >= neighbour_count:
        logger.warning(
            "dropping supersedes %s: only %d neighbours were shown",
            index,
            neighbour_count,
        )
        return None
    return index


def render_gate_prompt(window_text: str) -> str:
    return CONFIG.gate_prompt_path.read_text().replace("{window}", window_text)


def render_extract_prompt(window_text: str, neighbours: list[StoredFact]) -> str:
    shown = "\n".join(
        f"{index}: {neighbour.fact} — {neighbour.rationale}"
        for index, neighbour in enumerate(neighbours)
    )
    return (
        CONFIG.extract_prompt_path.read_text()
        .replace("{categories}", ", ".join(Category))
        .replace("{neighbours}", shown or "(none)")
        .replace("{window}", window_text)
    )


def parse_gate(text: str) -> bool:
    words = text.strip().lower().split()
    answer = words[0].strip(string.punctuation) if words else ""
    if answer == "yes":
        return True
    if answer == "no":
        return False
    raise LLMMalformed(f"gate answered neither yes nor no: {text!r}")


def parse_extraction(text: str, neighbour_count: int) -> list[ExtractedFact]:
    unfenced = re.sub(r"\A```[a-z]*\s*|\s*```\Z", "", text.strip())
    try:
        payload = json.loads(unfenced)
    except json.JSONDecodeError as error:
        raise LLMMalformed(f"extractor did not return JSON: {text!r}") from error
    if not isinstance(payload, list):
        raise LLMMalformed(f"extractor returned {type(payload).__name__}, not a list")

    extracted = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        fact = item.get("fact")
        rationale = item.get("rationale")
        if not isinstance(fact, str) or not fact.strip():
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        try:
            category = Category(item.get("category"))
        except ValueError:
            continue
        entities = item.get("entities")
        if not isinstance(entities, list):
            entities = []
        extracted.append(
            ExtractedFact(
                fact=fact.strip(),
                rationale=rationale.strip(),
                category=category,
                # TODO(#5): replace with text.normalise_entity
                entities=[
                    " ".join(entity.lower().split())
                    for entity in entities
                    if isinstance(entity, str)
                ],
                supersedes=_supersedes_id(item.get("supersedes"), neighbour_count),
            )
        )
    return extracted
