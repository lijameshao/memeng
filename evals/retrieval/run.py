import argparse
import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import yaml

from memeng import embed as embed_module
from memeng.config import CONFIG, Config
from memeng.database.utils import DB
from memeng.enums.common import Category
from memeng.facts.common import insert_fact, vectors_for
from memeng.models.common import Scope
from memeng.models.facts import StoredFact
from memeng.retrieve import search
from memeng.scoring import Breakdown
from memeng.text import lemmatise, normalise_entity
from memeng.utils.common import now_iso


@dataclass(frozen=True)
class Seed:
    key: str
    fact: str
    rationale: str
    category: str
    entities: list[str]
    created_at: str


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    query: str
    distractors: int
    seeds: list[Seed]
    expected: list[str]


@dataclass(frozen=True)
class Outcome:
    case: Case
    returned: list[str]
    conflicts: list[tuple[str, str]]
    explain: dict[str, Breakdown] = field(default_factory=dict)
    expected_cosine: dict[str, float] = field(default_factory=dict)


def load_cases(path: Path) -> list[Case]:
    return [
        Case(
            id=item["id"],
            category=item["category"],
            query=item["query"],
            distractors=int(item.get("distractors", 0)),
            seeds=[Seed(**seed) for seed in item.get("seed_facts", [])],
            expected=list(item.get("expected", [])),
        )
        for item in yaml.safe_load(path.read_text())
    ]


def load_distractors(path: Path) -> list[Seed]:
    return [
        Seed(
            key=f"d{index}",
            fact=item["fact"],
            rationale=item["rationale"],
            category=item["category"],
            entities=list(item.get("entities", [])),
            created_at=f"2026-01-01T00:{index:02d}:00Z",
        )
        for index, item in enumerate(yaml.safe_load(path.read_text()))
    ]


class EmbedCache:
    def __init__(self, embed):
        self._embed = embed
        self._vectors: dict[str, list[float]] = {}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if missing:
            self._vectors.update(zip(missing, self._embed(missing)))
        return [self._vectors[text] for text in texts]


def seed_store(case: Case, pool: list[Seed], embed) -> dict[str, str]:
    seeds = [*case.seeds, *pool[: case.distractors]]
    texts = [f"{seed.fact} {seed.rationale}" for seed in seeds]
    ids = {}
    for seed, text, vector in zip(seeds, texts, embed(texts)):
        fact = StoredFact(
            fact=seed.fact,
            rationale=seed.rationale,
            category=Category(seed.category),
            text_lemmatized=" ".join(lemmatise(text)),
            hash=hashlib.md5(seed.fact.encode()).hexdigest(),
            user_id="local",
            project_id="eval",
            created_at=seed.created_at,
        )
        insert_fact(fact, vector, [normalise_entity(e) for e in seed.entities], [])
        ids[seed.key] = fact.id
    return ids


def score(outcome: Outcome) -> dict[str, float | None]:
    expected, returned = outcome.case.expected, outcome.returned
    hits = [key for key in returned if key in expected]
    if outcome.case.category == "negative":
        return {
            "recall": None,
            "precision": None,
            "silence": 1.0 if not returned else 0.0,
            "temporal": None,
        }
    result = {
        "recall": len(hits) / len(expected) if expected else None,
        "precision": (len(hits) / len(returned)) if returned else 0.0,
        "silence": None,
        "temporal": None,
    }
    if outcome.case.category == "temporal":
        newer, older = expected[0], expected[1]
        ordered = (
            newer in returned
            and older in returned
            and returned.index(newer) < returned.index(older)
        )
        paired = (newer, older) in outcome.conflicts
        result["temporal"] = 1.0 if ordered and paired else 0.0
    return result


def run_case(case: Case, pool: list[Seed], base: Config, embed) -> Outcome:
    """`base` is the eval's own config: one store per case is derived from it,
    and a sweep rebinds it per value.
    """
    DB.open(replace(base, db_path=base.db_path.parent / f"{case.id}.db"))
    ids = seed_store(case, pool, embed)
    keys = {memory_id: key for key, memory_id in ids.items()}
    scope = Scope(user_id="local", project_id="eval")

    result = search(case.query, scope, k=10, explain=True, embed=embed)
    returned = [keys[m.fact.id] for m in result.memories]
    explain = {keys[m.fact.id]: m.explain for m in result.memories}
    conflicts = [(keys[c.newer_id], keys[c.older_id]) for c in result.conflicts]

    [query_vector] = embed([case.query])
    q = np.array(query_vector, dtype=float)
    q /= np.linalg.norm(q) or 1.0
    expected_ids = [ids[key] for key in case.expected if key in ids]
    cosines = {}
    for memory_id, vector in vectors_for(expected_ids).items():
        v = np.array(vector, dtype=float)
        cosines[keys[memory_id]] = float(v @ q / (np.linalg.norm(v) or 1.0))
    return Outcome(case, returned, conflicts, explain, cosines)


def summarise(outcomes: list[Outcome]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {"all": []}
    for outcome in outcomes:
        scored = score(outcome)
        groups.setdefault(outcome.case.category, []).append(scored)
        groups["all"].append(scored)
    rows = {}
    for category, scores in groups.items():
        row = {}
        for metric in ("recall", "precision", "silence", "temporal"):
            values = [s[metric] for s in scores if s[metric] is not None]
            row[metric] = statistics.mean(values) if values else None
        row["n"] = len(scores)
        rows[category] = row
    return rows


def table(rows: dict[str, dict[str, dict]]) -> str:
    def cell(value):
        return "-" if value is None else f"{value:.2f}"

    lines = [
        "| value | category | recall@10 | precision@10 | silence | temporal | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for value, by_category in rows.items():
        for category, row in by_category.items():
            lines.append(
                f"| {value} | {category} | {cell(row['recall'])} | {cell(row['precision'])} |"
                f" {cell(row['silence'])} | {cell(row['temporal'])} | {row['n']} |"
            )
    return "\n".join(lines)


def explain_misses(outcomes: list[Outcome]) -> list[str]:
    lines = []
    for outcome in outcomes:
        lowest = min((b.final_score for b in outcome.explain.values()), default=None)
        for key in outcome.case.expected:
            if key in outcome.returned:
                continue
            fact = next((s.fact for s in outcome.case.seeds if s.key == key), key)
            lines.append(
                f"{outcome.case.id}: missed {key!r} ({fact[:60]}): cosine to query"
                f" {outcome.expected_cosine.get(key, float('nan')):.3f},"
                f" semantic_threshold {CONFIG.semantic_threshold}, lowest returned final_score"
                f" {'-' if lowest is None else f'{lowest:.3f}'}"
            )
    return lines


def parse_sweep(text: str) -> tuple[str, list[float]]:
    key, _, values = text.partition("=")
    return key, [float(v) for v in values.split(",") if v]


def _with_value(base: Config, key: str, value: float) -> Config:
    kind = {f.name: f.type for f in fields(Config)}[key]
    return replace(base, **{key: int(value) if kind is int else value})


def main(argv: list[str] | None = None, *, embed=embed_module.embed, load_config=Config.load) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("fixture.yaml"))
    parser.add_argument(
        "--distractors", type=Path, default=Path(__file__).with_name("distractors.yaml")
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--sweep", help="key=v1,v2 for one Config field")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    cases, pool = load_cases(args.fixture), load_distractors(args.distractors)
    cache = EmbedCache(embed)
    key, values = parse_sweep(args.sweep) if args.sweep else (None, [None])

    rows, case_rows, misses = {}, [], []
    with tempfile.TemporaryDirectory() as stores:
        loaded = load_config()
        for value in values:
            label = "default" if value is None else str(value)
            base = replace(loaded, db_path=Path(stores) / label / "store.db")
            if value is not None:
                base = _with_value(base, key, value)
            outcomes = [run_case(case, pool, base, cache) for case in cases]
            rows[label] = summarise(outcomes)
            case_rows.extend(
                {
                    "id": o.case.id,
                    "value": label,
                    "returned": o.returned,
                    "conflicts": o.conflicts,
                    "scores": score(o),
                }
                for o in outcomes
            )
            if args.explain:
                misses.extend(f"[{label}] {line}" for line in explain_misses(outcomes))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{now_iso().replace(':', '')}.json"
    result = {"sweep": key, "embed_model": CONFIG.embed_model, "rows": rows, "cases": case_rows}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(table(rows))
    for line in misses:
        print(line)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
