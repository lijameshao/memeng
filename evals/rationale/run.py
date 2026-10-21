import argparse
import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from memeng import capture, llm
from memeng.config import CONFIG, Config
from memeng.database.utils import DB
from memeng.facts.common import get_facts
from memeng.hooks.stop import read_new_turns
from memeng.llm import LLMMalformed
from memeng.models.common import Scope
from memeng.models.facts import StoredFact
from memeng.utils.common import now_iso

JUDGE_PATH = Path(__file__).with_name("judge.md")
JUDGE_MODEL = "sonnet"


@dataclass(frozen=True)
class FactWithSources:
    fact: StoredFact
    message_ids: list[str]
    turns: str


def facts_with_sources(scope: Scope) -> list[FactWithSources]:
    items = []
    for fact in get_facts(scope.project_id, scope.user_id):
        rows = DB.conn.fetchall(
            """SELECT m.id, m.role, m.content
                 FROM fact_sources fs JOIN messages m ON m.id = fs.message_id
                WHERE fs.memory_id = ?
                ORDER BY m.created_at, m.rowid""",
            (fact.id,),
        )
        items.append(
            FactWithSources(
                fact=fact,
                message_ids=[row["id"] for row in rows],
                turns="\n\n".join(f"{row['role']}: {row['content']}" for row in rows),
            )
        )
    return items


def render_judge(rationale: str, turns: str) -> str:
    return (
        JUDGE_PATH.read_text()
        .replace("{rationale}", rationale)
        .replace("{turns}", turns)
    )


def parse_verdict(text: str) -> tuple[bool, str]:
    unfenced = re.sub(r"\A```[a-z]*\s*|\s*```\Z", "", text.strip())
    try:
        payload = json.loads(unfenced)
    except json.JSONDecodeError as error:
        raise LLMMalformed(f"judge did not return JSON: {text!r}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("faithful"), bool):
        raise LLMMalformed(f"judge returned no boolean verdict: {text!r}")
    return payload["faithful"], str(payload.get("why", ""))


def judge(scope: Scope, *, run_prompt=llm.run_prompt) -> dict:
    stored, malformed = [], []
    for item in facts_with_sources(scope):
        try:
            verdict, why = parse_verdict(
                run_prompt(render_judge(item.fact.rationale, item.turns), JUDGE_MODEL)
            )
        except LLMMalformed:
            verdict, why = None, ""
            malformed.append(item.fact.id)
        stored.append(
            {
                "fact": item.fact.fact,
                "rationale": item.fact.rationale,
                "message_ids": item.message_ids,
                "faithful": verdict,
                "why": why,
            }
        )
    return {
        "facts": len(stored),
        "faithful": sum(entry["faithful"] is True for entry in stored),
        "stored": stored,
        "unfaithful": [
            {
                "fact": entry["fact"],
                "rationale": entry["rationale"],
                "why": entry["why"],
                "message_ids": entry["message_ids"],
            }
            for entry in stored
            if entry["faithful"] is False
        ],
        "malformed": malformed,
    }


def run_transcript(path: Path, base: Config, *, add=capture.add, run_prompt=llm.run_prompt) -> dict:
    """`base` is the eval's own config: one store per transcript is derived
    from it, the way a sweep derives one config per value.
    """
    DB.open(replace(base, db_path=base.db_path.parent / f"{path.stem}.db"))
    scope = Scope(user_id="local", project_id=path.stem, run_id=path.stem)
    add(read_new_turns(path, None, scope), scope)
    return {"transcript": path.name} | judge(scope, run_prompt=run_prompt)


def main(
    argv: list[str] | None = None,
    *,
    add=capture.add,
    run_prompt=llm.run_prompt,
    model_id=llm.model_id,
    load_config=Config.load,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, default=Path(__file__).with_name("transcripts"))
    parser.add_argument("--expectations", type=Path, default=Path(__file__).with_name("expectations.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)

    expected = {
        item["transcript"]: item["expect_stored"]
        for item in yaml.safe_load(args.expectations.read_text())
    }
    with tempfile.TemporaryDirectory() as stores:
        base = replace(load_config(), db_path=Path(stores) / "store.db")
        reports = [
            run_transcript(path, base, add=add, run_prompt=run_prompt)
            for path in sorted(args.transcripts.glob("*.jsonl"))
        ]

    violations = [
        {"transcript": r["transcript"], "expect_stored": expected[r["transcript"]], "facts": r["facts"]}
        for r in reports
        if r["transcript"] in expected and expected[r["transcript"]] != (r["facts"] > 0)
    ]
    models = {"gate": CONFIG.gate_model, "extract": CONFIG.extract_model, "judge": JUDGE_MODEL}
    result = {
        "embed_model": CONFIG.embed_model,
        "models": {
            call: {"model": name, "model_id": model_id(name)} for call, name in models.items()
        },
        "transcripts": reports,
        "totals": {
            "facts": sum(r["facts"] for r in reports),
            "faithful": sum(r["faithful"] for r in reports),
            "unfaithful": sum(len(r["unfaithful"]) for r in reports),
        },
        "violations": violations,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{now_iso().replace(':', '')}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for r in reports:
        print(f"{r['transcript']}: {r['facts']} facts, {r['faithful']} faithful, {len(r['unfaithful'])} unfaithful")
    for v in violations:
        print(f"VIOLATION {v['transcript']}: expect_stored={v['expect_stored']} but stored {v['facts']}")
    print(f"wrote {out}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
