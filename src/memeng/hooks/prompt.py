import argparse
import json
import os
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from memeng.config import CONFIG, Config
from memeng.database.utils import DB
from memeng.enums.common import SupersessionStatus
from memeng.facts.common import facts_by_ids
from memeng.hooks.detach import spawn_detached
from memeng.injections.common import record_injection
from memeng.logs.common import insert_retrieval_log
from memeng.models.common import Scope
from memeng.models.facts import StoredFact
from memeng.models.logs import RetrievalHit, RetrievalLog
from memeng.project import resolve_project_id
from memeng.retrieve import SearchResult
from memeng.retrieve import search as search_facts


def build_block(result: SearchResult) -> tuple[dict, list[str]] | None:
    """Returns the block and the ids this session must not send again, which is
    every fact the block accounts for: the ones it renders and the older ones it
    withholds because their replacement is in the block.
    """
    if not result.memories:
        return None

    pairs: dict[str, list[tuple[str, SupersessionStatus]]] = {}
    for memory in result.memories:
        if memory.supersedes:
            pairs.setdefault(memory.fact.id, []).append(
                (memory.supersedes, SupersessionStatus.UNRESOLVED)
            )
        if memory.superseded_by:
            pairs.setdefault(memory.superseded_by, []).append(
                (memory.fact.id, SupersessionStatus.SUPERSEDED)
            )

    nested = {older for olders in pairs.values() for older, _ in olders}
    facts = {memory.fact.id: memory.fact for memory in result.memories}
    facts |= {fact.id: fact for fact in facts_by_ids(sorted((set(pairs) | nested) - set(facts)))}

    order = [memory.superseded_by or memory.fact.id for memory in result.memories]

    items, ids, seen = [], [], set()
    for memory_id in order:
        if memory_id in nested or memory_id in seen:
            continue
        seen.add(memory_id)
        fact = facts[memory_id]
        item = _item(fact)
        ids.append(fact.id)
        if memory_id in pairs:
            older_id, status = pairs[memory_id][0]
            older = facts[older_id]
            item["supersedes"] = {
                "fact": older.fact,
                "decided_at": older.created_at[:10],
                "status": status,
            }
            ids.extend(older_id for older_id, _ in pairs[memory_id])
        items.append(item)
    return {"memeng": items}, ids


def _item(fact: StoredFact) -> dict:
    return {
        "fact": fact.fact,
        "rationale": fact.rationale,
        "category": str(fact.category),
        "decided_at": fact.created_at[:10],
    }


def skip_prompt(prompt: str) -> bool:
    return len(prompt.split()) < 3 or prompt.lstrip().startswith("/")


def hook_output(
    payload: dict,
    *,
    search: Callable[[str, Scope], SearchResult] = search_facts,
    spawn: Callable[[list[str], Path], None] = spawn_detached,
) -> str | None:
    project = resolve_project_id(Path(payload["cwd"]))
    if project is None:
        return None
    scope = Scope(
        user_id=CONFIG.user_id,
        project_id=project.value,
        run_id=payload["session_id"],
    )
    result = search(payload["prompt"], scope)
    built = build_block(result)
    memory_ids = built[1] if built is not None else []
    if result.scored:
        spawn(
            [
                sys.executable,
                "-m",
                "memeng.hooks.prompt",
                "--record",
                str(_write_payload(payload["prompt"], scope, result, memory_ids)),
            ],
            CONFIG.db_path.expanduser().parent / "prompt.log",
        )
    if built is None:
        return None
    block, _ = built
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": json.dumps(block),
            }
        }
    )


def _write_payload(
    prompt: str, scope: Scope, result: SearchResult, memory_ids: list[str]
) -> Path:
    retrieval_id = str(uuid.uuid4())
    sent = set(memory_ids)
    order = list(result.scored)
    path = CONFIG.db_path.expanduser().parent / "pending" / f"{retrieval_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "id": retrieval_id,
                    "run_id": scope.run_id,
                    "user_id": scope.user_id,
                    "project_id": scope.project_id,
                    "prompt": prompt,
                    "k": CONFIG.top_k,
                    "bm25_ran": result.bm25_ran,
                    "entity_ran": result.entity_ran,
                },
                "hits": [
                    {
                        "memory_id": memory_id,
                        "rank": rank,
                        "cosine": result.scored[memory_id].semantic_score,
                        "bm25": result.scored[memory_id].bm25_score,
                        "entity_boost": result.scored[memory_id].entity_boost,
                        "final_score": result.scored[memory_id].final_score,
                        "injected": memory_id in sent,
                    }
                    for rank, memory_id in enumerate(order)
                ],
                "injected_ids": memory_ids,
            }
        )
    )
    return path


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    env: Mapping[str, str] | None = None,
    search: Callable[[str, Scope], SearchResult] = search_facts,
    spawn: Callable[[list[str], Path], None] = spawn_detached,
    load_config: Callable[[], Config] = Config.load,
) -> int:
    env = os.environ if env is None else env
    if env.get("MEMENG_GATE"):
        return 0

    args = _parser().parse_args(argv)
    config = None
    try:
        config = load_config()
        if args.record:
            DB.open(config)
            _record(Path(args.record))
            return 0

        payload = json.load(stdin or sys.stdin)
        if skip_prompt(payload.get("prompt", "")):
            return 0
        DB.open(config)
        output = hook_output(payload, search=search, spawn=spawn)
        if output is not None:
            print(output)
    except Exception:
        _log_failure(config)
    return 0


def _record(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text())
    retrieval = payload["retrieval"]
    record_injection(retrieval["run_id"], payload["injected_ids"])
    if CONFIG.record_decisions:
        insert_retrieval_log(
            RetrievalLog(**retrieval),
            [RetrievalHit(**hit) for hit in payload["hits"]],
        )
    payload_path.unlink()


def _log_failure(config: Config | None) -> None:
    if config is None:
        traceback.print_exc(file=sys.stderr)
        return
    log_path = config.db_path.expanduser().parent / "prompt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        traceback.print_exc(file=log)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memeng.hooks.prompt")
    parser.add_argument("--record", metavar="PAYLOAD_PATH")
    return parser


if __name__ == "__main__":
    sys.exit(main())
