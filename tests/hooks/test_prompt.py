import io
import json
import sys
import time
from pathlib import Path

import pytest

from memeng.database.utils import DB
from memeng.facts.common import insert_fact
from memeng.hooks.prompt import build_block, hook_output, main, skip_prompt
from memeng.injections.common import injected_ids, record_injection
from memeng.retrieve import RankedFact, SearchResult, search
from memeng.scoring import Breakdown
from memeng.supersession.common import insert_supersession


def ranked(fact, **overrides):
    return RankedFact(
        **{
            "fact": fact,
            "score": 0.9,
            "explain": None,
            "superseded_by": None,
            "supersedes": None,
            "supersession_status": None,
            **overrides,
        }
    )


def result_of(*memories, scored=None, bm25_ran=False, entity_ran=False, excluded=()):
    return SearchResult(
        memories=list(memories),
        conflicts=[],
        excluded_as_injected=list(excluded),
        scored=scored or {},
        bm25_ran=bm25_ran,
        entity_ran=entity_ran,
    )


def test_build_block_returns_none_for_an_empty_result():
    assert build_block(result_of()) is None


def test_build_block_renders_a_plain_fact(make_fact):
    fact = make_fact("vectors live in sqlite-vec", created_at="2026-09-03T09:00:00Z")

    block, ids = build_block(result_of(ranked(fact)))

    assert block == {
        "memeng": [
            {
                "fact": "vectors live in sqlite-vec",
                "rationale": "one file, no extra process",
                "category": "architecture",
                "decided_at": "2026-09-03",
            }
        ]
    }
    assert ids == [fact.id]


def test_build_block_nests_an_unresolved_older_fact_under_the_newer_one(superseding_pair):
    older, newer = superseding_pair

    block, ids = build_block(
        result_of(
            ranked(newer, supersedes=older.id, supersession_status="unresolved"),
            ranked(older, score=0.6),
        )
    )

    [item] = block["memeng"]
    assert item["fact"] == "vectors in qdrant"
    assert item["supersedes"] == {
        "fact": "vectors in sqlite-vec",
        "decided_at": "2026-09-01",
        "status": "unresolved",
    }
    assert ids == [newer.id, older.id]


def test_build_block_fetches_an_unresolved_older_fact_missing_from_the_result(superseding_pair):
    older, newer = superseding_pair

    block, ids = build_block(
        result_of(ranked(newer, supersedes=older.id, supersession_status="unresolved"))
    )

    [item] = block["memeng"]
    assert item["supersedes"]["fact"] == "vectors in sqlite-vec"
    assert ids == [newer.id, older.id]


def test_build_block_puts_a_fetched_replacement_in_place_of_a_superseded_fact(superseding_pair, make_fact):
    older, newer = superseding_pair
    other = make_fact("feature flags are read once at start", created_at="2026-09-05T09:00:00Z")

    block, ids = build_block(
        result_of(
            ranked(older, superseded_by=newer.id, supersession_status="superseded"),
            ranked(other, score=0.7),
        )
    )

    assert [item["fact"] for item in block["memeng"]] == [
        "vectors in qdrant",
        "feature flags are read once at start",
    ]
    assert block["memeng"][0]["supersedes"]["status"] == "superseded"
    assert ids == [newer.id, older.id, other.id]


def test_build_block_keeps_every_older_fact_of_one_replacement_out_of_the_top_level(
    superseding_pair, make_fact
):
    older, newer = superseding_pair
    also_older = make_fact("vectors in pgvector", created_at="2026-08-20T09:00:00Z")

    block, ids = build_block(
        result_of(
            ranked(older, superseded_by=newer.id, supersession_status="superseded"),
            ranked(also_older, superseded_by=newer.id, supersession_status="superseded"),
        )
    )

    assert [item["fact"] for item in block["memeng"]] == ["vectors in qdrant"]
    assert block["memeng"][0]["supersedes"]["fact"] == "vectors in sqlite-vec"
    assert ids == [newer.id, older.id, also_older.id]


@pytest.fixture
def recording_spawn():
    def spawn(argv, log_path):
        spawn.calls.append((argv, log_path))

    spawn.calls = []
    return spawn


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".memeng.toml").write_text('project_id = "memeng"\n')
    return root


def payload_for(repo, prompt="what did we decide about vector storage"):
    return {
        "session_id": "s1",
        "transcript_path": "/tmp/s1.jsonl",
        "cwd": str(repo),
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
    }


@pytest.mark.parametrize(
    "prompt, skipped",
    [
        ("run tests", True),
        ("/commit please now", True),
        ("why do we cache lookups", False),
    ],
)
def test_skip_prompt(prompt, skipped):
    assert skip_prompt(prompt) is skipped


def test_hook_output_prints_the_block_and_records_the_injected_ids_detached(
    repo, make_fact, recording_spawn, tmp_config
):
    fact = make_fact("vectors live in sqlite-vec", created_at="2026-09-03T09:00:00Z")
    seen = {}

    def search(query, scope):
        seen["query"], seen["scope"] = query, scope
        return result_of(ranked(fact), scored={fact.id: Breakdown(0.9, 0.0, 0.0, 0.9)})

    output = hook_output(payload_for(repo), search=search, spawn=recording_spawn)

    assert seen["query"] == "what did we decide about vector storage"
    assert (seen["scope"].user_id, seen["scope"].project_id, seen["scope"].run_id) == ("local", "memeng", "s1")
    envelope = json.loads(output)
    assert envelope["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert json.loads(envelope["hookSpecificOutput"]["additionalContext"]) == {
        "memeng": [
            {
                "fact": "vectors live in sqlite-vec",
                "rationale": "one file, no extra process",
                "category": "architecture",
                "decided_at": "2026-09-03",
            }
        ]
    }
    [(argv, log_path)] = recording_spawn.calls
    assert argv[:4] == [sys.executable, "-m", "memeng.hooks.prompt", "--record"]
    assert json.loads(Path(argv[4]).read_text())["injected_ids"] == [fact.id]
    assert log_path == tmp_config.db_path.parent / "prompt.log"


def test_hook_output_builds_the_scope_with_the_configured_user_id(repo, recording_spawn, bind_config):
    bind_config(user_id="alice")
    seen = {}

    def search(query, scope):
        seen["scope"] = scope
        return result_of()

    hook_output(payload_for(repo), search=search, spawn=recording_spawn)

    assert seen["scope"].user_id == "alice"


def test_hook_output_is_silent_when_nothing_is_relevant(repo, recording_spawn):
    output = hook_output(payload_for(repo), search=lambda *a, **k: result_of(), spawn=recording_spawn)

    assert output is None
    assert recording_spawn.calls == []


def test_hook_output_uses_the_real_search_and_nests_an_unresolved_pair(
    repo, superseding_pair, unit, embed_returning, recording_spawn
):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")

    output = hook_output(
        payload_for(repo),
        search=lambda query, scope: search(query, scope, embed=embed_returning(unit(1))),
        spawn=recording_spawn,
    )

    block = json.loads(json.loads(output)["hookSpecificOutput"]["additionalContext"])
    [item] = block["memeng"]
    assert item["fact"] == "vectors in qdrant"
    assert item["supersedes"]["status"] == "unresolved"


def stdin_for(repo, **overrides):
    return io.StringIO(json.dumps(payload_for(repo) | overrides))


def refusing_search(*args, **kwargs):
    raise AssertionError("search must not run")


def test_main_prints_the_hook_json_and_returns_0(repo, make_fact, recording_spawn, tmp_config, capsys):
    fact = make_fact("vectors live in sqlite-vec")

    code = main(
        [],
        stdin=stdin_for(repo),
        env={},
        search=lambda *a, **k: result_of(
            ranked(fact), scored={fact.id: Breakdown(0.9, 0.0, 0.0, 0.9)}
        ),
        spawn=recording_spawn,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert len(recording_spawn.calls) == 1


def test_main_prints_nothing_when_nothing_is_relevant(repo, recording_spawn, tmp_config, capsys):
    code = main(
        [], stdin=stdin_for(repo), env={}, search=lambda *a, **k: result_of(),
        spawn=recording_spawn, load_config=lambda: tmp_config,
    )

    assert code == 0
    assert capsys.readouterr().out == ""


def test_main_returns_0_silently_when_the_recursion_guard_is_set(repo, recording_spawn, tmp_config, capsys):
    code = main(
        [], stdin=stdin_for(repo), env={"MEMENG_GATE": "1"}, search=refusing_search,
        spawn=recording_spawn, load_config=lambda: tmp_config,
    )

    assert code == 0
    assert capsys.readouterr().out == ""


def test_main_skips_the_search_for_a_short_prompt_and_a_slash_command(repo, recording_spawn, tmp_config, capsys):
    for prompt in ("run tests", "/model"):
        code = main(
            [], stdin=stdin_for(repo, prompt=prompt), env={}, search=refusing_search,
            spawn=recording_spawn, load_config=lambda: tmp_config,
        )
        assert code == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_and_logs_when_search_raises(repo, recording_spawn, tmp_config, capsys):
    def broken_search(*args, **kwargs):
        raise RuntimeError("embedder exploded")

    code = main(
        [], stdin=stdin_for(repo), env={}, search=broken_search,
        spawn=recording_spawn, load_config=lambda: tmp_config,
    )

    assert code == 0
    assert capsys.readouterr().out == ""
    assert "embedder exploded" in (tmp_config.db_path.parent / "prompt.log").read_text()


def test_main_in_record_mode_writes_the_injection_rows(
    superseding_pair, tmp_config, tmp_path, row_count
):
    older, newer = superseding_pair
    (tmp_path / "pending").mkdir()

    for retrieval_id in ("r2", "r3"):
        payload_path = tmp_path / "pending" / f"{retrieval_id}.json"
        payload_path.write_text(
            json.dumps(
                {
                    "retrieval": {
                        "id": retrieval_id,
                        "run_id": "s1",
                        "user_id": "local",
                        "project_id": "memeng",
                        "prompt": "how do we cap retries?",
                        "k": 10,
                        "bm25_ran": False,
                        "entity_ran": False,
                    },
                    "hits": [],
                    "injected_ids": [older.id, newer.id],
                }
            )
        )

        code = main(["--record", str(payload_path)], env={}, load_config=lambda: tmp_config)

        assert code == 0
        assert not payload_path.exists()
    assert injected_ids("s1") == {older.id, newer.id}
    assert row_count("injections") == 2


def test_main_wall_time_with_a_fake_search(repo, make_fact, recording_spawn, tmp_config, capsys):
    fact = make_fact("vectors live in sqlite-vec")
    started = time.perf_counter()

    main(
        [], stdin=stdin_for(repo), env={}, search=lambda *a, **k: result_of(ranked(fact)),
        spawn=recording_spawn, load_config=lambda: tmp_config,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    capsys.readouterr()
    print(f"prompt hook main() with fake search: {elapsed_ms:.1f} ms")


def test_hook_output_writes_a_payload_the_child_can_read(
    repo, make_fact, unit, recording_spawn, tmp_config
):
    fact = make_fact("the retry budget is three attempts")
    insert_fact(fact, unit(0), [], [])
    scored = {fact.id: Breakdown(0.9, 0.4, 0.0, 0.86)}

    hook_output(
        payload_for(repo),
        search=lambda query, scope: result_of(
            ranked(fact), scored=scored, bm25_ran=True
        ),
        spawn=recording_spawn,
    )

    [(argv, _)] = recording_spawn.calls
    assert argv[:4] == [sys.executable, "-m", "memeng.hooks.prompt", "--record"]
    written = json.loads(Path(argv[4]).read_text())
    assert written["retrieval"]["prompt"] == "what did we decide about vector storage"
    assert written["retrieval"]["run_id"] == "s1"
    assert written["retrieval"]["project_id"] == "memeng"
    assert written["retrieval"]["k"] == tmp_config.top_k
    assert written["retrieval"]["bm25_ran"] is True
    assert written["retrieval"]["entity_ran"] is False
    assert written["hits"] == [
        {
            "memory_id": fact.id,
            "rank": 0,
            "cosine": 0.9,
            "bm25": 0.4,
            "entity_boost": 0.0,
            "final_score": 0.86,
            "injected": True,
        }
    ]
    assert written["injected_ids"] == [fact.id]


def test_a_memory_withheld_as_already_injected_is_recorded_as_not_injected(
    repo, make_fact, unit, recording_spawn, tmp_config
):
    sent = make_fact("the retry budget is three attempts")
    withheld = make_fact("deploys are blue-green")
    insert_fact(sent, unit(0), [], [])
    insert_fact(withheld, unit(1), [], [])
    scored = {
        withheld.id: Breakdown(0.9, 0.0, 0.0, 0.9),
        sent.id: Breakdown(0.7, 0.0, 0.0, 0.7),
    }

    hook_output(
        payload_for(repo),
        search=lambda query, scope: result_of(
            ranked(sent), scored=scored, excluded=[withheld.id]
        ),
        spawn=recording_spawn,
    )

    [(argv, _)] = recording_spawn.calls
    written = json.loads(Path(argv[4]).read_text())
    assert {hit["memory_id"]: hit["injected"] for hit in written["hits"]} == {
        sent.id: True,
        withheld.id: False,
    }
    assert {hit["memory_id"]: hit["rank"] for hit in written["hits"]} == {
        withheld.id: 0,
        sent.id: 1,
    }
    assert written["injected_ids"] == [sent.id]


def test_a_prompt_where_every_ranked_memory_is_withheld_still_writes_a_retrieval_row(
    repo, make_fact, unit, embed_returning, recording_spawn, tmp_config
):
    fact = make_fact("the retry budget is three attempts")
    insert_fact(fact, unit(0), [], [])
    record_injection("s1", [fact.id])

    output = hook_output(
        payload_for(repo),
        search=lambda query, scope: search(query, scope, embed=embed_returning(unit(0))),
        spawn=recording_spawn,
    )

    assert output is None
    [(argv, _)] = recording_spawn.calls
    code = main(["--record", argv[4]], env={}, load_config=lambda: tmp_config)

    assert code == 0
    assert (
        DB.conn.fetchone("SELECT run_id FROM logs_retrieval WHERE run_id = 's1'")
        is not None
    )
    hit = DB.conn.fetchone(
        "SELECT injected FROM logs_retrieval_hits WHERE memory_id = ?", (fact.id,)
    )
    assert hit["injected"] == 0


def test_record_writes_the_injection_and_the_retrieval_then_removes_the_file(
    make_fact, unit, tmp_config, tmp_path
):
    fact = make_fact("the retry budget is three attempts")
    insert_fact(fact, unit(0), [], [])
    payload_path = tmp_path / "pending" / "r1.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "id": "r1",
                    "run_id": "s1",
                    "user_id": "local",
                    "project_id": "memeng",
                    "prompt": "how do we cap retries?",
                    "k": 10,
                    "bm25_ran": True,
                    "entity_ran": False,
                },
                "hits": [
                    {
                        "memory_id": fact.id,
                        "rank": 0,
                        "cosine": 0.9,
                        "bm25": 0.4,
                        "entity_boost": 0.0,
                        "final_score": 0.86,
                        "injected": True,
                    }
                ],
                "injected_ids": [fact.id],
            }
        )
    )

    code = main(["--record", str(payload_path)], env={}, load_config=lambda: tmp_config)

    assert code == 0
    assert injected_ids("s1") == {fact.id}
    retrieval = DB.conn.fetchone(
        "SELECT prompt, k, bm25_ran, entity_ran FROM logs_retrieval WHERE id = 'r1'"
    )
    assert tuple(retrieval) == ("how do we cap retries?", 10, 1, 0)
    hits = DB.conn.fetchall(
        """SELECT memory_id, rank, cosine, bm25, entity_boost, final_score, injected
             FROM logs_retrieval_hits WHERE retrieval_id = 'r1'"""
    )
    assert [tuple(hit) for hit in hits] == [(fact.id, 0, 0.9, 0.4, 0.0, 0.86, 1)]
    assert not payload_path.exists()


def test_record_decisions_false_writes_the_injection_but_skips_the_retrieval_log(
    make_fact, unit, bind_config, row_count, tmp_path
):
    fact = make_fact("the retry budget is three attempts")
    insert_fact(fact, unit(0), [], [])
    config = bind_config(record_decisions=False)
    payload_path = tmp_path / "pending" / "r3.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "id": "r3",
                    "run_id": "s1",
                    "user_id": "local",
                    "project_id": "memeng",
                    "prompt": "how do we cap retries?",
                    "k": 10,
                    "bm25_ran": True,
                    "entity_ran": False,
                },
                "hits": [
                    {
                        "memory_id": fact.id,
                        "rank": 0,
                        "cosine": 0.9,
                        "bm25": 0.4,
                        "entity_boost": 0.0,
                        "final_score": 0.86,
                        "injected": True,
                    }
                ],
                "injected_ids": [fact.id],
            }
        )
    )

    code = main(["--record", str(payload_path)], env={}, load_config=lambda: config)

    assert code == 0
    assert injected_ids("s1") == {fact.id}
    assert row_count("logs_retrieval") == 0
    assert row_count("logs_retrieval_hits") == 0
    assert not payload_path.exists()


def test_main_injects_nothing_where_there_is_no_alias(
    tmp_path, recording_spawn, tmp_config, capsys
):
    cwd = tmp_path / "feature-flags"
    cwd.mkdir()

    code = main(
        [], stdin=stdin_for(cwd), env={}, search=refusing_search,
        spawn=recording_spawn, load_config=lambda: tmp_config,
    )

    assert code == 0
    assert capsys.readouterr().out == ""
    assert recording_spawn.calls == []
    assert not (tmp_config.db_path.parent / "prompt.log").exists()
