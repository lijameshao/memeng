import hashlib
import json
import logging

import pytest

from memeng.capture import add
from memeng.database.utils import DB
from memeng.embed import EmbedUnavailable
from memeng.enums.common import Category
from memeng.facts.common import facts_by_ids, get_entity_vocab, insert_fact
from memeng.llm import LLMTimeout
from memeng.messages.common import (
    get_unprocessed_messages,
    insert_messages,
)
from memeng.models.common import Scope
from memeng.supersession.common import supersessions_for

SCOPE = Scope("local", "memeng", run_id="run-1")
FACT = "we cache user lookups for five minutes"
RATIONALE = "the same query was hitting the database hundreds of times a second"
OTHER_FACT = "we turn new features on for staff accounts first"
WINDOW = f"user: why cache user lookups?\n\nuser: {RATIONALE}"


def extractor_json(*items):
    return json.dumps(
        [
            {
                "fact": FACT,
                "rationale": RATIONALE,
                "category": "architecture",
                "entities": ["user lookups"],
                "supersedes": None,
            }
            | item
            for item in items
        ]
    )


@pytest.fixture
def turns(message):
    return [message("why cache user lookups?"), message(RATIONALE)]


@pytest.fixture
def refuse_to_embed():
    def embed(texts):
        raise AssertionError("embedder called")

    return embed


def test_add_stores_one_fact_and_marks_the_window_processed(
    turns, fake_embed, scripted_run_prompt, row_count
):
    decided = "Cache user lookups for five minutes"
    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt(
            "yes",
            extractor_json({"fact": f"  {decided}\n", "rationale": f" {RATIONALE} "}),
        ),
        embed=fake_embed,
    )

    assert facts_by_ids([stored.id]) == [stored]
    assert (stored.fact, stored.rationale) == (decided, RATIONALE)
    assert stored.category == Category.ARCHITECTURE
    assert stored.hash == hashlib.md5(decided.encode()).hexdigest()
    assert stored.text_lemmatized == (
        "cache user lookup minute query hit database hundred time second"
    )
    assert (stored.user_id, stored.project_id, stored.agent_id, stored.run_id) == (
        "local",
        "memeng",
        None,
        "run-1",
    )
    assert stored.source == "session run-1"
    assert get_entity_vocab("memeng", "local") == ["user lookups"]
    assert supersessions_for([stored.id]) == []
    assert row_count("fact_sources") == 2
    assert row_count("memories_vec") == 1
    assert row_count("memories_fts") == 1
    assert DB.conn.fetchall("PRAGMA foreign_key_check") == []
    assert get_unprocessed_messages("memeng", "local") == []


def test_add_runs_the_gate_and_the_extractor_on_the_configured_models(
    tmp_path, bind_config, turns, fake_embed, scripted_run_prompt
):
    gate_prompt = tmp_path / "my-gate.md"
    gate_prompt.write_text("Decide.\n\n{window}\n")
    config = bind_config(gate_prompt_path=gate_prompt)
    run_prompt = scripted_run_prompt("yes", extractor_json({}))

    add(turns, SCOPE, run_prompt=run_prompt, embed=fake_embed)

    [(gate, gate_model), (extraction, extract_model)] = run_prompt.calls
    assert (gate_model, extract_model) == (config.gate_model, config.extract_model)
    assert gate == f"Decide.\n\n{WINDOW}\n"
    assert (
        "`category` is one of: business_logic, architecture, integration,"
        " data_model, process, constraint."
    ) in extraction
    assert '[\n  {\n    "fact": "Vectors moved from sqlite-vec to Qdrant",' in extraction
    assert extraction.endswith(f"Existing facts:\n(none)\n\nConversation:\n{WINDOW}\n")


def test_add_embeds_the_window_first_then_every_new_fact_in_one_batch(
    message, fake_embed, scripted_run_prompt
):
    batches = []

    def recording_embed(texts):
        batches.append(texts)
        return fake_embed(texts)

    answer = message(
        f"{RATIONALE}\nso five minutes is enough",
        role="assistant",
        agent_id="reviewer",
        run_id="run-1",
        created_at="2026-09-03T10:00:01Z",
    )
    question = message("why cache user lookups?", created_at="2026-09-03T10:00:00Z")

    add(
        [answer, question],
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({}, {"fact": OTHER_FACT})),
        embed=recording_embed,
    )

    assert len(batches) == 2
    assert batches[0] == [
        f"user: why cache user lookups?\n\nassistant: {RATIONALE}\nso five minutes is enough"
    ]
    assert batches[1] == [
        f"{FACT} {RATIONALE}",
        f"{OTHER_FACT} {RATIONALE}",
    ]
    rows = DB.conn.fetchall("SELECT id, agent_id, run_id FROM messages ORDER BY created_at")
    assert [tuple(row) for row in rows] == [
        (question.id, None, None),
        (answer.id, "reviewer", "run-1"),
    ]


@pytest.mark.parametrize("token_cap", [0, -1])
def test_add_raises_when_the_window_cap_leaves_no_room(
    turns, bind_config, scripted_run_prompt, refuse_to_embed, token_cap
):
    bind_config(window_token_cap=token_cap)

    with pytest.raises(ValueError):
        add(turns, SCOPE, run_prompt=scripted_run_prompt(), embed=refuse_to_embed)


def test_a_second_add_on_the_same_messages_stores_nothing_and_calls_no_model(
    turns, fake_embed, scripted_run_prompt, refuse_to_embed, row_count
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    run_prompt = scripted_run_prompt()
    again = add(turns, SCOPE, run_prompt=run_prompt, embed=refuse_to_embed)

    assert again == []
    assert run_prompt.calls == []
    assert row_count("memories") == 1


def test_add_of_no_messages_calls_no_model(scripted_run_prompt, refuse_to_embed):
    run_prompt = scripted_run_prompt()

    assert add([], SCOPE, run_prompt=run_prompt, embed=refuse_to_embed) == []
    assert run_prompt.calls == []


@pytest.mark.parametrize("gate_answer", ["no", "No, nothing was decided"])
def test_a_gate_no_leaves_the_window_unprocessed(
    turns, scripted_run_prompt, refuse_to_embed, row_count, gate_answer
):
    stored = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt(gate_answer),
        embed=refuse_to_embed,
    )

    assert stored == []
    released = get_unprocessed_messages("memeng", "local")
    assert [message.id for message in released] == [message.id for message in turns]
    assert [message.gate_count for message in released] == [1, 1]
    assert row_count("memories") == 0


@pytest.mark.parametrize(
    "extraction",
    [
        "[]",
        extractor_json({"fact": ""}),
        extractor_json({"fact": "  "}),
        extractor_json({"fact": None}),
        extractor_json({"fact": 7}),
        extractor_json({"rationale": ""}),
        extractor_json({"rationale": "  "}),
        extractor_json({"rationale": None}),
        extractor_json({"category": "opinion"}),
        extractor_json({"category": None}),
        extractor_json({"category": ["architecture"]}),
        extractor_json({"category": "ARCHITECTURE"}),
    ],
)
def test_an_extraction_with_no_facts_leaves_the_window_unprocessed(
    turns, fake_embed, scripted_run_prompt, row_count, extraction
):
    stored = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extraction),
        embed=fake_embed,
    )

    assert stored == []
    released = get_unprocessed_messages("memeng", "local")
    assert [message.id for message in released] == [message.id for message in turns]
    assert row_count("memories") == 0


@pytest.mark.parametrize(
    "supersedes, shown_index",
    [
        (0, 0),
        (2, 2),
        ("2", 2),
        (3, None),
        ("3", None),
        (-1, None),
        (2.0, None),
        ("the sqlite one", None),
        (["2"], None),
        (None, None),
    ],
)
def test_a_supersedes_index_writes_one_unresolved_extractor_proposal(
    turns, make_fact, unit, embed_returning, scripted_run_prompt, supersedes, shown_index
):
    neighbours = [
        make_fact("we cache user lookups for one minute"),
        make_fact("we cache user lookups for ten minutes"),
        make_fact("we cache user lookups for an hour"),
    ]
    for index in (2, 1, 0):
        insert_fact(neighbours[index], unit(index), [], [])
    run_prompt = scripted_run_prompt("yes", extractor_json({"supersedes": supersedes}))

    [stored] = add(
        turns,
        SCOPE,
        run_prompt=run_prompt,
        embed=embed_returning([0.8, 0.5, 0.3] + [0.0] * 5),
    )

    assert (
        "Existing facts:\n"
        + "".join(
            f"{index}: {neighbour.fact} — {neighbour.rationale}\n"
            for index, neighbour in enumerate(neighbours)
        )
        + f"\nConversation:\n{WINDOW}"
    ) in run_prompt.calls[1][0]
    proposals = [
        (row.memory_id, row.supersedes, row.status, row.proposed_by)
        for row in supersessions_for([stored.id])
    ]
    if shown_index is None:
        assert proposals == []
    else:
        assert proposals == [
            (stored.id, neighbours[shown_index].id, "unresolved", "extractor")
        ]


@pytest.mark.parametrize(
    "entities, expected",
    [
        (
            ["  Rate   Limiter ", 7, "CIRCUIT BREAKER"],
            ["circuit breaker", "rate limiter"],
        ),
        ([None, ["retry budget"], {"term": "retry budget"}], []),
        (None, []),
        ("rate limiter", []),
        ({"term": "rate limiter"}, []),
    ],
)
def test_add_links_only_the_entities_that_are_strings(
    turns, fake_embed, scripted_run_prompt, entities, expected
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({"entities": entities})),
        embed=fake_embed,
    )

    assert sorted(get_entity_vocab("memeng", "local")) == expected


def test_add_reads_an_extraction_inside_a_json_fence(
    turns, fake_embed, scripted_run_prompt
):
    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", f"```json\n{extractor_json({})}\n```"),
        embed=fake_embed,
    )

    assert stored.fact == FACT


def test_add_stores_the_sound_entry_beside_the_malformed_ones(
    turns, fake_embed, scripted_run_prompt
):
    extraction = json.dumps(
        [
            OTHER_FACT,
            None,
            [{"fact": OTHER_FACT, "rationale": RATIONALE}],
            {"fact": 7, "rationale": RATIONALE, "category": "architecture"},
            {"fact": OTHER_FACT, "rationale": None, "category": "architecture"},
            {"fact": OTHER_FACT, "rationale": "", "category": "architecture"},
            {"fact": FACT, "rationale": RATIONALE, "category": "architecture"},
        ]
    )

    stored = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extraction),
        embed=fake_embed,
    )

    assert [fact.fact for fact in stored] == [FACT]


def test_add_skips_a_fact_whose_hash_is_already_in_scope_and_logs_the_count(
    turns, make_fact, fake_embed, scripted_run_prompt, caplog, row_count
):
    stored_already = make_fact(FACT)
    insert_fact(stored_already, fake_embed([FACT])[0], [], [])
    caplog.set_level(logging.INFO)

    stored = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({}, {"fact": OTHER_FACT})),
        embed=fake_embed,
    )

    assert [fact.fact for fact in stored] == [OTHER_FACT]
    assert row_count("memories") == 2
    assert "dedupe: 1 of 2 facts skipped by hash" in caplog.text


def test_the_hash_check_is_scoped_to_the_project(
    turns, make_fact, fake_embed, scripted_run_prompt, caplog
):
    elsewhere = make_fact(FACT, project_id="other")
    insert_fact(elsewhere, fake_embed([FACT])[0], [], [])
    caplog.set_level(logging.INFO)

    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    assert stored.fact == FACT
    assert "dedupe: 0 of 1 facts skipped by hash" in caplog.text


def test_the_hash_check_is_scoped_to_the_user(
    turns, make_fact, fake_embed, scripted_run_prompt, caplog
):
    another_user = make_fact(FACT, user_id="other")
    insert_fact(another_user, fake_embed([FACT])[0], [], [])
    caplog.set_level(logging.INFO)

    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    assert (stored.fact, stored.user_id) == (FACT, "local")
    assert "dedupe: 0 of 1 facts skipped by hash" in caplog.text


def test_add_leaves_another_users_messages_out_of_the_window(
    turns, message, fake_embed, scripted_run_prompt
):
    theirs = message("we pinned the schema migration", user_id="ada")
    elsewhere = message("we moved the feature flag service", project_id="other")
    insert_messages([theirs, elsewhere])
    run_prompt = scripted_run_prompt("yes", extractor_json({}))

    add(turns, SCOPE, run_prompt=run_prompt, embed=fake_embed)

    assert theirs.content not in run_prompt.calls[0][0]
    assert elsewhere.content not in run_prompt.calls[0][0]
    assert get_unprocessed_messages("memeng", "ada") == [theirs]
    assert get_unprocessed_messages("other", "local") == [elsewhere]


@pytest.mark.parametrize(
    "answers, embedder_is_down",
    [
        (["yes", LLMTimeout("claude exceeded 120s")], False),
        (["yes"], True),
        (["maybe"], False),
        (["   "], False),
        (["yes", "I could not find a decision."], False),
        (["yes", ""], False),
        (["yes", "   "], False),
        (["yes", '[{"fact": "vectors live in sqlite-vec", "rationale": "one fi'], False),
        (["yes", '{"fact": "vectors live in sqlite-vec"}'], False),
        (["yes", '"vectors live in sqlite-vec"'], False),
        (["yes", "null"], False),
    ],
    ids=[
        "extractor timed out",
        "embedder down",
        "malformed gate answer",
        "blank gate answer",
        "prose extraction",
        "empty extraction",
        "blank extraction",
        "cut-off extraction",
        "object extraction",
        "string extraction",
        "null extraction",
    ],
)
def test_a_failure_after_the_claim_releases_the_window(
    turns, fake_embed, scripted_run_prompt, caplog, answers, embedder_is_down, row_count
):
    def unavailable(texts):
        raise EmbedUnavailable("ollama not running")

    caplog.set_level(logging.WARNING)

    stored = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt(*answers),
        embed=unavailable if embedder_is_down else fake_embed,
    )

    assert stored == []
    released = get_unprocessed_messages("memeng", "local")
    assert [message.id for message in released] == [message.id for message in turns]
    assert row_count("memories") == 0
    assert "capture released 2 messages after a failure" in caplog.text


def test_released_messages_join_the_next_capture_window(
    turns, fake_embed, scripted_run_prompt, row_count
):
    first, second = turns
    add(
        [first],
        SCOPE,
        run_prompt=scripted_run_prompt(LLMTimeout("claude exceeded 120s")),
        embed=fake_embed,
    )
    run_prompt = scripted_run_prompt("yes", extractor_json({}))

    [stored] = add([second], SCOPE, run_prompt=run_prompt, embed=fake_embed)

    assert first.content in run_prompt.calls[0][0]
    assert second.content in run_prompt.calls[0][0]
    assert row_count("fact_sources") == 2
    assert get_unprocessed_messages("memeng", "local") == []
    assert stored.fact == FACT


def test_three_gate_misses_retire_the_window(
    turns, fake_embed, scripted_run_prompt, bind_config, caplog
):
    bind_config(gate_retire_after=3)
    caplog.set_level(logging.INFO)

    for _ in range(3):
        add(turns, SCOPE, run_prompt=scripted_run_prompt("no"), embed=fake_embed)

    assert get_unprocessed_messages("memeng", "local") == []
    counts = [
        row["gate_count"] for row in DB.conn.fetchall("SELECT gate_count FROM messages")
    ]
    assert counts == [3, 3]
    assert "retired 2 messages after 3 gate misses" in caplog.text


def test_an_empty_extraction_counts_as_a_gate_miss(
    turns, fake_embed, scripted_run_prompt
):
    add(turns, SCOPE, run_prompt=scripted_run_prompt("yes", "[]"), embed=fake_embed)

    [first, second] = get_unprocessed_messages("memeng", "local")
    assert (first.gate_count, second.gate_count) == (1, 1)


def test_a_model_failure_does_not_count_as_a_gate_miss(
    turns, fake_embed, scripted_run_prompt
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt(LLMTimeout("slow")),
        embed=fake_embed,
    )

    assert [m.gate_count for m in get_unprocessed_messages("memeng", "local")] == [0, 0]


def capture_rows():
    return DB.conn.fetchall("SELECT * FROM logs_captures ORDER BY rowid")


def test_add_links_each_stored_fact_to_its_capture_row(
    turns, fake_embed, scripted_run_prompt
):
    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    [row] = capture_rows()
    [read_back] = facts_by_ids([stored.id])
    assert read_back.capture_id == row["id"]
    assert row["window_text"] == DB.conn.fetchone(
        "SELECT window_text FROM logs_captures WHERE id = ?", (read_back.capture_id,)
    )["window_text"]


def test_add_records_a_gate_no_with_the_window_the_model_read(
    turns, fake_embed, scripted_run_prompt
):
    add(turns, SCOPE, run_prompt=scripted_run_prompt("no"), embed=fake_embed)

    [row] = capture_rows()
    assert row["outcome"] == "gate_no"
    assert row["gate_answer"] == "no"
    assert row["stored_count"] == 0
    assert row["message_count"] == 2
    assert row["truncated"] == 0
    assert row["run_id"] == "run-1"
    assert row["project_id"] == "memeng"
    assert "why cache user lookups?" in row["window_text"]
    assert RATIONALE in row["window_text"]


def test_add_records_a_truncated_window_with_its_marker(
    message, fake_embed, scripted_run_prompt, bind_config
):
    bind_config(window_token_cap=20)
    turns = [message(f"turn {index} " + "detail " * 5) for index in range(5)]

    add(
        turns,
        Scope("local", "memeng"),
        run_prompt=scripted_run_prompt("yes", extractor_json({}, {"fact": OTHER_FACT})),
        embed=fake_embed,
    )

    [row] = capture_rows()
    assert row["truncated"] == 1
    assert "messages truncated ...]" in row["window_text"]
    assert row["message_count"] == 5
    assert row["stored_count"] == 2
    assert row["run_id"] is None


@pytest.mark.parametrize("gate_answer", ["yes", "  YES\n", "Yes."])
def test_add_records_a_stored_outcome_with_the_count(
    turns, fake_embed, scripted_run_prompt, gate_answer
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt(gate_answer, extractor_json({})),
        embed=fake_embed,
    )

    [row] = capture_rows()
    assert row["outcome"] == "stored"
    assert row["gate_answer"] == gate_answer
    assert row["stored_count"] == 1


def test_add_records_nothing_extracted_when_the_extractor_returns_an_empty_list(
    turns, fake_embed, scripted_run_prompt
):
    add(turns, SCOPE, run_prompt=scripted_run_prompt("yes", "[]"), embed=fake_embed)

    [row] = capture_rows()
    assert row["outcome"] == "nothing_extracted"
    assert row["stored_count"] == 0


def test_add_records_all_duplicate_when_every_fact_is_already_stored(
    turns, message, fake_embed, scripted_run_prompt
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )
    add(
        [message("and why five minutes?"), message(RATIONALE)],
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    outcomes = [row["outcome"] for row in capture_rows()]
    assert outcomes == ["stored", "all_duplicate"]


def test_add_records_failed_when_the_model_times_out(
    turns, fake_embed, scripted_run_prompt
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", LLMTimeout("claude did not answer")),
        embed=fake_embed,
    )

    [row] = capture_rows()
    assert row["outcome"] == "failed"
    assert row["gate_answer"] == "yes"
    assert row["stored_count"] == 0


def test_add_records_nothing_when_no_messages_are_unprocessed(
    turns, fake_embed, scripted_run_prompt
):
    add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    add(turns, SCOPE, run_prompt=scripted_run_prompt(), embed=fake_embed)

    assert len(capture_rows()) == 1


def test_record_decisions_false_stores_the_fact_but_skips_the_capture_log(
    turns, fake_embed, scripted_run_prompt, bind_config, row_count
):
    bind_config(record_decisions=False)

    [stored] = add(
        turns,
        SCOPE,
        run_prompt=scripted_run_prompt("yes", extractor_json({})),
        embed=fake_embed,
    )

    assert stored.capture_id is None
    assert row_count("logs_captures") == 0
    assert get_unprocessed_messages("memeng", "local") == []
