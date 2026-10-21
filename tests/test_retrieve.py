import logging
from dataclasses import replace
from pathlib import Path

import pytest

import memeng.retrieve
from memeng.embed import EmbedUnavailable
from memeng.facts.common import insert_fact
from memeng.injections.common import record_injection
from memeng.models.common import Scope
from memeng.retrieve import Conflict, SearchResult, search
from memeng.scoring import Breakdown
from memeng.supersession.common import (
    insert_supersession,
    set_supersession_status,
    supersessions_for,
)

SCOPE = Scope(user_id="local", project_id="memeng")
QUERY = [1.0, 0.0] + [0.0] * 6
VERY_NEAR = [0.96, 0.28] + [0.0] * 6
NEAR = [0.75, 0.6614378] + [0.0] * 6
LEFT = [0.8, 0.6] + [0.0] * 6
RIGHT = [0.8, -0.6] + [0.0] * 6
THIRD_AXIS = [0.745, 0.0, 0.6670645] + [0.0] * 5
OFF = [0.0, 1.0] + [0.0] * 6
JUST_UNDER = [0.79, -0.6131068] + [0.0] * 6


def test_search_drops_a_candidate_below_the_semantic_threshold(
    make_fact, bind_config, embed_returning
):
    bind_config(semantic_threshold=0.8)
    nearest = make_fact("vectors live in sqlite-vec")
    insert_fact(nearest, QUERY, [], [])
    insert_fact(make_fact("the queue is redis"), NEAR, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [nearest.id]


def test_search_returns_an_empty_result_when_nothing_clears_the_threshold(
    make_fact, tmp_config, embed_returning
):
    insert_fact(make_fact("deploys are blue-green"), OFF, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result == SearchResult(memories=[], conflicts=[], excluded_as_injected=[])


def test_search_takes_k_from_config_unless_given(
    make_fact, bind_config, embed_returning
):
    bind_config(top_k=2)
    nearest = make_fact("vectors live in sqlite-vec")
    second = make_fact("the queue is redis")
    insert_fact(nearest, QUERY, [], [])
    insert_fact(second, LEFT, [], [])
    insert_fact(make_fact("deploys are blue-green"), THIRD_AXIS, [], [])

    embed = embed_returning(QUERY)

    assert {m.fact.id for m in search("vector store", SCOPE, embed=embed).memories} == {
        nearest.id,
        second.id,
    }
    assert [
        m.fact.id for m in search("vector store", SCOPE, k=1, embed=embed).memories
    ] == [nearest.id]


def test_search_returns_an_empty_result_and_logs_once_when_embedding_is_unavailable(
    make_fact, tmp_config, caplog
):
    insert_fact(make_fact("vectors live in sqlite-vec"), QUERY, [], [])

    def unavailable(texts):
        raise EmbedUnavailable("ollama not running")

    with caplog.at_level(logging.WARNING):
        result = search("vector store", SCOPE, embed=unavailable)

    assert result == SearchResult(memories=[], conflicts=[], excluded_as_injected=[])
    logged = [record for record in caplog.records if record.name == "memeng.retrieve"]
    assert [record.levelname for record in logged] == ["WARNING"]


def test_search_reranks_by_keyword_without_adding_a_candidate(
    make_fact, tmp_config, embed_returning
):
    hit = make_fact("the retry budget caps how often we retry")
    weaker = make_fact("the retry queue drains nightly")
    miss = make_fact("deploys are blue-green")
    keyword_only = make_fact("the retry budget is per call")
    insert_fact(hit, LEFT, [], [])
    insert_fact(weaker, RIGHT, [], [])
    insert_fact(miss, THIRD_AXIS, [], [])
    insert_fact(keyword_only, OFF, [], [])

    result = search("retry budget", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [
        hit.id,
        weaker.id,
        miss.id,
    ]


def test_search_boosts_a_fact_linked_to_an_entity_in_the_query(
    make_fact, tmp_config, embed_returning
):
    text = "we tuned the rate limiter and the retry budget"
    both, one, neither = make_fact(text), make_fact(text), make_fact(text)
    insert_fact(both, LEFT, ["rate limiter", "retry budget"], [])
    insert_fact(one, RIGHT, ["rate limiter"], [])
    insert_fact(neither, NEAR, [], [])

    result = search(
        "Rate Limiter and Retry Budget",
        SCOPE,
        explain=True,
        embed=embed_returning(QUERY),
    )

    assert [memory.fact.id for memory in result.memories] == [both.id, one.id, neither.id]
    assert [memory.explain.entity_boost for memory in result.memories] == [
        tmp_config.entity_boost_cap,
        pytest.approx(tmp_config.entity_boost_cap / 2),
        0.0,
    ]
    top = result.memories[0].explain
    assert top.semantic_score == pytest.approx(0.8)
    assert top.bm25_score > 0
    assert top.final_score == pytest.approx(
        (top.semantic_score + top.bm25_score + top.entity_boost)
        / (2 + tmp_config.entity_boost_cap)
    )


def test_search_divides_by_the_configured_entity_cap(
    make_fact, bind_config, embed_returning
):
    bind_config(entity_boost_cap=1.0)
    insert_fact(make_fact("deploys are blue-green"), QUERY, ["rate limiter"], [])

    result = search("rate limiter", SCOPE, explain=True, embed=embed_returning(QUERY))

    [memory] = result.memories
    assert memory.explain.entity_boost == 1.0
    assert memory.explain.final_score == pytest.approx(2 / 3)


def test_search_scores_out_of_two_when_no_entity_is_stored(
    make_fact, tmp_config, embed_returning
):
    insert_fact(make_fact("deploys are blue-green"), QUERY, [], [])

    result = search("rate limiter", SCOPE, explain=True, embed=embed_returning(QUERY))

    breakdown = result.memories[0].explain
    assert breakdown.semantic_score == pytest.approx(1.0)
    assert breakdown.bm25_score == 0.0
    assert breakdown.entity_boost == 0.0
    assert breakdown.final_score == pytest.approx(0.5)


def test_search_ignores_an_entity_stored_for_another_user(
    make_fact, tmp_config, embed_returning
):
    insert_fact(make_fact("deploys are blue-green"), QUERY, [], [])
    insert_fact(make_fact("we tuned it", user_id="ada"), OFF, ["rate limiter"], [])
    insert_fact(
        make_fact("we tuned it too", project_id="other"), OFF, ["rate limiter"], []
    )

    result = search("rate limiter", SCOPE, explain=True, embed=embed_returning(QUERY))

    assert result.entity_ran is False
    assert result.memories[0].explain.final_score == pytest.approx(0.5)


def test_search_flags_a_near_duplicate_pair_as_a_cosine_conflict_newest_first(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    unflagged = make_fact("vectors live in pgvector", created_at="2026-09-03T00:00:00Z")
    insert_fact(newer, VERY_NEAR, [], [])
    insert_fact(older, LEFT, [], [])
    insert_fact(unflagged, JUST_UNDER, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == [
        Conflict(newer.id, older.id, pytest.approx(0.936, abs=1e-6), "cosine")
    ]
    assert [memory.fact.id for memory in result.memories] == [
        newer.id,
        older.id,
        unflagged.id,
    ]


def test_search_does_not_flag_two_facts_with_the_same_text(
    make_fact, tmp_config, embed_returning
):
    first = make_fact("vectors live in sqlite-vec")
    second = make_fact("vectors live in sqlite-vec")
    insert_fact(first, QUERY, [], [])
    insert_fact(second, QUERY, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == []
    assert len(result.memories) == 2


def test_search_orders_equal_scores_by_id(make_fact, tmp_config, embed_returning):
    insert_fact(make_fact("vectors live in sqlite-vec", id="a"), QUERY, [], [])
    insert_fact(make_fact("vectors live in sqlite-vec", id="b"), QUERY, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == ["a", "b"]


def test_search_matches_an_entity_stored_after_an_earlier_search(
    make_fact, tmp_config, embed_returning
):
    embed = embed_returning(QUERY)
    insert_fact(make_fact("we tuned the rate limiter"), QUERY, ["rate limiter"], [])
    search("rate limiter and retry budget", SCOPE, embed=embed)
    budget = make_fact("we tuned the retry budget")
    insert_fact(budget, LEFT, ["retry budget"], [])

    result = search("rate limiter and retry budget", SCOPE, explain=True, embed=embed)

    boosts = {memory.fact.id: memory.explain.entity_boost for memory in result.memories}
    assert boosts[budget.id] == pytest.approx(tmp_config.entity_boost_cap / 2)


def test_search_returns_only_facts_in_its_scope(make_fact, tmp_config, embed_returning):
    ours = make_fact("vectors live in sqlite-vec")
    insert_fact(ours, QUERY, [], [])
    insert_fact(make_fact("vectors live in qdrant", project_id="other"), QUERY, [], [])
    insert_fact(make_fact("vectors live in pgvector", user_id="ada"), QUERY, [], [])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [ours.id]


def test_search_moves_the_older_partner_directly_after_the_newer(
    make_fact, tmp_config, embed_returning
):
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    unrelated = make_fact("the queue is redis")
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    insert_fact(older, LEFT, [], [])
    insert_fact(unrelated, JUST_UNDER, [], [])
    insert_fact(newer, NEAR, [], [])

    result = search("how do we deploy", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [
        unrelated.id,
        newer.id,
        older.id,
    ]


def test_search_stops_flagging_a_pair_someone_retained(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, VERY_NEAR, [], [])
    insert_supersession(newer.id, older.id, "extractor")
    set_supersession_status(supersessions_for([newer.id])[0].id, "retained", "ada")

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == []
    assert [memory.fact.id for memory in result.memories] == [newer.id, older.id]
    assert [memory.supersession_status for memory in result.memories] == [None, None]


def test_search_annotates_a_superseded_fact_and_drops_the_pair_from_conflicts(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, VERY_NEAR, [], [])
    insert_supersession(newer.id, older.id, "extractor")
    set_supersession_status(supersessions_for([newer.id])[0].id, "superseded", "ada")

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == []
    assert [memory.fact.id for memory in result.memories] == [newer.id, older.id]
    assert result.memories[1].superseded_by == newer.id
    assert result.memories[1].supersession_status == "superseded"
    assert result.memories[0].supersedes is None


def test_search_reports_an_unresolved_extractor_proposal_as_a_conflict(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant")
    older = make_fact("vectors live in sqlite-vec")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, NEAR, [], [])
    insert_supersession(newer.id, older.id, "extractor")

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == [
        Conflict(newer.id, older.id, pytest.approx(0.75, abs=1e-6), "extractor")
    ]
    assert result.memories[0].supersedes == older.id
    assert result.memories[0].supersession_status == "unresolved"


def test_search_emits_one_conflict_when_the_extractor_and_cosine_agree(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, VERY_NEAR, [], [])
    insert_supersession(newer.id, older.id, "extractor")

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert result.conflicts == [
        Conflict(newer.id, older.id, pytest.approx(0.96, abs=1e-6), "extractor")
    ]


def test_search_keeps_a_superseded_fact_whose_replacement_was_already_injected(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, VERY_NEAR, [], [])
    insert_supersession(newer.id, older.id, "extractor")
    set_supersession_status(supersessions_for([newer.id])[0].id, "superseded", "ada")
    record_injection("run-1", [newer.id])

    scope = replace(SCOPE, run_id="run-1")

    result = search("vector store", scope, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [older.id]
    assert result.memories[0].superseded_by == newer.id
    assert result.excluded_as_injected == [newer.id]


def test_search_pairs_conflicts_before_excluding_injected_ids(
    make_fact, tmp_config, embed_returning
):
    newer = make_fact("vectors live in qdrant", created_at="2026-09-02T00:00:00Z")
    older = make_fact("vectors live in sqlite-vec", created_at="2026-09-01T00:00:00Z")
    insert_fact(newer, QUERY, [], [])
    insert_fact(older, VERY_NEAR, [], [])
    record_injection("run-1", [newer.id])

    scope = replace(SCOPE, run_id="run-1")

    result = search("vector store", scope, embed=embed_returning(QUERY))

    assert result.conflicts == [
        Conflict(newer.id, older.id, pytest.approx(0.96, abs=1e-6), "cosine")
    ]
    assert [memory.fact.id for memory in result.memories] == [older.id]


def test_search_excludes_nothing_without_a_run_id(
    make_fact, tmp_config, embed_returning
):
    fact = make_fact("vectors live in sqlite-vec")
    insert_fact(fact, QUERY, [], [])
    record_injection("run-1", [fact.id])

    result = search("vector store", SCOPE, embed=embed_returning(QUERY))

    assert [memory.fact.id for memory in result.memories] == [fact.id]
    assert result.excluded_as_injected == []


def test_search_fills_the_breakdown_only_when_asked(
    make_fact, tmp_config, embed_returning
):
    insert_fact(make_fact("vectors live in sqlite-vec"), QUERY, [], [])
    insert_fact(make_fact("the queue is redis"), NEAR, [], [])

    embed = embed_returning(QUERY)
    quiet = search("vector store", SCOPE, embed=embed)
    loud = search("vector store", SCOPE, explain=True, embed=embed)

    assert [memory.explain for memory in quiet.memories] == [None, None]
    assert all(isinstance(memory.explain, Breakdown) for memory in loud.memories)
    assert all(memory.explain.final_score == memory.score for memory in loud.memories)


def test_search_returns_a_breakdown_for_every_ranked_memory(
    make_fact, tmp_config, embed_returning
):
    near = make_fact("the retry budget is three attempts")
    far = make_fact("deploys are blue-green")
    insert_fact(near, QUERY, [], [])
    insert_fact(far, LEFT, [], [])

    result = search("retry budget", SCOPE, embed=embed_returning(QUERY))

    assert set(result.scored) == {near.id, far.id}
    [ranked_near] = [memory for memory in result.memories if memory.fact.id == near.id]
    assert result.scored[near.id].final_score == ranked_near.score


def test_search_scores_a_memory_it_withheld_as_already_injected(
    make_fact, tmp_config, embed_returning
):
    already = make_fact("the retry budget is three attempts")
    insert_fact(already, QUERY, [], [])
    record_injection("run-1", [already.id])
    scope = Scope(user_id="local", project_id="memeng", run_id="run-1")

    result = search("retry budget", scope, embed=embed_returning(QUERY))

    assert result.memories == []
    assert result.excluded_as_injected == [already.id]
    assert already.id in result.scored


def test_search_says_which_signals_ran(make_fact, tmp_config, embed_returning):
    insert_fact(make_fact("the retry budget is three attempts"), QUERY, ["retry budget"], [])

    worded = search("retry budget", SCOPE, embed=embed_returning(QUERY))
    wordless = search("!!!", SCOPE, explain=True, embed=embed_returning(QUERY))

    assert (worded.bm25_ran, worded.entity_ran) == (True, True)
    assert wordless.bm25_ran is False
    assert wordless.memories[0].explain == Breakdown(
        semantic_score=pytest.approx(1.0),
        bm25_score=0.0,
        entity_boost=0.0,
        final_score=pytest.approx(1.0),
    )


def test_retrieve_writes_no_sql():
    source = Path(memeng.retrieve.__file__).read_text().upper()

    assert "INSERT INTO" not in source
    assert "UPDATE " not in source
    assert "DELETE FROM" not in source
