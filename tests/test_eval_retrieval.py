import json
from pathlib import Path

import pytest

from evals.retrieval import run
from memeng.facts.common import facts_by_ids

FIXTURE = Path(run.__file__).with_name("fixture.yaml")
DISTRACTORS = Path(run.__file__).with_name("distractors.yaml")


def case_of(category, expected, **overrides):
    return run.Case(
        **{
            "id": f"{category}-x",
            "category": category,
            "query": "where do vectors live",
            "distractors": 0,
            "seeds": [],
            "expected": expected,
            **overrides,
        }
    )


def outcome_of(case, returned, conflicts=()):
    return run.Outcome(
        case=case,
        returned=returned,
        conflicts=list(conflicts),
        explain={},
        expected_cosine={},
    )


def test_load_cases_reads_keys_and_expected_order():
    cases = {case.id: case for case in run.load_cases(FIXTURE)}

    assert cases["temporal-01"].expected == ["new", "old"]
    assert [seed.key for seed in cases["temporal-01"].seeds] == ["old", "new"]
    assert cases["negative-01"].expected == [] and cases["negative-01"].seeds == []


def test_load_distractors_gives_indexed_keys():
    pool = run.load_distractors(DISTRACTORS)

    assert len(pool) == 40
    assert pool[0].key == "d0" and pool[39].key == "d39"


@pytest.mark.parametrize(
    "category, expected, returned, conflicts, scores",
    [
        (
            "retrieval",
            ["a", "b"],
            ["a", "d1", "d2"],
            [],
            {"recall": 0.5, "precision": 1 / 3, "silence": None, "temporal": None},
        ),
        (
            "retrieval",
            ["a"],
            [],
            [],
            {"recall": 0.0, "precision": 0.0, "silence": None, "temporal": None},
        ),
        (
            "negative",
            [],
            [],
            [],
            {"recall": None, "precision": None, "silence": 1.0, "temporal": None},
        ),
        (
            "negative",
            [],
            ["d3"],
            [],
            {"recall": None, "precision": None, "silence": 0.0, "temporal": None},
        ),
        (
            "temporal",
            ["new", "old"],
            ["new", "old"],
            [("new", "old")],
            {"recall": 1.0, "precision": 1.0, "silence": None, "temporal": 1.0},
        ),
        (
            "temporal",
            ["new", "old"],
            ["old", "new"],
            [("new", "old")],
            {"recall": 1.0, "precision": 1.0, "silence": None, "temporal": 0.0},
        ),
        (
            "temporal",
            ["new", "old"],
            ["new", "old"],
            [],
            {"recall": 1.0, "precision": 1.0, "silence": None, "temporal": 0.0},
        ),
    ],
)
def test_score(category, expected, returned, conflicts, scores):
    assert run.score(outcome_of(case_of(category, expected), returned, conflicts)) == scores


def test_embed_cache_embeds_each_text_once(fake_embed):
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    cache = run.EmbedCache(counting)
    first = cache(["a", "b"])
    second = cache(["b", "c"])

    assert first[1] == second[0]
    assert calls == [["a", "b"], ["c"]]


def test_seed_store_inserts_seeds_and_distractors_with_the_case_order(fake_embed):
    [case] = [c for c in run.load_cases(FIXTURE) if c.id == "temporal-01"]
    pool = run.load_distractors(DISTRACTORS)

    ids = run.seed_store(
        run.Case(**{**case.__dict__, "distractors": 2}), pool, run.EmbedCache(fake_embed)
    )

    assert set(ids) == {"old", "new", "d0", "d1"}
    [old, new] = facts_by_ids([ids["old"], ids["new"]])
    assert old.created_at < new.created_at
    assert old.project_id == "eval" and old.user_id == "local"


def test_run_case_maps_returned_ids_back_to_fixture_keys(tmp_config, unit, embed_returning):
    case = case_of(
        "retrieval",
        ["a"],
        seeds=[
            run.Seed(
                "a",
                "vectors live in sqlite-vec",
                "one file",
                "architecture",
                ["vector store"],
                "2026-09-01T00:00:00Z",
            ),
            run.Seed(
                "b",
                "flags are read once",
                "hot path",
                "process",
                ["feature flag"],
                "2026-09-02T00:00:00Z",
            ),
        ],
    )
    pool = run.load_distractors(DISTRACTORS)

    outcome = run.run_case(case, pool, tmp_config, run.EmbedCache(embed_returning(unit(0))))

    assert set(outcome.returned) >= {"a", "b"}
    assert outcome.returned[0] in {"a", "b"}
    assert set(outcome.explain) == set(outcome.returned)
    assert "a" in outcome.expected_cosine


def test_summarise_averages_per_category_and_overall():
    outcomes = [
        outcome_of(case_of("retrieval", ["a"], id="r1"), ["a"]),
        outcome_of(case_of("retrieval", ["a"], id="r2"), []),
        outcome_of(case_of("negative", [], id="n1"), []),
    ]

    rows = run.summarise(outcomes)

    assert rows["retrieval"] == {
        "recall": 0.5,
        "precision": 0.5,
        "silence": None,
        "temporal": None,
        "n": 2,
    }
    assert rows["negative"]["silence"] == 1.0 and rows["negative"]["n"] == 1
    assert rows["all"]["n"] == 3 and rows["all"]["recall"] == 0.5


def test_parse_sweep():
    assert run.parse_sweep("semantic_threshold=0.3,0.4") == ("semantic_threshold", [0.3, 0.4])


@pytest.fixture
def two_case_fixture(tmp_path):
    path = tmp_path / "fixture.yaml"
    path.write_text(
        "- id: retrieval-s\n  category: retrieval\n  query: where do vectors live\n  distractors: 2\n"
        "  seed_facts:\n    - {key: v, fact: vectors live in sqlite-vec, rationale: one file,"
        " category: architecture, entities: [vector store], created_at: '2026-09-01T00:00:00Z'}\n"
        "  expected: [v]\n"
        "- id: negative-s\n  category: negative\n  query: run the tests\n  distractors: 2\n"
        "  seed_facts: []\n  expected: []\n"
    )
    return path


def test_main_smoke_runs_two_cases_with_a_fake_embedder_and_writes_a_result(
    tmp_path, tmp_config, two_case_fixture, fake_embed, capsys
):
    code = run.main(
        [
            "--fixture",
            str(two_case_fixture),
            "--distractors",
            str(DISTRACTORS),
            "--out-dir",
            str(tmp_path / "out"),
            "--explain",
        ],
        embed=fake_embed,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    [written] = (tmp_path / "out").glob("*.json")
    result = json.loads(written.read_text())
    assert result["sweep"] is None
    assert result["embed_model"] == tmp_config.embed_model
    assert set(result["rows"]["default"]) == {"retrieval", "negative", "all"}
    assert [c["id"] for c in result["cases"]] == ["retrieval-s", "negative-s"]
    assert all(isinstance(c["conflicts"], list) for c in result["cases"])
    out = capsys.readouterr().out
    assert "| value | category | recall@10 |" in out


def test_main_sweep_runs_every_case_against_a_store_that_only_that_value_seeded(
    tmp_path, tmp_config, two_case_fixture, fake_embed
):
    code = run.main(
        [
            "--fixture",
            str(two_case_fixture),
            "--distractors",
            str(DISTRACTORS),
            "--out-dir",
            str(tmp_path / "out"),
            "--sweep",
            "semantic_threshold=0.1,0.2",
        ],
        embed=fake_embed,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    [written] = (tmp_path / "out").glob("*.json")
    result = json.loads(written.read_text())
    assert result["sweep"] == "semantic_threshold"
    assert set(result["rows"]) == {"0.1", "0.2"}
    assert result["rows"]["0.1"] == result["rows"]["0.2"]
