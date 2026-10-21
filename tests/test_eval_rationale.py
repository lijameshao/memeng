import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evals.rationale import run
from memeng.facts.common import insert_fact
from memeng.llm import LLMMalformed
from memeng.messages.common import insert_messages
from memeng.models.common import Scope

SCOPE = Scope(user_id="local", project_id="memeng", run_id="s1")


@pytest.fixture
def stored_fact_with_two_sources(make_fact, message, unit):
    first = message("cap retries at three", id="m1", created_at="2026-09-20T10:01:00Z")
    second = message(
        "the upstream p99 is three seconds",
        id="m2",
        role="assistant",
        created_at="2026-09-20T10:02:00Z",
    )
    insert_messages([first, second])
    fact = make_fact(
        "retries are capped at three", rationale="the upstream p99 is three seconds"
    )
    insert_fact(fact, unit(0), [], ["m1", "m2"])
    return fact


def test_facts_with_sources_renders_the_source_turns_oldest_first(
    stored_fact_with_two_sources,
):
    [item] = run.facts_with_sources(SCOPE)

    assert item.fact.id == stored_fact_with_two_sources.id
    assert item.message_ids == ["m1", "m2"]
    assert (
        item.turns
        == "user: cap retries at three\n\nassistant: the upstream p99 is three seconds"
    )


def test_render_judge_fills_both_placeholders():
    prompt = run.render_judge("because reads dominate", "user: cache it")

    assert "because reads dominate" in prompt
    assert "user: cache it" in prompt
    assert "{rationale}" not in prompt and "{turns}" not in prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"faithful": true, "why": "stated in turn two"}', (True, "stated in turn two")),
        ('```json\n{"faithful": false, "why": "nobody said it"}\n```', (False, "nobody said it")),
    ],
)
def test_parse_verdict(text, expected):
    assert run.parse_verdict(text) == expected


def test_parse_verdict_rejects_prose():
    with pytest.raises(LLMMalformed):
        run.parse_verdict("I think it is faithful.")


def test_judge_reports_faithful_and_unfaithful_facts(
    stored_fact_with_two_sources, make_fact, message, unit
):
    third = message("we drain the pool", id="m3")
    insert_messages([third])
    other = make_fact(
        "the deploy drains the old pool", rationale="draining avoids dropped requests"
    )
    insert_fact(other, unit(1), [], ["m3"])
    verdicts = {
        "the upstream p99 is three seconds": '{"faithful": true, "why": "turn two says it"}',
        "draining avoids dropped requests": '{"faithful": false, "why": "no turn gives a reason"}',
    }

    def run_prompt(prompt, model):
        for rationale, verdict in verdicts.items():
            if rationale in prompt:
                return verdict
        raise AssertionError(prompt)

    report = run.judge(SCOPE, run_prompt=run_prompt)

    assert (report["facts"], report["faithful"]) == (2, 1)
    assert sorted(report["stored"], key=lambda entry: entry["fact"]) == [
        {
            "fact": "retries are capped at three",
            "rationale": "the upstream p99 is three seconds",
            "message_ids": ["m1", "m2"],
            "faithful": True,
            "why": "turn two says it",
        },
        {
            "fact": "the deploy drains the old pool",
            "rationale": "draining avoids dropped requests",
            "message_ids": ["m3"],
            "faithful": False,
            "why": "no turn gives a reason",
        },
    ]
    assert report["unfaithful"] == [
        {
            "fact": "the deploy drains the old pool",
            "rationale": "draining avoids dropped requests",
            "why": "no turn gives a reason",
            "message_ids": ["m3"],
        }
    ]
    assert report["malformed"] == []


def test_judge_records_an_unjudged_fact_when_the_verdict_is_malformed(
    stored_fact_with_two_sources,
):
    report = run.judge(SCOPE, run_prompt=lambda prompt, model: "I think it is faithful.")

    assert (report["facts"], report["faithful"], report["unfaithful"]) == (1, 0, [])
    assert report["malformed"] == [stored_fact_with_two_sources.id]
    assert [(entry["fact"], entry["faithful"]) for entry in report["stored"]] == [
        ("retries are capped at three", None)
    ]


TRANSCRIPTS = Path(run.__file__).with_name("transcripts")


def seeding_add(make_fact, unit):
    def add(messages, scope):
        insert_messages(messages)
        fact = make_fact(
            "retries are capped at three",
            rationale="the upstream p99 is three seconds",
            project_id=scope.project_id,
        )
        insert_fact(fact, unit(0), [], [m.id for m in messages])
        return [fact]

    return add


def storing_nothing(messages, scope):
    insert_messages(messages)
    return []


FAITHFUL = lambda prompt, model: '{"faithful": true, "why": "stated"}'  # noqa: E731


def test_run_transcript_reports_the_facts_the_pipeline_stored(tmp_config, make_fact, unit):
    report = run.run_transcript(
        TRANSCRIPTS / "01-explicit-reason.jsonl",
        tmp_config,
        add=seeding_add(make_fact, unit),
        run_prompt=FAITHFUL,
    )

    assert report["transcript"] == "01-explicit-reason.jsonl"
    assert (report["facts"], report["faithful"], report["unfaithful"]) == (1, 1, [])


def test_main_exits_1_when_an_expectation_is_violated(
    tmp_path, tmp_config, fake_model_id, capsys
):
    expectations = tmp_path / "expectations.yaml"
    expectations.write_text(
        yaml.safe_dump([{"transcript": "01-explicit-reason.jsonl", "expect_stored": True}])
    )

    code = run.main(
        ["--transcripts", str(TRANSCRIPTS), "--expectations", str(expectations), "--out-dir", str(tmp_path / "out")],
        add=storing_nothing,
        run_prompt=FAITHFUL,
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    assert code == 1
    [written] = (tmp_path / "out").glob("*.json")
    result = json.loads(written.read_text())
    assert result["violations"] == [
        {"transcript": "01-explicit-reason.jsonl", "expect_stored": True, "facts": 0}
    ]
    assert "01-explicit-reason.jsonl" in capsys.readouterr().out


def test_main_exits_0_and_records_the_models_when_every_expectation_holds(
    tmp_path, tmp_config, make_fact, unit, fake_model_id
):
    expectations = tmp_path / "expectations.yaml"
    expectations.write_text(
        yaml.safe_dump([{"transcript": "01-explicit-reason.jsonl", "expect_stored": True}])
    )

    code = run.main(
        ["--transcripts", str(TRANSCRIPTS), "--expectations", str(expectations), "--out-dir", str(tmp_path / "out")],
        add=seeding_add(make_fact, unit),
        run_prompt=FAITHFUL,
        model_id=fake_model_id,
        load_config=lambda: replace(tmp_config, gate_model="haiku", extract_model="opus"),
    )

    assert code == 0
    [written] = (tmp_path / "out").glob("*.json")
    result = json.loads(written.read_text())
    assert result["embed_model"] == tmp_config.embed_model
    assert result["models"] == {
        "gate": {"model": "haiku", "model_id": "claude-haiku-5-5"},
        "extract": {"model": "opus", "model_id": "claude-opus-5-5"},
        "judge": {"model": "sonnet", "model_id": "claude-sonnet-5-5"},
    }
