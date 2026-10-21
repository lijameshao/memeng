import json
from pathlib import Path

import pytest

from evals.gate import run


@pytest.fixture
def two_case_fixture(tmp_path):
    path = tmp_path / "fixture.yaml"
    path.write_text(
        "- id: g-yes\n"
        "  label: yes\n"
        "  window: |\n"
        "    user: we picked sqlite-vec because it is one file and needs no daemon\n"
        "- id: g-no\n"
        "  label: no\n"
        "  window: |\n"
        "    user: run the tests\n"
        "\n"
        "    assistant: 42 passed\n"
    )
    return path


@pytest.mark.parametrize(
    "answer, fn_rate, fp_rate, misses",
    [
        ("yes", 0.0, 1.0, [("g-no", "no", "yes")]),
        ("no", 1.0, 0.0, [("g-yes", "yes", "no")]),
        ("maybe", 1.0, 0.0, [("g-yes", "yes", "malformed"), ("g-no", "no", "malformed")]),
    ],
)
def test_main_reports_the_false_negative_and_false_positive_rates(
    tmp_path, tmp_config, two_case_fixture, fake_model_id, answer, fn_rate, fp_rate, misses
):
    exit_code = run.main(
        ["--model", "haiku", "--fixture", str(two_case_fixture), "--out-dir", str(tmp_path)],
        run_prompt=lambda prompt, model: answer,
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    [written] = tmp_path.glob("*.json")
    result = json.loads(written.read_text())
    assert exit_code == 0
    assert written.name.endswith("-haiku.json")
    assert (result["model"], result["model_id"], result["n"]) == ("haiku", "claude-haiku-5-5", 2)
    assert (result["fn_rate"], result["fp_rate"]) == (fn_rate, fp_rate)
    assert [(m["id"], m["label"], m["answer"]) for m in result["misses"]] == misses


def test_main_reports_a_window_head_that_names_the_missed_case(
    tmp_path, tmp_config, two_case_fixture, fake_model_id
):
    run.main(
        ["--model", "haiku", "--fixture", str(two_case_fixture), "--out-dir", str(tmp_path)],
        run_prompt=lambda prompt, model: "yes",
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    [written] = tmp_path.glob("*.json")
    [miss] = json.loads(written.read_text())["misses"]
    assert miss["window_head"] == "user: run the tests"


def test_main_renders_the_bundled_gate_prompt_around_each_window(
    tmp_path, tmp_config, two_case_fixture, fake_model_id
):
    calls = []

    run.main(
        ["--model", "sonnet", "--fixture", str(two_case_fixture), "--out-dir", str(tmp_path)],
        run_prompt=lambda prompt, model: calls.append((prompt, model)) or "yes",
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    assert [model for _, model in calls] == ["sonnet", "sonnet"]
    assert "we picked sqlite-vec" in calls[0][0]
    assert "user: run the tests\n\nassistant: 42 passed" in calls[1][0]
    for prompt, _ in calls:
        assert "Answer with exactly one word: yes or no." in prompt


def test_the_committed_fixture_carries_thirty_labelled_cases_with_unique_ids():
    cases = run.load_cases(Path(run.__file__).with_name("fixture.yaml"))

    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)
    assert {case.expected for case in cases} == {True, False}


def test_load_cases_rejects_a_label_that_is_not_yes_or_no(tmp_path):
    path = tmp_path / "fixture.yaml"
    path.write_text("- id: g-01\n  label: maybe\n  window: |\n    user: run the tests\n")

    with pytest.raises(ValueError, match="maybe"):
        run.load_cases(path)
