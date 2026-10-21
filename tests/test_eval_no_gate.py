import json

import pytest

from evals.gate import no_gate

INVENTED = json.dumps(
    [
        {
            "fact": "tests run before every push",
            "rationale": "the team agreed it",
            "category": "process",
            "entities": [],
            "supersedes": None,
        }
    ]
)


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


def invent_on_no_windows(prompt, model):
    return INVENTED if "run the tests" in prompt else "[]"


def test_main_reports_invented_and_empty_yes_rates(
    tmp_path, tmp_config, two_case_fixture, fake_model_id
):
    code = no_gate.main(
        ["--model", "sonnet", "--fixture", str(two_case_fixture), "--out-dir", str(tmp_path)],
        run_prompt=invent_on_no_windows,
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    [written] = tmp_path.glob("*.json")
    result = json.loads(written.read_text())
    assert code == 0
    assert written.name.endswith("-no-gate-sonnet.json")
    assert (result["model"], result["model_id"], result["n"]) == ("sonnet", "claude-sonnet-5-5", 2)
    assert (result["invented_rate"], result["empty_yes_rate"]) == (1.0, 1.0)
    assert result["malformed"] == 0
    [invented] = result["invented"]
    assert invented["id"] == "g-no"
    assert invented["window_head"] == "user: run the tests"
    assert invented["facts"] == [
        {"fact": "tests run before every push", "rationale": "the team agreed it"}
    ]
    assert [item["id"] for item in result["empty_yes"]] == ["g-yes"]


def test_main_counts_a_malformed_answer_and_keeps_going(
    tmp_path, tmp_config, two_case_fixture, fake_model_id
):
    code = no_gate.main(
        ["--model", "sonnet", "--fixture", str(two_case_fixture), "--out-dir", str(tmp_path)],
        run_prompt=lambda prompt, model: "I would rather read the file first",
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    [written] = tmp_path.glob("*.json")
    result = json.loads(written.read_text())
    assert code == 0
    assert result["malformed"] == 2
    assert result["invented_rate"] == 0.0
    assert result["empty_yes_rate"] == 1.0


def test_report_gives_none_for_a_rate_with_no_cases_of_that_label(
    tmp_path, tmp_config, fake_model_id
):
    fixture = tmp_path / "fixture.yaml"
    fixture.write_text(
        "- id: only\n  label: yes\n  window: |\n    user: we cache because reads dominate\n"
    )

    no_gate.main(
        ["--model", "sonnet", "--fixture", str(fixture), "--out-dir", str(tmp_path)],
        run_prompt=lambda prompt, model: "[]",
        model_id=fake_model_id,
        load_config=lambda: tmp_config,
    )

    [written] = tmp_path.glob("*.json")
    result = json.loads(written.read_text())
    assert result["invented_rate"] is None
    assert result["empty_yes_rate"] == 1.0
