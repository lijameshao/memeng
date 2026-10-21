import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "latency_measure",
    Path(__file__).parents[1] / "evals" / "latency" / "measure.py",
)
measure_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure_module)

Row = measure_module.Row
Step = measure_module.Step
main = measure_module.main
measure = measure_module.measure
render = measure_module.render
steps_for = measure_module.steps_for


@pytest.fixture
def recording_script(tmp_path):
    def write(body):
        script = tmp_path / "record"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        return str(script)

    return write


def test_a_parent_timed_step_records_one_wall_time_per_run():
    row = measure(Step("start", (sys.executable, "-c", "pass")), 3)

    assert len(row.samples) == 3
    assert all(sample > 0 for sample in row.samples)


def test_a_child_timed_step_records_the_number_the_child_prints():
    row = measure(
        Step("call", (sys.executable, "-c", "print(42.5)"), timed_in_child=True), 2
    )

    assert row.samples == (42.5, 42.5)


def test_a_child_timed_step_ignores_everything_the_child_printed_earlier():
    snippet = "print('loading model'); print(7.25)"

    row = measure(Step("call", (sys.executable, "-c", snippet), timed_in_child=True), 2)

    assert row.samples == (7.25, 7.25)


def test_a_failing_child_is_skipped_with_its_last_stderr_line():
    snippet = (
        "import sys; print('noise', file=sys.stderr); "
        "raise SystemExit('ollama not running')"
    )

    row = measure(Step("embed", (sys.executable, "-c", snippet)), 3)

    assert row.samples == ()
    assert row.note == "ollama not running"


def test_a_failing_child_that_says_nothing_is_skipped_with_its_exit_code():
    row = measure(Step("embed", (sys.executable, "-c", "raise SystemExit(3)")), 3)

    assert (row.samples, row.note) == ((), "exit 3")


def test_a_binary_that_is_not_installed_is_skipped_with_its_path(tmp_path):
    absent = str(tmp_path / "ollama")

    row = measure(Step("evicted", (absent, "stop", "embeddinggemma")), 3)

    assert row.samples == ()
    assert row.note == f"{absent}: No such file or directory"


def test_a_before_each_command_that_fails_skips_the_step(recording_script):
    stop = recording_script("echo 'no such model' >&2\nexit 1")

    row = measure(
        Step("evicted", (sys.executable, "-c", "pass"), before_each=(stop,)), 3
    )

    assert (row.samples, row.note) == ((), "no such model")


def test_before_each_runs_once_per_timed_run(recording_script, tmp_path):
    calls = tmp_path / "calls"
    stop = recording_script(f"echo ran >> {calls}")

    measure(Step("evicted", (sys.executable, "-c", "pass"), before_each=(stop,)), 3)

    assert calls.read_text().splitlines() == ["ran", "ran", "ran"]


def test_render_reports_the_median_and_p95_of_each_step():
    table = render([Row("python-start", tuple(float(ms) for ms in range(10, 110, 10)))])

    assert table.splitlines()[-1] == "| python-start | 55.0 | 104.5 |  |"


def test_render_marks_a_step_that_never_ran_as_skipped():
    table = render([Row("ollama-embed-warm", (), "ollama not running")])

    assert table.splitlines()[-1] == (
        "| ollama-embed-warm | - | - | skipped: ollama not running |"
    )


def test_main_measures_the_given_steps_and_prints_one_table(capsys):
    main(["--runs", "2"], steps=[Step("start", (sys.executable, "-c", "pass"))])

    printed = capsys.readouterr().out.splitlines()

    assert printed[0] == "| step | median ms | p95 ms | notes |"
    assert printed[-1].startswith("| start | ")


def test_main_refuses_a_run_count_that_has_no_p95(capsys):
    with pytest.raises(SystemExit):
        main(["--runs", "1"], steps=[])

    assert "--runs must be at least 2" in capsys.readouterr().err


def test_the_evicted_step_stops_the_model_the_config_names(tmp_config):
    evicting = [step for step in steps_for(tmp_config) if step.before_each]

    assert [step.before_each for step in evicting] == [
        ("ollama", "stop", tmp_config.embed_model)
    ]
