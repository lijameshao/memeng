import os

import pytest

from memeng.llm import (
    LLMError,
    LLMMalformed,
    LLMRateLimited,
    LLMTimeout,
    model_id,
    run_prompt,
)


@pytest.fixture
def fake_claude(tmp_path):
    def write(body):
        script = tmp_path / "claude"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        return str(script)

    return write


@pytest.fixture
def claude_dumping_its_call(fake_claude):
    return fake_claude(
        "prompt=$(cat)\n"
        "printf '{\"result\": \"%s|%s|%s\"}' \"$MEMENG_GATE\" \"$*\" \"$prompt\""
    )


def test_returns_the_result_field_of_the_json_envelope(fake_claude):
    envelope = (
        '{"type":"result","subtype":"success","is_error":false,'
        '"result":"sqlite-vec, not chroma"}'
    )
    claude = fake_claude(f"echo '{envelope}'")

    answer = run_prompt("which vector store", "haiku", claude_bin=claude)

    assert answer == "sqlite-vec, not chroma"


def test_model_id_returns_the_model_that_answered_for_the_name_passed(fake_claude):
    claude = fake_claude(
        'printf \'{"result":"ok","modelUsage":{"claude-%s-5-5":{}}}\' "$3"'
    )

    assert model_id("sonnet", claude_bin=claude) == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    "usage",
    ["", ',"modelUsage":{}', ',"modelUsage":{"claude-sonnet-5-5":{},"claude-haiku-5-5":{}}'],
    ids=["missing", "empty", "two models"],
)
def test_model_id_maps_an_envelope_without_one_model_to_llm_malformed(fake_claude, usage):
    claude = fake_claude(f"""echo '{{"result":"ok"{usage}}}'""")

    with pytest.raises(LLMMalformed):
        model_id("sonnet", claude_bin=claude)


def test_sets_the_recursion_guard_in_the_child_environment(claude_dumping_its_call):
    gate, _, _ = run_prompt(
        "who wins", "haiku", claude_bin=claude_dumping_its_call
    ).split("|")

    assert gate == "1"


def test_leaves_the_rest_of_the_parent_environment_in_place(fake_claude):
    claude = fake_claude('printf \'{"result": "%s"}\' "$HOME"')

    assert run_prompt("who wins", "haiku", claude_bin=claude) == os.environ["HOME"]


def test_runs_print_mode_with_the_model_and_every_tool_disallowed(
    claude_dumping_its_call,
):
    _, argv, _ = run_prompt(
        "who wins", "haiku", claude_bin=claude_dumping_its_call
    ).split("|")

    assert argv == (
        "-p --model haiku --output-format json "
        "--no-session-persistence --disallowed-tools *"
    )


def test_passes_the_prompt_on_stdin_rather_than_as_an_argument(
    claude_dumping_its_call,
):
    _, argv, stdin = run_prompt(
        "who wins", "haiku", claude_bin=claude_dumping_its_call
    ).split("|")

    assert stdin == "who wins"
    assert "who wins" not in argv


@pytest.mark.parametrize(
    "body, failure, reported",
    [
        ("echo 'not json'", LLMMalformed, "not json"),
        ("echo 42", LLMMalformed, "42"),
        ("printf '\\377'", LLMMalformed, "\ufffd"),
        (
            """echo '{"is_error":false}'""",
            LLMMalformed,
            '{"is_error":false}',
        ),
        (
            """echo '{"result":null}'""",
            LLMMalformed,
            '{"result":null}',
        ),
        (
            """echo '{"result":{"text":"sqlite-vec"}}'""",
            LLMMalformed,
            '{"result":{"text":"sqlite-vec"}}',
        ),
        (
            'echo "Rate Limit reached, retry later" >&2\nexit 1',
            LLMRateLimited,
            "Rate Limit reached, retry later",
        ),
        (
            """echo '{"is_error":true,"result":"API error 429"}'""",
            LLMRateLimited,
            "API error 429",
        ),
        (
            """echo '{"is_error":true,"result":"credentials expired"}'""",
            LLMError,
            "credentials expired",
        ),
        (
            'echo "credentials expired" >&2\nexit 1',
            LLMError,
            "credentials expired",
        ),
    ],
)
def test_maps_every_child_failure_to_its_named_error(
    fake_claude, body, failure, reported
):
    claude = fake_claude(body)

    with pytest.raises(LLMError) as raised:
        run_prompt("which vector store", "haiku", claude_bin=claude)

    assert type(raised.value) is failure
    assert reported in str(raised.value)


@pytest.mark.parametrize(
    "permissions", [None, 0o644], ids=["not installed", "not executable"]
)
def test_maps_a_binary_that_will_not_run_to_llm_error(tmp_path, permissions):
    claude = tmp_path / "claude"
    if permissions is not None:
        claude.write_text("#!/bin/sh\n")
        claude.chmod(permissions)

    with pytest.raises(LLMError) as raised:
        run_prompt("which vector store", "haiku", claude_bin=str(claude))

    assert type(raised.value) is LLMError
    assert str(claude) in str(raised.value)


@pytest.mark.parametrize(
    "body, failure",
    [
        ("head -c 4000 /dev/zero | tr '\\0' x", LLMMalformed),
        ("head -c 4000 /dev/zero | tr '\\0' x >&2\nexit 1", LLMError),
        (
            "echo \"rate limit\" >&2\nhead -c 4000 /dev/zero | tr '\\0' x >&2\nexit 1",
            LLMRateLimited,
        ),
    ],
)
def test_caps_a_flood_of_child_output_at_five_hundred_characters(
    fake_claude, body, failure
):
    claude = fake_claude(body)

    with pytest.raises(failure) as raised:
        run_prompt("which vector store", "haiku", claude_bin=claude)

    assert len(str(raised.value)) == 500


def test_maps_a_call_that_outlives_its_budget_to_llm_timeout(fake_claude):
    claude = fake_claude("sleep 5")

    with pytest.raises(LLMTimeout):
        run_prompt("which vector store", "haiku", 0.5, claude_bin=claude)
