import fcntl
import io
import json
import logging
import os
import sys
from dataclasses import replace

import pytest

from memeng.database.utils import DB
from memeng.hooks.stop import capture, main, read_new_turns
from memeng.models.common import Scope

SCOPE = Scope(user_id="local", project_id="memeng", run_id="s1")


def transcript_line(kind, uuid, content, **extra):
    return {
        "type": kind,
        "uuid": uuid,
        "timestamp": f"2026-09-20T10:00:{uuid[-1]}0Z",
        "message": {"role": kind, "content": content},
        **extra,
    }


@pytest.fixture
def transcript(tmp_path):
    def write(*lines):
        path = tmp_path / "session.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        return path

    return write


def test_read_new_turns_keeps_text_and_drops_tool_blocks_meta_and_summaries(transcript, message):
    path = transcript(
        transcript_line("user", "u1", "should we cache the rate limiter state?"),
        transcript_line(
            "assistant",
            "a2",
            [
                {"type": "text", "text": "Yes, for five minutes."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ],
        ),
        transcript_line("user", "u3", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
        transcript_line("user", "u4", "<command-name>/model</command-name>", isMeta=True),
        transcript_line("user", "u5", "side quest", isSidechain=True),
        {"type": "summary", "summary": "irrelevant"},
    )

    turns = read_new_turns(path, None, SCOPE)

    assert turns == [
        message(
            "should we cache the rate limiter state?",
            id="u1",
            created_at="2026-09-20T10:00:10Z",
            run_id="s1",
        ),
        message(
            "Yes, for five minutes.",
            role="assistant",
            id="a2",
            created_at="2026-09-20T10:00:20Z",
            run_id="s1",
        ),
    ]


def test_read_new_turns_returns_only_the_lines_after_since(transcript):
    path = transcript(
        transcript_line("user", "u1", "first"),
        transcript_line("assistant", "a2", "second"),
        transcript_line("user", "u3", "third"),
    )

    turns = read_new_turns(path, "a2", SCOPE)

    assert [turn.id for turn in turns] == ["u3"]


def test_read_new_turns_without_since_takes_the_last_configured_count(transcript, bind_config):
    bind_config(first_capture_turns=2)
    path = transcript(
        transcript_line("user", "u1", "first"),
        transcript_line("assistant", "a2", "second"),
        transcript_line("user", "u3", "third"),
    )

    turns = read_new_turns(path, None, SCOPE)

    assert [turn.id for turn in turns] == ["a2", "u3"]


def test_read_new_turns_with_an_unknown_since_falls_back_to_the_last_configured_count(transcript, bind_config):
    bind_config(first_capture_turns=1)
    path = transcript(
        transcript_line("user", "u1", "first"),
        transcript_line("assistant", "a2", "second"),
    )

    turns = read_new_turns(path, "gone", SCOPE)

    assert [turn.id for turn in turns] == ["a2"]


@pytest.fixture
def recording_add():
    def add(messages, scope):
        add.calls.append((messages, scope))
        return []

    add.calls = []
    return add


@pytest.fixture
def repo(tmp_path):
    def build(project_id="retry-budget"):
        root = tmp_path / project_id.replace("/", "-")
        root.mkdir()
        (root / ".memeng.toml").write_text(f'project_id = "{project_id}"\n')
        return root

    return build


def test_capture_reads_the_new_turns_calls_add_and_records_the_last_uuid(transcript, repo, recording_add):
    path = transcript(
        transcript_line("user", "u1", "first"),
        transcript_line("assistant", "a2", "second"),
    )
    DB.conn.meta_set("last_uuid:s1", "u1")

    assert capture(path, "s1", repo(), add=recording_add) == 0

    [(messages, scope)] = recording_add.calls
    assert [m.id for m in messages] == ["a2"]
    assert scope == Scope(user_id="local", project_id="retry-budget", run_id="s1")
    assert DB.conn.meta_get("last_uuid:s1") == "a2"


def test_capture_builds_the_scope_with_the_configured_user_id(transcript, repo, recording_add, bind_config):
    bind_config(user_id="alice")
    path = transcript(transcript_line("user", "u1", "first"))

    assert capture(path, "s1", repo(), add=recording_add) == 0

    [(_, scope)] = recording_add.calls
    assert scope.user_id == "alice"


def test_capture_with_no_new_turns_calls_nothing(transcript, repo, recording_add):
    path = transcript(transcript_line("user", "u1", "first"))
    DB.conn.meta_set("last_uuid:s1", "u1")

    assert capture(path, "s1", repo(), add=recording_add) == 0

    assert recording_add.calls == []


def test_capture_exits_when_another_capture_holds_the_project_lock(transcript, repo, recording_add, tmp_config, caplog):
    caplog.set_level(logging.INFO)
    path = transcript(transcript_line("user", "u1", "first"))
    lock_path = tmp_config.db_path.parent / "capture-retry-budget.lock"
    held = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        assert capture(path, "s1", repo(), add=recording_add) == 0
    finally:
        os.close(held)

    assert recording_add.calls == []
    assert "capture already running for retry-budget" in caplog.text


@pytest.fixture
def recording_spawn():
    def spawn(argv, log_path):
        spawn.calls.append((argv, log_path))

    spawn.calls = []
    return spawn


def hook_stdin(**overrides):
    payload = {
        "session_id": "s1",
        "transcript_path": "/tmp/s1.jsonl",
        "cwd": "/tmp/repo",
        "hook_event_name": "Stop",
        "stop_hook_active": False,
    } | overrides
    return io.StringIO(json.dumps(payload))


def test_main_in_hook_mode_spawns_the_capture_child_and_returns_0(recording_spawn, tmp_config):
    code = main([], stdin=hook_stdin(), env={}, spawn=recording_spawn, load_config=lambda: tmp_config)

    assert code == 0
    [(argv, log_path)] = recording_spawn.calls
    assert argv == [
        sys.executable, "-m", "memeng.hooks.stop", "--capture", "/tmp/s1.jsonl", "s1", "/tmp/repo",
    ]
    assert log_path == tmp_config.db_path.parent / "capture.log"


def test_main_returns_0_without_spawning_when_the_recursion_guard_is_set(recording_spawn, tmp_config):
    code = main([], stdin=hook_stdin(), env={"MEMENG_GATE": "1"}, spawn=recording_spawn, load_config=lambda: tmp_config)

    assert code == 0
    assert recording_spawn.calls == []


def test_main_returns_0_without_spawning_when_a_stop_hook_is_already_active(recording_spawn, tmp_config):
    code = main(
        [], stdin=hook_stdin(stop_hook_active=True), env={}, spawn=recording_spawn, load_config=lambda: tmp_config
    )

    assert code == 0
    assert recording_spawn.calls == []


def test_main_in_hook_mode_fails_open_on_a_bad_payload(recording_spawn, tmp_config, capsys):
    code = main([], stdin=io.StringIO("not json"), env={}, spawn=recording_spawn, load_config=lambda: tmp_config)

    assert code == 0
    assert recording_spawn.calls == []
    assert "JSONDecodeError" in capsys.readouterr().err


def test_main_in_capture_mode_opens_the_store_and_runs_capture(
    transcript, repo, recording_add, tmp_config, tmp_path
):
    path = transcript(transcript_line("user", "u1", "first"))
    config = replace(tmp_config, db_path=tmp_path / "absent" / "store.db")

    code = main(
        ["--capture", str(path), "s1", str(repo())],
        env={},
        load_config=lambda: config,
        add=recording_add,
    )

    assert code == 0
    [(messages, scope)] = recording_add.calls
    assert [m.id for m in messages] == ["u1"]
    assert scope.project_id == "retry-budget"
    assert config.db_path.exists()
    assert DB.conn.meta_get("schema_version") == "1"
    assert DB.conn.meta_get("embed_model") == config.embed_model
    assert DB.conn.fetchone("PRAGMA journal_mode")[0] == "wal"
    assert DB.conn.fetchone("PRAGMA busy_timeout")[0] == 5000


def test_capture_locks_on_a_flat_filename_when_the_project_alias_holds_a_separator(
    transcript, repo, recording_add, tmp_config, caplog
):
    caplog.set_level(logging.INFO)
    path = transcript(transcript_line("user", "u1", "first"))
    lock_path = tmp_config.db_path.parent / "capture-platform_rate-limiter.lock"
    held = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        assert capture(path, "s1", repo("platform/rate-limiter"), add=recording_add) == 0
    finally:
        os.close(held)

    assert recording_add.calls == []
    assert "capture already running for platform/rate-limiter" in caplog.text


def test_main_in_capture_mode_captures_nothing_where_there_is_no_alias(
    transcript, tmp_path, recording_add, tmp_config
):
    path = transcript(transcript_line("user", "u1", "first"))
    cwd = tmp_path / "feature-flags"
    cwd.mkdir()

    code = main(
        ["--capture", str(path), "s1", str(cwd)],
        env={},
        load_config=lambda: tmp_config,
        add=recording_add,
    )

    assert code == 0
    assert recording_add.calls == []
    assert DB.conn.meta_get("last_uuid:s1") is None
