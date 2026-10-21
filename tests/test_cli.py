import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from memeng.capture import add
from memeng.cli import (
    main,
    memeng_root,
    register_hooks,
    run_checks,
    write_project_alias,
)
from memeng.config import CONFIG
from memeng.database.utils import DB
from memeng.embed import check_ollama, embed, embedding_dim
from memeng.facts.common import insert_fact, vectors_for
from memeng.facts.searchable_text import searchable_text
from memeng.models.common import Scope


def test_register_hooks_writes_both_entries_into_a_new_file(tmp_path):
    settings = tmp_path / "repo" / ".claude" / "settings.json"

    added = register_hooks(settings, Path("/abs/memeng"))

    assert added == ["Stop", "UserPromptSubmit"]
    written = json.loads(settings.read_text())
    assert written["hooks"]["Stop"] == [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "uv run --directory /abs/memeng python -m memeng.hooks.stop",
                }
            ]
        }
    ]
    assert written["hooks"]["UserPromptSubmit"] == [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "uv run --directory /abs/memeng python -m memeng.hooks.prompt",
                }
            ]
        }
    ]
    assert settings.read_text().endswith("}\n")


def test_register_hooks_twice_leaves_one_entry_of_each_kind(tmp_path):
    settings = tmp_path / "settings.json"
    register_hooks(settings, Path("/abs/memeng"))

    added = register_hooks(settings, Path("/abs/memeng"))

    assert added == []
    written = json.loads(settings.read_text())
    assert len(written["hooks"]["Stop"]) == 1
    assert len(written["hooks"]["UserPromptSubmit"]) == 1


def test_register_hooks_keeps_other_hooks_in_the_same_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(ls)"]},
                "hooks": {
                    "Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]
                },
            }
        )
    )

    register_hooks(settings, Path("/abs/memeng"))

    written = json.loads(settings.read_text())
    assert written["permissions"] == {"allow": ["Bash(ls)"]}
    assert [h["hooks"][0]["command"] for h in written["hooks"]["Stop"]] == [
        "say done",
        "uv run --directory /abs/memeng python -m memeng.hooks.stop",
    ]


def test_memeng_root_is_the_checkout_that_holds_pyproject():
    assert (memeng_root() / "pyproject.toml").exists()
    assert (memeng_root() / "src" / "memeng" / "cli.py").exists()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "rate-limiter"
    root.mkdir()
    return root


def test_write_project_alias_defaults_to_the_directory_name_and_keeps_an_existing_file(
    repo,
):
    assert write_project_alias(repo, None) == "rate-limiter"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'

    assert write_project_alias(repo, "other") == "rate-limiter"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'


def test_write_project_alias_uses_the_given_name_for_a_new_file(repo):
    assert write_project_alias(repo, "retry-budget") == "retry-budget"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "retry-budget"\n'


def test_run_checks_reports_each_check_by_name(repo, tmp_config):
    settings = repo / ".claude" / "settings.json"

    checks = run_checks(
        repo,
        settings,
        check_ollama=lambda: (False, "model missing; run: ollama pull embeddinggemma"),
    )

    assert [(name, ok) for name, ok, _ in checks] == [
        ("ollama", False),
        ("spacy-model", True),
        ("store", True),
        ("project-alias", False),
        ("hooks", False),
    ]
    messages = {name: message for name, _, message in checks}
    assert messages["ollama"] == "model missing; run: ollama pull embeddinggemma"
    assert str(tmp_config.db_path) in messages["store"]


def test_run_checks_passes_after_alias_and_hooks_are_written(repo):
    settings = repo / ".claude" / "settings.json"
    write_project_alias(repo, None)
    register_hooks(settings, Path("/abs/memeng"))
    requests = []

    def ollama(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3:8b"}, {"name": CONFIG.embed_model}]},
        )

    checks = run_checks(
        repo,
        settings,
        check_ollama=lambda: check_ollama(transport=httpx.MockTransport(ollama)),
    )

    assert all(ok for _, ok, _ in checks)
    assert checks[0] == ("ollama", True, f"ollama ok: {CONFIG.embed_model}")
    assert str(requests[0].url) == "http://localhost:11434/api/tags"


OLLAMA_UP = lambda: (True, "ollama ok: embeddinggemma")  # noqa: E731


def test_setup_twice_leaves_one_hook_entry_of_each_kind_and_returns_0(
    repo, tmp_config, capsys
):
    settings = repo / ".claude" / "settings.json"

    for _ in range(2):
        code = main(
            ["setup"],
            root=repo,
            stdin=io.StringIO(),
            check_ollama=OLLAMA_UP,
            load_config=lambda: tmp_config,
        )
        assert code == 0

    written = json.loads(settings.read_text())
    assert len(written["hooks"]["Stop"]) == 1
    assert len(written["hooks"]["UserPromptSubmit"]) == 1
    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'
    assert "ok hooks" in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure, printed",
    [
        (
            httpx.Response(200, json={"models": [{"name": "llama3:8b"}]}),
            "model missing; run: ollama pull embeddinggemma-2:270m",
        ),
        (
            httpx.ConnectError("[Errno 61] Connection refused"),
            "Ollama not running at http://localhost:11434",
        ),
        (
            httpx.Response(500, text="upstream exploded"),
            "Ollama not running at http://localhost:11434",
        ),
    ],
    ids=["model missing", "daemon down", "daemon erroring"],
)
def test_setup_with_ollama_down_exits_0_and_prints_the_pull_instruction(
    repo, tmp_config, capsys, transport_for, failure, printed
):
    code = main(
        ["setup"],
        root=repo,
        stdin=io.StringIO(),
        check_ollama=lambda: check_ollama(transport=transport_for(failure)),
        load_config=lambda: tmp_config,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert f"missing ollama: {printed}" in out
    assert (repo / ".claude" / "settings.json").exists()


def test_setup_honours_project_id_and_settings_arguments(repo, tmp_path, tmp_config):
    settings = tmp_path / "elsewhere" / "settings.json"

    main(
        ["setup", "--project-id", "retry-budget", "--settings", str(settings)],
        root=repo,
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert (repo / ".memeng.toml").read_text() == 'project_id = "retry-budget"\n'
    assert "memeng.hooks.stop" in settings.read_text()


def test_doctor_exits_1_before_setup_and_0_after(repo, tmp_config, capsys):
    assert (
        main(
            ["doctor"],
            root=repo,
            check_ollama=OLLAMA_UP,
            load_config=lambda: tmp_config,
        )
        == 1
    )
    assert (
        f"missing project-alias: {repo / '.memeng.toml'} missing;"
        " this directory is not enabled, run: memeng setup"
    ) in capsys.readouterr().out

    main(
        ["setup"],
        root=repo,
        stdin=io.StringIO(),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )
    capsys.readouterr()

    assert (
        main(
            ["doctor"],
            root=repo,
            check_ollama=OLLAMA_UP,
            load_config=lambda: tmp_config,
        )
        == 0
    )
    assert (
        f"ok project-alias: project_id rate-limiter from {repo / '.memeng.toml'}"
        in capsys.readouterr().out
    )


def test_reembed_rewrites_every_vector_and_the_meta_keys(
    superseding_pair, make_fact, unit, tmp_config, fake_embed, capsys
):
    older, newer = superseding_pair
    elsewhere = make_fact("feature flags default to off", project_id="other")
    insert_fact(elsewhere, unit(2), [], [])
    facts = [older, newer, elsewhere]
    texts = [searchable_text(fact.fact, fact.rationale) for fact in facts]
    requests = []

    def ollama(request):
        requests.append(request)
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": fake_embed(inputs)})

    code = main(
        ["reembed"],
        embed=lambda texts: embed(texts, transport=httpx.MockTransport(ollama)),
        embedding_dim=lambda: 8,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    [request] = requests
    assert str(request.url) == "http://localhost:11434/api/embed"
    assert json.loads(request.content) == {"model": CONFIG.embed_model, "input": texts}
    assert vectors_for([fact.id for fact in facts]) == {
        fact.id: pytest.approx(vector) for fact, vector in zip(facts, fake_embed(texts))
    }
    assert DB.conn.meta_get("embed_model") == CONFIG.embed_model
    assert "re-embedded 3 facts" in capsys.readouterr().out


def test_reembed_embeds_the_same_text_capture_embedded(
    message, tmp_config, fake_embed, scripted_run_prompt
):
    batches = []

    def recording_embed(texts):
        batches.append(texts)
        return fake_embed(texts)

    extraction = json.dumps(
        [
            {
                "fact": "the retry budget caps retries at ten percent of requests",
                "rationale": "unbounded retries turned one slow replica into an outage",
                "category": "architecture",
            }
        ]
    )
    add(
        [message("why cap retries?")],
        Scope("local", "memeng"),
        run_prompt=scripted_run_prompt("yes", extraction),
        embed=recording_embed,
    )
    captured = batches[-1]

    main(
        ["reembed"],
        embed=recording_embed,
        embedding_dim=lambda: 8,
        load_config=lambda: tmp_config,
    )

    assert batches[-1] == captured


def test_reembed_refuses_a_new_width_without_force(
    superseding_pair, tmp_config, unit, capsys
):
    older, newer = superseding_pair
    wide = lambda texts: [unit(0, dim=16) for _ in texts]  # noqa: E731

    code = main(
        ["reembed"], embed=wide, embedding_dim=lambda: 16, load_config=lambda: tmp_config
    )

    assert code == 1
    assert "8" in capsys.readouterr().out and DB.conn.meta_get("embed_dim") == "8"
    assert vectors_for([older.id]) == {older.id: unit(0)}


def test_reembed_with_force_recreates_the_vector_table(
    superseding_pair, tmp_config, unit
):
    older, newer = superseding_pair
    wide = lambda texts: [unit(0, dim=16) for _ in texts]  # noqa: E731
    probes = []

    def ollama(request):
        probes.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [unit(0, dim=16)]})

    code = main(
        ["reembed", "--force"],
        embed=wide,
        embedding_dim=lambda: embedding_dim(transport=httpx.MockTransport(ollama)),
        load_config=lambda: tmp_config,
    )

    assert code == 0
    assert probes == [["probe"]]
    assert DB.conn.meta_get("embed_dim") == "16"
    assert vectors_for([older.id, newer.id]) == {
        older.id: unit(0, dim=16),
        newer.id: unit(0, dim=16),
    }


def test_reembed_with_one_vector_too_few_raises_and_keeps_every_vector(
    superseding_pair, tmp_config, unit
):
    older, newer = superseding_pair
    one_short = lambda texts: [unit(3) for _ in texts[1:]]  # noqa: E731

    with pytest.raises(KeyError):
        main(
            ["reembed"],
            embed=one_short,
            embedding_dim=lambda: 8,
            load_config=lambda: tmp_config,
        )

    assert vectors_for([older.id, newer.id]) == {older.id: unit(0), newer.id: unit(1)}


def test_reembed_on_an_empty_store_does_nothing(tmp_config, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("no embedding on an empty store")

    code = main(
        ["reembed"], embed=refuse, embedding_dim=refuse, load_config=lambda: tmp_config
    )

    assert code == 0
    assert "nothing to re-embed" in capsys.readouterr().out


def test_judge_records_a_verdict_against_a_fact(superseding_pair, tmp_config, capsys):
    older, _ = superseding_pair

    code = main(
        ["judge", "fact", older.id, "noise", "--note", "too narrow"],
        load_config=lambda: tmp_config,
    )

    assert code == 0
    row = DB.conn.fetchone("SELECT * FROM logs_verdicts")
    assert (row["target"], row["target_id"], row["verdict"]) == (
        "fact",
        older.id,
        "noise",
    )
    assert row["judged_by"] == tmp_config.user_id
    assert row["note"] == "too narrow"
    assert older.id in capsys.readouterr().out


def test_judge_by_a_second_judge_keeps_the_first_verdict(superseding_pair, tmp_config):
    older, _ = superseding_pair

    main(["judge", "fact", older.id, "noise"], load_config=lambda: tmp_config)
    main(
        ["judge", "fact", older.id, "keep", "--by", "sonnet"],
        load_config=lambda: tmp_config,
    )

    rows = DB.conn.fetchall(
        "SELECT judged_by, verdict FROM logs_verdicts ORDER BY judged_by"
    )
    assert [tuple(row) for row in rows] == [
        (tmp_config.user_id, "noise"),
        ("sonnet", "keep"),
    ]


def test_judge_twice_by_the_same_judge_keeps_the_latest_verdict(
    superseding_pair, tmp_config
):
    older, _ = superseding_pair

    main(["judge", "fact", older.id, "noise"], load_config=lambda: tmp_config)
    main(["judge", "fact", older.id, "keep"], load_config=lambda: tmp_config)

    rows = DB.conn.fetchall("SELECT verdict FROM logs_verdicts")
    assert [row["verdict"] for row in rows] == ["keep"]


def test_judge_refuses_a_target_that_is_not_a_logged_population(tmp_config, capsys):
    with pytest.raises(SystemExit):
        main(["judge", "window", "x", "noise"], load_config=lambda: tmp_config)

    assert "(choose from fact, capture, hit)" in capsys.readouterr().err


def test_doctor_reports_a_missing_store_and_creates_nothing(repo, tmp_config, tmp_path, capsys):
    absent = tmp_path / "absent" / "store.db"

    code = main(
        ["doctor"],
        root=repo,
        check_ollama=OLLAMA_UP,
        load_config=lambda: replace(tmp_config, db_path=absent),
    )

    assert code == 1
    assert f"missing store: {absent} missing" in capsys.readouterr().out
    assert not absent.parent.exists()


def test_setup_takes_the_repository_from_the_root_argument(repo, tmp_config):
    settings = repo / ".claude" / "settings.json"

    code = main(
        ["setup", "--root", str(repo)],
        stdin=io.StringIO(),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'
    assert "memeng.hooks.stop" in settings.read_text()


def test_setup_says_when_an_existing_alias_overrides_project_id(repo, tmp_config, capsys):
    write_project_alias(repo, None)

    main(
        ["setup", "--project-id", "retry-budget"],
        root=repo,
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    out = capsys.readouterr().out
    assert "--project-id retry-budget ignored" in out
    assert "already says rate-limiter" in out


def test_setup_on_a_terminal_lists_the_users_stored_project_ids_and_takes_a_pick(
    repo, make_fact, unit, tmp_config, capsys, terminal
):
    for index, (fact, project_id, user_id) in enumerate(
        [
            ("retries stop after three attempts", "retry-budget", "local"),
            ("breakers open after five failures", "circuit-breaker", "local"),
            ("retries back off exponentially", "retry-budget", "local"),
            ("flags default to off", "feature-flags", "someone-else"),
        ]
    ):
        insert_fact(
            make_fact(fact, project_id=project_id, user_id=user_id), unit(index), [], []
        )

    code = main(
        ["setup"],
        root=repo,
        stdin=terminal("2\n"),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert code == 0
    assert (repo / ".memeng.toml").read_text() == 'project_id = "retry-budget"\n'
    assert (
        "project_id values already in the store:\n"
        "  1. circuit-breaker\n"
        "  2. retry-budget\n"
        "Pick a number, type a new project_id, or press Enter for rate-limiter: "
    ) in capsys.readouterr().out


def test_setup_on_a_terminal_takes_a_new_name(repo, tmp_config, terminal):
    main(
        ["setup"],
        root=repo,
        stdin=terminal("  feature-flags  \n"),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert (repo / ".memeng.toml").read_text() == 'project_id = "feature-flags"\n'


@pytest.mark.parametrize("typed", ["\n", ""])
def test_setup_on_a_terminal_takes_enter_or_end_of_input_as_the_directory_name(
    repo, tmp_config, typed, terminal
):
    main(
        ["setup"],
        root=repo,
        stdin=terminal(typed),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'


def test_setup_on_a_terminal_asks_again_for_a_number_that_lists_nothing(
    repo, make_fact, unit, tmp_config, capsys, terminal
):
    insert_fact(
        make_fact("breakers open after five failures", project_id="circuit-breaker"),
        unit(0),
        [],
        [],
    )

    main(
        ["setup"],
        root=repo,
        stdin=terminal("5\n1\n"),
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert (repo / ".memeng.toml").read_text() == 'project_id = "circuit-breaker"\n'
    assert "no project numbered 5" in capsys.readouterr().out


def test_setup_with_a_project_id_on_a_terminal_reads_nothing(repo, tmp_config, terminal):
    stdin = terminal("circuit-breaker\n")

    main(
        ["setup", "--project-id", "retry-budget"],
        root=repo,
        stdin=stdin,
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert stdin.read() == "circuit-breaker\n"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "retry-budget"\n'


def test_setup_without_a_terminal_reads_nothing_and_writes_the_directory_name(
    repo, tmp_config
):
    stdin = io.StringIO("circuit-breaker\n")

    main(
        ["setup"],
        root=repo,
        stdin=stdin,
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert stdin.read() == "circuit-breaker\n"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "rate-limiter"\n'


def test_setup_on_a_terminal_reads_nothing_when_the_directory_already_has_an_alias(
    repo, tmp_config, terminal
):
    write_project_alias(repo, "retry-budget")
    stdin = terminal("circuit-breaker\n")

    main(
        ["setup"],
        root=repo,
        stdin=stdin,
        check_ollama=OLLAMA_UP,
        load_config=lambda: tmp_config,
    )

    assert stdin.read() == "circuit-breaker\n"
    assert (repo / ".memeng.toml").read_text() == 'project_id = "retry-budget"\n'
