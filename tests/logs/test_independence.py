import json

from memeng.capture import add
from memeng.cli import main as judge_main
from memeng.database.utils import DB
from memeng.hooks.prompt import hook_output
from memeng.hooks.prompt import main as record_main
from memeng.models.common import Scope
from memeng.retrieve import search

SCOPE = Scope("local", "memeng", run_id="run-1")
EXTRACTION = json.dumps(
    [
        {
            "fact": "the retry budget is three attempts",
            "rationale": "a fourth attempt never succeeded in the logs",
            "category": "constraint",
            "entities": ["retry budget"],
            "supersedes": None,
        }
    ]
)


def a_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".memeng.toml").write_text('project_id = "memeng"\n')
    return root


def a_payload(root, session_id):
    return {
        "cwd": str(root),
        "session_id": session_id,
        "prompt": "how many times do we retry?",
        "hook_event_name": "UserPromptSubmit",
    }


def test_deleting_every_logs_row_leaves_the_injected_block_unchanged(
    message, fake_embed, scripted_run_prompt, row_count, tmp_config, tmp_path
):
    root = a_repo(tmp_path)
    [stored] = add(
        [
            message("how many times do we retry?"),
            message("a fourth attempt never succeeded in the logs"),
        ],
        SCOPE,
        run_prompt=scripted_run_prompt("yes", EXTRACTION),
        embed=fake_embed,
    )

    recorded = []
    before = hook_output(
        a_payload(root, "s1"),
        search=lambda query, scope: search(query, scope, embed=fake_embed),
        spawn=lambda argv, log_path: recorded.append(argv),
    )
    [argv] = recorded
    record_main(["--record", argv[4]], env={}, load_config=lambda: tmp_config)
    judge_main(["judge", "fact", stored.id, "noise"], load_config=lambda: tmp_config)

    assert row_count("logs_captures") > 0
    assert row_count("logs_retrieval") > 0
    assert row_count("logs_retrieval_hits") > 0
    assert row_count("logs_verdicts") > 0

    DB.conn.execute("DELETE FROM logs_retrieval")
    assert row_count("logs_retrieval_hits") == 0
    DB.conn.execute("DELETE FROM logs_captures")
    DB.conn.execute("DELETE FROM logs_verdicts")

    after = hook_output(
        a_payload(root, "s2"),
        search=lambda query, scope: search(query, scope, embed=fake_embed),
        spawn=lambda argv, log_path: None,
    )

    assert before is not None
    assert after == before
