import argparse
import importlib.util
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

import memeng
from memeng import embed as embed_module
from memeng.config import CONFIG, Config
from memeng.database.utils import DB
from memeng.enums.common import VerdictTarget
from memeng.facts.common import get_all_facts, get_project_ids, replace_all_vectors
from memeng.facts.searchable_text import searchable_text
from memeng.logs.common import insert_verdict
from memeng.models.logs import Verdict
from memeng.project import resolve_project_id

HOOK_MODULES = {"Stop": "memeng.hooks.stop", "UserPromptSubmit": "memeng.hooks.prompt"}


def memeng_root() -> Path:
    return Path(memeng.__file__).resolve().parents[2]


def register_hooks(settings_path: Path, checkout: Path) -> list[str]:
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    hooks = settings.setdefault("hooks", {})
    added = []
    for event, module in HOOK_MODULES.items():
        entries = hooks.setdefault(event, [])
        commands = [
            h.get("command", "") for entry in entries for h in entry.get("hooks", [])
        ]
        if any(module in command for command in commands):
            continue
        entries.append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"uv run --directory {checkout} python -m {module}",
                    }
                ]
            }
        )
        added.append(event)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return added


def write_project_alias(root: Path, project_id: str | None) -> str:
    alias_file = root / ".memeng.toml"
    if alias_file.exists():
        return tomllib.loads(alias_file.read_text())["project_id"]
    name = project_id or root.name
    alias_file.write_text(f'project_id = "{name}"\n')
    return name


def pick_project_id(default: str, stdin: TextIO) -> str:
    known = get_project_ids(CONFIG.user_id)
    if known:
        print("project_id values already in the store:")
    for number, project_id in enumerate(known, start=1):
        print(f"  {number}. {project_id}")
    while True:
        print(
            f"Pick a number, type a new project_id, or press Enter for {default}: ",
            end="",
            flush=True,
        )
        answer = stdin.readline().strip()
        if not answer.isdecimal():
            return answer or default
        if 1 <= int(answer) <= len(known):
            return known[int(answer) - 1]
        print(f"no project numbered {answer}")


def run_checks(
    root: Path,
    settings_path: Path,
    *,
    check_ollama: Callable[[], tuple[bool, str]] = embed_module.check_ollama,
) -> list[tuple[str, bool, str]]:
    ok, message = check_ollama()
    checks = [("ollama", ok, message)]

    has_model = importlib.util.find_spec("en_core_web_sm") is not None
    checks.append(
        (
            "spacy-model",
            has_model,
            "en_core_web_sm importable"
            if has_model
            else "en_core_web_sm missing; run: uv sync",
        )
    )

    store = CONFIG.db_path.expanduser()
    checks.append(
        (
            "store",
            store.exists(),
            f"store at {store}" if store.exists() else f"{store} missing; run: memeng setup",
        )
    )

    project = resolve_project_id(root)
    checks.append(
        (
            "project-alias",
            project is not None,
            f"project_id {project.value} from {project.path}"
            if project
            else f"{root / '.memeng.toml'} missing; this directory is not enabled,"
            " run: memeng setup",
        )
    )

    missing = [event for event in HOOK_MODULES if event not in _registered_events(settings_path)]
    checks.append(
        (
            "hooks",
            not missing,
            f"missing in {settings_path}: {', '.join(missing)}"
            if missing
            else f"hooks registered in {settings_path}",
        )
    )
    return checks


def _registered_events(settings_path: Path) -> set[str]:
    if not settings_path.exists():
        return set()
    hooks = json.loads(settings_path.read_text()).get("hooks", {})
    registered = set()
    for event, module in HOOK_MODULES.items():
        commands = [
            h.get("command", "")
            for entry in hooks.get(event, [])
            for h in entry.get("hooks", [])
        ]
        if any(module in command for command in commands):
            registered.add(event)
    return registered


def reembed(
    *,
    force: bool,
    embed: Callable[[list[str]], list[list[float]]],
    embedding_dim: Callable[[], int],
) -> int:
    facts = get_all_facts()
    if not facts:
        print("nothing to re-embed")
        return 0

    new_dim = embedding_dim()
    current = DB.conn.meta_get("embed_dim")
    if current is not None and int(current) != new_dim and not force:
        print(
            f"stored vectors are {current} wide, {CONFIG.embed_model} returns"
            f" {new_dim}; rerun with --force to recreate the table"
        )
        return 1

    texts = [searchable_text(fact.fact, fact.rationale) for fact in facts]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        vectors.extend(embed(texts[start : start + 64]))
    replace_all_vectors({fact.id: vector for fact, vector in zip(facts, vectors)})
    print(f"re-embedded {len(facts)} facts with {CONFIG.embed_model}")
    return 0


def judge(
    target: VerdictTarget, target_id: str, verdict: str, note: str | None, by: str
) -> int:
    insert_verdict(
        Verdict(
            target=target,
            target_id=target_id,
            verdict=verdict,
            judged_by=by,
            note=note,
        )
    )
    print(f"{target} {target_id}: {verdict} (by {by})")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    root: Path | None = None,
    stdin: TextIO | None = None,
    check_ollama: Callable[[], tuple[bool, str]] = embed_module.check_ollama,
    embed: Callable[[list[str]], list[list[float]]] = embed_module.embed,
    embedding_dim: Callable[[], int] = embed_module.embedding_dim,
    load_config: Callable[[], Config] = Config.load,
) -> int:
    args = _parser().parse_args(argv)
    given_root = getattr(args, "root", None)
    root = root or (Path(given_root).expanduser() if given_root else git_root())
    settings_path = (
        Path(args.settings).expanduser()
        if getattr(args, "settings", None)
        else root / ".claude" / "settings.json"
    )

    config = load_config()
    if args.command == "doctor":
        CONFIG.bind(config)
    else:
        DB.open(config)
    if args.command == "reembed":
        return reembed(force=args.force, embed=embed, embedding_dim=embedding_dim)
    if args.command == "judge":
        return judge(
            VerdictTarget(args.target),
            args.target_id,
            args.verdict,
            args.note,
            args.by or CONFIG.user_id,
        )

    checks = run_checks(root, settings_path, check_ollama=check_ollama)
    for name, ok, message in checks:
        print(f"{'ok' if ok else 'missing'} {name}: {message}")

    if args.command == "doctor":
        return 0 if all(ok for _, ok, _ in checks) else 1

    stdin = stdin or sys.stdin
    project_id = args.project_id
    if project_id is None and resolve_project_id(root) is None and stdin.isatty():
        project_id = pick_project_id(root.name, stdin)
    alias = write_project_alias(root, project_id)
    if args.project_id and args.project_id != alias:
        print(f"--project-id {args.project_id} ignored; {root / '.memeng.toml'} already says {alias}")
    print(f"project_id: {alias}")
    for event in register_hooks(settings_path, memeng_root()):
        print(f"registered {event} hook in {settings_path}")
    return 0


def git_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        return Path.cwd()
    return Path(completed.stdout.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memeng")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "setup",
        help="create the store, write .memeng.toml, register the hooks in this repository",
    )
    setup.add_argument("--project-id")
    setup.add_argument(
        "--root",
        help="repository to set up; default the git root of the working directory."
        " From another repository: uv run --project /abs/memeng memeng setup",
    )
    setup.add_argument(
        "--settings",
        help="settings file to register hooks in; default <git root>/.claude/settings.json",
    )
    doctor = commands.add_parser("doctor", help="run the setup checks without writing")
    doctor.add_argument("--root")
    doctor.add_argument("--settings")
    rebuild = commands.add_parser(
        "reembed", help="rebuild every vector with the configured embedding model"
    )
    rebuild.add_argument("--force", action="store_true")
    verdict = commands.add_parser(
        "judge", help="record whether a logged fact, capture or hit was worth it"
    )
    verdict.add_argument("target", choices=list(VerdictTarget))
    verdict.add_argument("target_id")
    verdict.add_argument("verdict")
    verdict.add_argument("--note")
    verdict.add_argument("--by")
    return parser
