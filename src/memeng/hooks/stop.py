import argparse
import fcntl
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from memeng.capture import add as add_facts
from memeng.config import CONFIG, Config
from memeng.database.utils import DB
from memeng.hooks.detach import spawn_detached
from memeng.models.common import Scope
from memeng.models.messages import Message
from memeng.project import resolve_project_id
from memeng.utils.common import now_iso

logger = logging.getLogger(__name__)


def read_new_turns(transcript_path: Path, since: str | None, scope: Scope) -> list[Message]:
    turns = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("type") not in ("user", "assistant"):
            continue
        if entry.get("isMeta") or entry.get("isSidechain"):
            continue
        body = entry.get("message") or {}
        content = _text_of(body.get("content"))
        if not content:
            continue
        turns.append(
            Message(
                role=body.get("role") or entry["type"],
                content=content,
                user_id=scope.user_id,
                project_id=scope.project_id,
                agent_id=scope.agent_id,
                run_id=scope.run_id,
                id=entry["uuid"],
                created_at=entry.get("timestamp") or now_iso(),
            )
        )

    seen = [turn.id for turn in turns]
    if since in seen:
        return turns[seen.index(since) + 1 :]
    return turns[-CONFIG.first_capture_turns :]


def capture(
    transcript_path: Path,
    session_id: str,
    cwd: Path,
    *,
    add: Callable[[list[Message], Scope], list] = add_facts,
) -> int:
    project = resolve_project_id(cwd)
    if project is None:
        return 0
    project_id = project.value
    scope = Scope(user_id=CONFIG.user_id, project_id=project_id, run_id=session_id)
    lock = _try_lock(project_id)
    if lock is None:
        logger.info("capture already running for %s", project_id)
        return 0
    try:
        key = f"last_uuid:{session_id}"
        turns = read_new_turns(transcript_path, DB.conn.meta_get(key), scope)
        if not turns:
            return 0
        add(turns, scope)
        DB.conn.meta_set(key, turns[-1].id)
        return 0
    finally:
        os.close(lock)


def _try_lock(project_id: str) -> int | None:
    name = project_id.replace(os.sep, "_")
    path = CONFIG.db_path.expanduser().parent / f"capture-{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _text_of(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if block.get("type") == "text"]
        return "\n".join(text for text in texts if text).strip()
    return ""


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    env: Mapping[str, str] | None = None,
    spawn: Callable[[list[str], Path], None] = spawn_detached,
    load_config: Callable[[], Config] = Config.load,
    add: Callable[[list[Message], Scope], list] = add_facts,
) -> int:
    env = os.environ if env is None else env
    if env.get("MEMENG_GATE"):
        return 0

    args = _parser().parse_args(argv)
    if args.capture:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        transcript_path, session_id, cwd = args.capture
        DB.open(load_config())
        return capture(Path(transcript_path), session_id, Path(cwd), add=add)

    try:
        payload = json.load(stdin or sys.stdin)
        if payload.get("stop_hook_active"):
            return 0
        config = load_config()
        spawn(
            [
                sys.executable, "-m", "memeng.hooks.stop", "--capture",
                payload["transcript_path"], payload["session_id"], payload["cwd"],
            ],
            config.db_path.expanduser().parent / "capture.log",
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memeng.hooks.stop")
    parser.add_argument("--capture", nargs=3, metavar=("TRANSCRIPT", "SESSION_ID", "CWD"))
    return parser


if __name__ == "__main__":
    sys.exit(main())
