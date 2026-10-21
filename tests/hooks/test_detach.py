import sys
import time

from memeng.hooks.detach import spawn_detached


def test_spawn_detached_returns_at_once_and_appends_the_child_output_to_the_log(tmp_path):
    log = tmp_path / "logs" / "capture.log"

    started = time.perf_counter()
    spawn_detached([sys.executable, "-c", "print('child ran')"], log)
    elapsed = time.perf_counter() - started

    deadline = time.time() + 5
    while "child ran" not in (log.read_text() if log.exists() else "") and time.time() < deadline:
        time.sleep(0.05)
    assert "child ran" in log.read_text()
    assert elapsed < 1.0
