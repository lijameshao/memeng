import subprocess
import sys


def test_import_memeng_in_a_fresh_interpreter():
    subprocess.run(
        [sys.executable, "-c", "import memeng"], check=True, capture_output=True
    )
