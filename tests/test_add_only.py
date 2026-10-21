import re
from pathlib import Path

import memeng

PROTECTED = (
    "memories",
    "messages",
    "supersession",
    "injections",
    "fact_sources",
    "entities",
)


def test_no_primitive_deletes_a_row():
    """memories_vec and memories_fts are indexes rebuilt from memories; the
    add-only rule covers the rows they index, not the index itself.
    """
    package = Path(memeng.__file__).parent
    source = "\n".join(path.read_text() for path in package.rglob("*.py")).upper()

    assert "INSERT INTO" in source
    for table in PROTECTED:
        assert not re.search(
            rf"(DELETE FROM|DROP TABLE(?: IF EXISTS)?)\s+{table.upper()}\b", source
        ), table
