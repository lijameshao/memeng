import sqlite3

import pytest

from memeng.database.utils import DB
from memeng.facts.common import get_hashes, insert_fact


def test_insert_fact_writes_nothing_when_one_insert_fails(make_fact, unit, row_count):
    with pytest.raises(sqlite3.IntegrityError):
        insert_fact(make_fact(), unit(0), ["sqlite-vec"], ["no-such-message"])

    assert row_count("memories") == 0
    assert row_count("memories_fts") == 0
    assert row_count("entities") == 0
    assert row_count("fact_sources") == 0
    assert DB.conn.meta_get("embed_dim") is None


def test_the_vector_dimension_comes_from_the_first_stored_vector(make_fact, unit):
    insert_fact(make_fact(), unit(0, dim=8), [], [])

    assert DB.conn.meta_get("embed_dim") == "8"
    with pytest.raises(sqlite3.OperationalError):
        insert_fact(make_fact("shorter"), unit(0, dim=4), [], [])


def test_get_hashes_asks_nothing_of_the_store_for_an_empty_list():
    assert get_hashes([], "memeng", "local") == set()
