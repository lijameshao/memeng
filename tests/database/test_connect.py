import pytest

from memeng.database.utils import DB


def test_reaching_for_a_handle_before_open_names_the_call_that_is_missing():
    DB.close()

    with pytest.raises(RuntimeError, match=r"no open database handle"):
        DB.conn
