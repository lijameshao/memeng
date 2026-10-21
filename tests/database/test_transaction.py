import pytest

from memeng.database.utils import DB


def test_transaction_rolls_back_when_the_body_is_interrupted():
    with pytest.raises(KeyboardInterrupt):
        with DB.conn.transaction() as writer:
            writer.meta_set("probe", "uncommitted")
            raise KeyboardInterrupt

    assert DB.conn.meta_get("probe") is None
    assert not DB.conn.in_transaction
