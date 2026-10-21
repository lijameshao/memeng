from memeng.supersession.common import (
    insert_supersession,
    set_supersession_status,
    supersessions_for,
)


def test_insert_supersession_keeps_the_verdict_when_the_pair_is_proposed_again(
    superseding_pair,
):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")
    (row,) = supersessions_for([newer.id])
    set_supersession_status(row.id, "retained", "user")

    insert_supersession(newer.id, older.id, "cosine")

    (again,) = supersessions_for([newer.id])
    assert again.status == "retained"
