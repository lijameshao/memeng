from memeng.facts.common import bm25, insert_fact


def test_bm25_is_scoped_to_the_project_and_the_user(make_fact, unit):
    ours = make_fact("vector search")
    insert_fact(ours, unit(0), [], [])
    insert_fact(make_fact("vector search", project_id="other"), unit(1), [], [])
    insert_fact(make_fact("vector search", user_id="ada"), unit(2), [], [])

    assert list(bm25(["vector"], "memeng", "local")) == [ours.id]
