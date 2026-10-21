from memeng.messages.common import release_gated


def test_release_gated_with_no_ids_touches_nothing(message):
    assert release_gated([]) == 0
