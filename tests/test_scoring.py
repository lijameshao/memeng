
import pytest

from memeng.scoring import entity_boost, fuse, normalise_bm25


def test_normalise_bm25_never_leaves_the_unit_interval_on_extreme_input():
    for raw in (-1e6, -1000.0, 1000.0, 1e6):
        assert 0.0 <= normalise_bm25(raw, query_term_count=4) <= 1.0


@pytest.mark.parametrize(
    ("query_term_count", "raw", "expected"),
    [
        (3, 6.0, 0.6681877721681662),
        (4, 8.0, 0.6456563062257954),
        (6, 8.0, 0.6456563062257954),
        (7, 10.0, 0.6224593312018546),
        (9, 10.0, 0.6224593312018546),
        (10, 11.0, 0.6224593312018546),
        (15, 11.0, 0.6224593312018546),
        (16, 13.0, 0.6224593312018546),
    ],
)
def test_normalise_bm25_keeps_mem0s_midpoint_and_steepness_per_tier(
    query_term_count, raw, expected
):
    assert normalise_bm25(raw, query_term_count) == pytest.approx(expected)


def test_normalise_bm25_raises_the_midpoint_for_a_longer_query():
    short = normalise_bm25(8.0, query_term_count=2)
    long = normalise_bm25(8.0, query_term_count=20)

    assert short > 0.5 > long


def test_entity_boost_is_zero_when_the_query_named_no_entities():
    assert entity_boost(matched=0, total_query_entities=0, cap=0.5) == 0.0


@pytest.mark.parametrize(
    ("semantic", "bm25", "boost", "expected"),
    [
        (0.8, None, None, 0.8),
        (0.8, 0.9, 0.5, min(2.2 / 2.5, 1)),
        (0.8, 0.9, None, 1.7 / 2.0),
        (0.8, None, 0.5, 1.3 / 1.5),
        (0.8, 0.0, None, 0.8 / 2.0),
        (0.8, None, 0.0, 0.8 / 1.5),
    ],
    ids=[
        "semantic only",
        "every signal fired",
        "no entity signal",
        "no keyword signal",
        "keyword ran and missed",
        "entity ran and missed",
    ],
)
def test_fuse_divides_by_the_maximum_the_signals_that_ran_could_reach(
    semantic, bm25, boost, expected
):
    assert fuse(semantic, bm25, boost).final_score == pytest.approx(expected)


def test_fuse_caps_the_final_score_at_one_when_the_signals_beat_the_divisor():
    assert fuse(1.0, 1.0, 0.6).final_score == 1.0
