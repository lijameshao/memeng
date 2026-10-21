
import pytest

from memeng import text


@pytest.mark.parametrize(
    "content, vocabulary, expected",
    [
        ("the rate limiter changed", ["rate limiter", "rate"], ["rate limiter"]),
        ("the rate limiter changed", ["rate", "rate limiter"], ["rate limiter"]),
        ("the rate\nlimiter changed", ["rate limiter"], ["rate limiter"]),
        ("CQRS replaced the cache; cqrs again", ["Cache", "cqrs"], ["cqrs", "cache"]),
        ("the caches were purged", ["cache"], []),
        ("the rate limiter changed", [], []),
    ],
)
def test_match_entities_returns_the_normalised_terms_found_in_the_text(
    content, vocabulary, expected
):
    assert text.match_entities(content, vocabulary) == expected


def test_the_matcher_is_built_once_per_vocabulary():
    matcher = text._matcher(("rate limiter",))

    assert text._matcher(("rate limiter",)) is matcher
    assert text._matcher(("circuit breaker",)) is not matcher
