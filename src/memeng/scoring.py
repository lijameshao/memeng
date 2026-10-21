import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Breakdown:
    semantic_score: float
    bm25_score: float
    entity_boost: float
    final_score: float


def normalise_bm25(raw: float, query_term_count: int) -> float:
    """mem0 `mem0/utils/scoring.py` lines 31-40 and 54. The midpoint rises with
    term count because raw BM25 grows with query length, so a fixed midpoint
    would rank long and short queries on different scales.
    """
    if query_term_count <= 3:
        midpoint, steepness = 5.0, 0.7
    elif query_term_count <= 6:
        midpoint, steepness = 7.0, 0.6
    elif query_term_count <= 9:
        midpoint, steepness = 9.0, 0.5
    elif query_term_count <= 15:
        midpoint, steepness = 10.0, 0.5
    else:
        midpoint, steepness = 12.0, 0.5

    distance = steepness * (raw - midpoint)
    if distance >= 0:
        return 1.0 / (1.0 + math.exp(-distance))
    below = math.exp(distance)
    return below / (1.0 + below)


def entity_boost(matched: int, total_query_entities: int, cap: float) -> float:
    if total_query_entities == 0:
        return 0.0
    return cap * matched / total_query_entities


def fuse(
    semantic: float,
    bm25: float | None,
    boost: float | None,
    *,
    entity_cap: float = 0.5,
) -> Breakdown:
    """mem0 `mem0/utils/scoring.py` lines 97-101 and 118-119. `None` says the
    signal did not run and `0.0` that it ran and missed, so a query naming no
    entities is not penalised for having none. `entity_cap` must be `boost`'s cap.
    """
    max_possible = 1.0
    if bm25 is not None:
        max_possible += 1.0
    if boost is not None:
        max_possible += entity_cap

    raw = semantic + (bm25 or 0.0) + (boost or 0.0)
    return Breakdown(
        semantic_score=semantic,
        bm25_score=bm25 or 0.0,
        entity_boost=boost or 0.0,
        final_score=min(raw / max_possible, 1.0),
    )


def rank(candidates: dict[str, Breakdown], k: int) -> list[str]:
    return sorted(
        candidates, key=lambda memory_id: (-candidates[memory_id].final_score, memory_id)
    )[:k]
