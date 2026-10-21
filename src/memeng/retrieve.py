import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from memeng.config import CONFIG
from memeng.embed import EmbedUnavailable
from memeng.embed import embed as embed_texts
from memeng.enums.common import ConflictSource, SupersessionStatus
from memeng.facts.common import (
    bm25,
    facts_by_ids,
    get_entity_memory_ids,
    get_entity_vocab,
    knn,
    vectors_for,
)
from memeng.injections.common import injected_ids
from memeng.models.common import Scope
from memeng.models.facts import StoredFact
from memeng.models.supersession import SupersessionRow
from memeng.scoring import Breakdown, entity_boost, fuse, normalise_bm25, rank
from memeng.supersession.common import supersessions_for
from memeng.text import lemmatise, match_entities

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedFact:
    fact: StoredFact
    score: float
    explain: Breakdown | None
    superseded_by: str | None
    supersedes: str | None
    supersession_status: SupersessionStatus | None


@dataclass(frozen=True)
class Conflict:
    newer_id: str
    older_id: str
    cosine: float
    source: ConflictSource


@dataclass(frozen=True)
class SearchResult:
    memories: list[RankedFact]
    conflicts: list[Conflict]
    excluded_as_injected: list[str]
    scored: dict[str, Breakdown] = field(default_factory=dict)
    bm25_ran: bool = False
    entity_ran: bool = False


def search(
    query: str,
    scope: Scope,
    k: int | None = None,
    explain: bool = False,
    *,
    embed=embed_texts,
) -> SearchResult:
    k = CONFIG.top_k if k is None else k
    try:
        [query_vector] = embed([query])
    except EmbedUnavailable as unavailable:
        logger.warning("search skipped, embedding unavailable: %s", unavailable)
        return SearchResult(memories=[], conflicts=[], excluded_as_injected=[])

    hits = knn(
        query_vector, scope.project_id, scope.user_id, max(k * CONFIG.overfetch, 60)
    )
    semantic = {
        memory_id: cosine
        for memory_id, cosine in hits
        if cosine >= CONFIG.semantic_threshold
    }
    if not semantic:
        return SearchResult(memories=[], conflicts=[], excluded_as_injected=[])

    scored, bm25_ran, entity_ran = _breakdowns(query, semantic, scope)
    top_ids = rank(scored, k)

    facts = {fact.id: fact for fact in facts_by_ids(top_ids)}
    rows = supersessions_for(top_ids)
    conflicts = _conflicts(top_ids, facts, vectors_for(top_ids), rows)

    ordered = list(top_ids)
    for conflict in conflicts:
        ordered.remove(conflict.older_id)
        ordered.insert(ordered.index(conflict.newer_id) + 1, conflict.older_id)

    superseded_by = {
        row.supersedes: row.memory_id
        for row in rows
        if row.status == SupersessionStatus.SUPERSEDED
    }
    supersedes = {
        row.memory_id: row.supersedes
        for row in rows
        if row.status == SupersessionStatus.UNRESOLVED
    }
    injected = injected_ids(scope.run_id) if scope.run_id else set()

    memories = []
    for memory_id in ordered:
        if memory_id in injected:
            continue
        replacement = superseded_by.get(memory_id)
        replaced = supersedes.get(memory_id)
        memories.append(
            RankedFact(
                fact=facts[memory_id],
                score=scored[memory_id].final_score,
                explain=scored[memory_id] if explain else None,
                superseded_by=replacement,
                supersedes=replaced,
                supersession_status=(
                    SupersessionStatus.SUPERSEDED
                    if replacement
                    else SupersessionStatus.UNRESOLVED
                    if replaced
                    else None
                ),
            )
        )
    return SearchResult(
        memories=memories,
        conflicts=conflicts,
        excluded_as_injected=sorted(injected & set(top_ids)),
        scored={memory_id: scored[memory_id] for memory_id in top_ids},
        bm25_ran=bm25_ran,
        entity_ran=entity_ran,
    )


def _breakdowns(
    query: str, semantic: dict[str, float], scope: Scope
) -> tuple[dict[str, Breakdown], bool, bool]:
    terms = lemmatise(query)
    keyword = bm25(terms, scope.project_id, scope.user_id)
    query_entities = match_entities(
        query, get_entity_vocab(scope.project_id, scope.user_id)
    )
    linked = get_entity_memory_ids(query_entities, scope.project_id, scope.user_id)

    breakdowns = {}
    for memory_id, cosine in semantic.items():
        keyword_score = None
        if terms:
            raw = keyword.get(memory_id)
            keyword_score = 0.0 if raw is None else normalise_bm25(raw, len(terms))
        boost = None
        if query_entities:
            matched = sum(
                1 for entity in query_entities if memory_id in linked.get(entity, ())
            )
            boost = entity_boost(matched, len(query_entities), CONFIG.entity_boost_cap)
        breakdowns[memory_id] = fuse(
            cosine, keyword_score, boost, entity_cap=CONFIG.entity_boost_cap
        )
    return breakdowns, bool(terms), bool(query_entities)


def _conflicts(
    top_ids: list[str],
    facts: dict[str, StoredFact],
    vectors: dict[str, list[float]],
    rows: list[SupersessionRow],
) -> list[Conflict]:
    # A pair can hold a row in each direction; the later row is the standing verdict.
    by_pair = {frozenset((row.memory_id, row.supersedes)): row for row in rows}
    matrix = np.array([vectors[memory_id] for memory_id in top_ids], dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = matrix @ matrix.T

    conflicts = []
    for first, second in combinations(range(len(top_ids)), 2):
        pair = (top_ids[first], top_ids[second])
        cosine = float(similarity[first, second])
        row = by_pair.get(frozenset(pair))
        if row is not None:
            if row.status == SupersessionStatus.UNRESOLVED:
                conflicts.append(
                    Conflict(
                        row.memory_id,
                        row.supersedes,
                        cosine,
                        ConflictSource.EXTRACTOR,
                    )
                )
            continue
        if cosine < CONFIG.conflict_threshold:
            continue
        if facts[pair[0]].fact == facts[pair[1]].fact:
            continue
        newer, older = sorted(
            pair,
            key=lambda memory_id: (facts[memory_id].created_at, memory_id),
            reverse=True,
        )
        conflicts.append(Conflict(newer, older, cosine, ConflictSource.COSINE))
    return conflicts
