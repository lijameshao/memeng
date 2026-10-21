import hashlib
import logging
import uuid
from collections.abc import Callable

from memeng import embed, llm
from memeng.config import CONFIG
from memeng.database.utils import DB
from memeng.embed import EmbedUnavailable
from memeng.enums.common import CaptureOutcome, ConflictSource
from memeng.extract import (
    parse_extraction,
    parse_gate,
    render_extract_prompt,
    render_gate_prompt,
)
from memeng.facts.common import (
    facts_by_ids,
    get_hashes,
    insert_fact,
    knn,
)
from memeng.facts.searchable_text import searchable_text
from memeng.llm import LLMError
from memeng.logs.common import insert_capture_log
from memeng.messages.common import (
    clear_processed,
    get_unprocessed_messages,
    insert_messages,
    mark_processed,
    release_gated,
)
from memeng.models.common import Scope
from memeng.models.facts import ExtractedFact, StoredFact
from memeng.models.logs import CaptureLog
from memeng.models.messages import Message
from memeng.supersession.common import insert_supersession
from memeng.text import lemmatise
from memeng.window import Window, build_window

logger = logging.getLogger(__name__)


def add(
    messages: list[Message],
    scope: Scope,
    *,
    run_prompt: Callable[[str, str], str] = llm.run_prompt,
    embed: Callable[[list[str]], list[list[float]]] = embed.embed,
) -> list[StoredFact]:
    window_messages = _claim_messages_to_extract(messages, scope)
    if not window_messages:
        return []

    window = build_window(window_messages, CONFIG.window_token_cap)
    capture_id = str(uuid.uuid4()) if CONFIG.record_decisions else None
    outcome, gate_answer, stored = _run_capture(
        window, capture_id, scope, run_prompt=run_prompt, embed=embed
    )
    if CONFIG.record_decisions:
        insert_capture_log(
            CaptureLog(
                id=capture_id,
                run_id=scope.run_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
                window_text=window.text,
                truncated=window.truncated,
                message_count=len(window.messages),
                gate_answer=gate_answer,
                outcome=outcome,
                stored_count=len(stored),
            )
        )
    return stored


def _run_capture(
    window: Window,
    capture_id: str | None,
    scope: Scope,
    *,
    run_prompt: Callable[[str, str], str],
    embed: Callable[[list[str]], list[list[float]]],
) -> tuple[CaptureOutcome, str | None, list[StoredFact]]:
    window_ids = [message.id for message in window.messages]
    gate_answer = None
    try:
        gate_answer = run_prompt(render_gate_prompt(window.text), CONFIG.gate_model)
        if not parse_gate(gate_answer):
            retired = release_gated(window_ids)
            if retired:
                logger.info(
                    "retired %d messages after %d gate misses",
                    retired,
                    CONFIG.gate_retire_after,
                )
            return CaptureOutcome.GATE_NO, gate_answer, []

        similar_facts = _get_similar_facts(window.text, scope, embed)
        extraction = run_prompt(
            render_extract_prompt(window.text, similar_facts),
            CONFIG.extract_model,
        )
        extracted = parse_extraction(extraction, len(similar_facts))
        if not extracted:
            retired = release_gated(window_ids)
            if retired:
                logger.info(
                    "retired %d messages after %d gate misses",
                    retired,
                    CONFIG.gate_retire_after,
                )
            return CaptureOutcome.NOTHING_EXTRACTED, gate_answer, []

        unseen = _get_unseen_facts(extracted, scope)
        if not unseen:
            return CaptureOutcome.ALL_DUPLICATE, gate_answer, []
        stored = _store_facts(
            unseen, similar_facts, window_ids, scope, capture_id, embed
        )
        return CaptureOutcome.STORED, gate_answer, stored
    except (LLMError, EmbedUnavailable) as failure:
        clear_processed(window_ids)
        logger.warning(
            "capture released %d messages after a failure: %s", len(window_ids), failure
        )
        return CaptureOutcome.FAILED, gate_answer, []


def _claim_messages_to_extract(messages: list[Message], scope: Scope) -> list[Message]:
    """The write lock spans the insert, the read and the mark, so a capture that
    starts while this one runs finds nothing unprocessed and extracts nothing twice.
    """
    with DB.conn.transaction():
        insert_messages(messages)
        window_messages = get_unprocessed_messages(scope.project_id, scope.user_id)
        mark_processed([message.id for message in window_messages])
    return window_messages


def _get_similar_facts(
    window_text: str,
    scope: Scope,
    embed: Callable[[list[str]], list[list[float]]],
) -> list[StoredFact]:
    [vector] = embed([window_text])
    hits = knn(vector, scope.project_id, scope.user_id, CONFIG.top_k)
    return facts_by_ids([memory_id for memory_id, _ in hits])


def _get_unseen_facts(
    extracted: list[ExtractedFact], scope: Scope
) -> list[ExtractedFact]:
    digests = [hashlib.md5(fact.fact.encode()).hexdigest() for fact in extracted]
    seen = get_hashes(digests, scope.project_id, scope.user_id)
    unseen = []
    for fact, digest in zip(extracted, digests):
        if digest not in seen:
            unseen.append(fact)
            seen.add(digest)
    logger.info(
        "dedupe: %d of %d facts skipped by hash",
        len(extracted) - len(unseen),
        len(extracted),
    )
    return unseen


def _store_facts(
    unseen: list[ExtractedFact],
    similar_facts: list[StoredFact],
    window_ids: list[str],
    scope: Scope,
    capture_id: str | None,
    embed: Callable[[list[str]], list[list[float]]],
) -> list[StoredFact]:
    vectors = embed([searchable_text(fact.fact, fact.rationale) for fact in unseen])
    stored = []
    for extracted, vector in zip(unseen, vectors):
        fact = StoredFact(
            fact=extracted.fact,
            rationale=extracted.rationale,
            category=extracted.category,
            text_lemmatized=" ".join(
                lemmatise(searchable_text(extracted.fact, extracted.rationale))
            ),
            hash=hashlib.md5(extracted.fact.encode()).hexdigest(),
            source=f"session {scope.run_id}" if scope.run_id else None,
            user_id=scope.user_id,
            project_id=scope.project_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            capture_id=capture_id,
        )
        insert_fact(fact, vector, extracted.entities, window_ids)
        if extracted.supersedes is not None:
            insert_supersession(
                fact.id,
                similar_facts[int(extracted.supersedes)].id,
                ConflictSource.EXTRACTOR,
            )
        stored.append(fact)
    return stored
