import functools

import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans


@functools.lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


def lemmatise(text: str) -> list[str]:
    return [
        token.lemma_
        for token in _nlp()(text.lower())
        if not (token.is_stop or token.is_punct or token.is_space)
    ]


def normalise_entity(term: str) -> str:
    return " ".join(term.lower().split())


@functools.lru_cache(maxsize=8)
def _matcher(vocabulary: tuple[str, ...]) -> PhraseMatcher:
    nlp = _nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("ENTITY", [nlp.make_doc(term) for term in vocabulary])
    return matcher


def match_entities(text: str, vocabulary: list[str]) -> list[str]:
    # Vocabulary terms are stored whitespace-collapsed, so a prompt that wraps
    # mid-term only matches when the text is collapsed to the same shape.
    doc = _nlp().make_doc(" ".join(text.split()))
    spans = _matcher(tuple(vocabulary))(doc, as_spans=True)
    return list(dict.fromkeys(normalise_entity(span.text) for span in filter_spans(spans)))
