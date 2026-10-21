import hashlib
import io
from dataclasses import replace

import httpx
import pytest

from memeng.config import Config
from memeng.database.utils import DB
from memeng.enums.common import Category
from memeng.facts.common import insert_fact
from memeng.models.facts import StoredFact
from memeng.models.messages import Message


@pytest.fixture
def tmp_config(tmp_path):
    return Config(db_path=tmp_path / "store.db")


@pytest.fixture(autouse=True)
def open_db(tmp_config):
    DB.open(tmp_config)
    yield DB
    DB.close()


@pytest.fixture
def bind_config(tmp_config):
    def bind(**overrides):
        config = replace(tmp_config, **overrides)
        DB.open(config)
        return config

    return bind


@pytest.fixture
def row_count():
    def count(table):
        return DB.conn.fetchone(f"SELECT count(*) FROM {table}")[0]

    return count


@pytest.fixture
def make_fact():
    def build(fact="vectors live in sqlite-vec", **overrides):
        return StoredFact(
            **{
                "fact": fact,
                "rationale": "one file, no extra process",
                "category": Category.ARCHITECTURE,
                "text_lemmatized": fact,
                "hash": hashlib.md5(fact.encode()).hexdigest(),
                "user_id": "local",
                "project_id": "memeng",
                **overrides,
            }
        )

    return build


@pytest.fixture
def superseding_pair(make_fact, unit):
    older = make_fact("vectors in sqlite-vec", created_at="2026-09-01T09:00:00Z")
    newer = make_fact("vectors in qdrant", created_at="2026-09-02T09:00:00Z")
    insert_fact(older, unit(0), [], [])
    insert_fact(newer, unit(1), [], [])
    return older, newer


@pytest.fixture
def message():
    def build(content, **overrides):
        return Message(
            **{
                "role": "user",
                "content": content,
                "user_id": "local",
                "project_id": "memeng",
                **overrides,
            }
        )

    return build


@pytest.fixture
def unit():
    def build(index, dim=8):
        vector = [0.0] * dim
        vector[index] = 1.0
        return vector

    return build


@pytest.fixture
def embed_returning():
    def build(vector):
        return lambda texts: [vector for _ in texts]

    return build


@pytest.fixture
def transport_for():
    def build(failure):
        def handler(request):
            if isinstance(failure, httpx.Response):
                return failure
            raise failure

        return httpx.MockTransport(handler)

    return build


@pytest.fixture
def fake_embed():
    def embed(texts):
        return [
            [byte / 255 for byte in hashlib.md5(text.encode()).digest()[:8]]
            for text in texts
        ]

    return embed


@pytest.fixture
def scripted_run_prompt():
    def build(*answers):
        queue = list(answers)

        def run_prompt(prompt, model):
            run_prompt.calls.append((prompt, model))
            answer = queue.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        run_prompt.calls = []
        return run_prompt

    return build


@pytest.fixture
def fake_model_id():
    return lambda model: f"claude-{model}-5-5"


@pytest.fixture
def terminal():
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    return Terminal
