from pathlib import Path

import pytest

import memeng
from memeng.capture import add
from memeng.enums.common import Category
from memeng.facts.common import insert_fact
from memeng.models.common import Scope
from memeng.query import browse, dump, resolve
from memeng.retrieve import search
from memeng.supersession.common import insert_supersession, supersessions_for

SCOPE = Scope(user_id="local", project_id="memeng")


def test_browse_returns_the_scoped_rows_oldest_first(make_fact, unit):
    last = make_fact("retry budget per route", created_at="2026-09-03T10:00:02Z")
    first = make_fact("retry budget per caller", created_at="2026-09-03T10:00:00Z")
    middle = make_fact("retry budget per tenant", created_at="2026-09-03T10:00:01Z")
    other_project = make_fact("feature flags expire", project_id="other")
    other_user = make_fact("blue-green deploys by default", user_id="other")
    for index, fact in enumerate([last, first, middle, other_project, other_user]):
        insert_fact(fact, unit(index), [], [])

    assert browse(SCOPE) == [first, middle, last]


def test_browse_orders_equal_timestamps_by_id(make_fact, unit):
    second = make_fact(
        "circuit breaker per host", id="b", created_at="2026-09-03T10:00:00Z"
    )
    first = make_fact(
        "circuit breaker per route", id="a", created_at="2026-09-03T10:00:00Z"
    )
    insert_fact(second, unit(0), [], [])
    insert_fact(first, unit(1), [], [])

    assert [fact.id for fact in browse(SCOPE)] == ["a", "b"]


def test_browse_with_a_category_returns_only_that_category(make_fact, unit):
    architecture = make_fact("vectors live in sqlite-vec")
    process = make_fact("schema migrations run on deploy", category=Category.PROCESS)
    insert_fact(architecture, unit(0), [], [])
    insert_fact(process, unit(1), [], [])

    assert browse(SCOPE, category="process") == [process]


def test_dump_on_an_empty_scope_writes_nothing(tmp_path):
    out = tmp_path / "out"

    dump(SCOPE, out)

    assert not out.exists()


def test_dump_writes_one_file_per_category_in_the_section_template(
    make_fact, unit, tmp_path
):
    architecture = make_fact(
        "vectors live in sqlite-vec",
        source="review of the retry budget",
        created_at="2026-09-03T22:41:55Z",
    )
    process = make_fact(
        "schema migrations run on deploy",
        category=Category.PROCESS,
        created_at="2026-09-03T22:41:55Z",
    )
    insert_fact(architecture, unit(0), [], [])
    insert_fact(process, unit(1), [], [])
    out = tmp_path / "out"

    dump(SCOPE, out)

    assert sorted(path.name for path in out.iterdir()) == [
        "architecture.md",
        "process.md",
    ]
    assert (out / "architecture.md").read_text() == (
        "## vectors live in sqlite-vec\n"
        "\n"
        "one file, no extra process\n"
        "\n"
        "- source: review of the retry budget\n"
        "- decided_at: 2026-09-03\n"
    )
    assert (out / "process.md").read_text() == (
        "## schema migrations run on deploy\n"
        "\n"
        "one file, no extra process\n"
        "\n"
        "- source: -\n"
        "- decided_at: 2026-09-03\n"
    )


def test_dump_prints_the_partner_fact_and_status_on_both_sides(
    superseding_pair, tmp_path
):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")
    (row,) = supersessions_for([newer.id])
    resolve(row.id, "superseded", "agent")
    out = tmp_path / "out"

    dump(SCOPE, out)

    assert (out / "architecture.md").read_text() == (
        "## vectors in sqlite-vec\n"
        "\n"
        "one file, no extra process\n"
        "\n"
        "- source: -\n"
        "- decided_at: 2026-09-01\n"
        "- superseded_by: vectors in qdrant (superseded)\n"
        "\n"
        "## vectors in qdrant\n"
        "\n"
        "one file, no extra process\n"
        "\n"
        "- source: -\n"
        "- decided_at: 2026-09-02\n"
        "- supersedes: vectors in sqlite-vec (superseded)\n"
    )


def test_dump_names_an_out_of_scope_partner_by_id(make_fact, unit, tmp_path):
    older = make_fact("vectors in sqlite-vec", created_at="2026-09-01T09:00:00Z")
    newer = make_fact("vectors in qdrant", project_id="other")
    insert_fact(older, unit(0), [], [])
    insert_fact(newer, unit(1), [], [])
    insert_supersession(newer.id, older.id, "extractor")
    out = tmp_path / "out"

    dump(SCOPE, out)

    assert (out / "architecture.md").read_text() == (
        "## vectors in sqlite-vec\n"
        "\n"
        "one file, no extra process\n"
        "\n"
        "- source: -\n"
        "- decided_at: 2026-09-01\n"
        f"- superseded_by: {newer.id} (unresolved)\n"
    )


def test_dump_twice_produces_byte_identical_files(
    superseding_pair, make_fact, unit, tmp_path
):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")
    process = make_fact("schema migrations run on deploy", category=Category.PROCESS)
    insert_fact(process, unit(2), [], [])
    out = tmp_path / "out"

    dump(SCOPE, out)
    first = {path.name: path.read_bytes() for path in out.iterdir()}
    dump(SCOPE, out)

    assert len(first) == 2
    assert {path.name: path.read_bytes() for path in out.iterdir()} == first


def test_resolve_records_the_verdict_and_resolved_at(superseding_pair, make_fact, unit):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")
    (row,) = supersessions_for([newer.id])
    per_caller = make_fact("retry budget per caller")
    per_route = make_fact("retry budget per route")
    insert_fact(per_caller, unit(2), [], [])
    insert_fact(per_route, unit(3), [], [])
    insert_supersession(per_route.id, per_caller.id, "extractor")

    resolve(row.id, "retained", "user")

    (resolved,) = supersessions_for([newer.id])
    assert (resolved.status, resolved.resolved_by) == ("retained", "user")
    assert resolved.resolved_at is not None
    (untouched,) = supersessions_for([per_route.id])
    assert (untouched.status, untouched.resolved_by) == ("unresolved", None)


def test_resolve_rejects_an_unknown_id():
    with pytest.raises(ValueError, match="unknown supersession id"):
        resolve("no-such-id", "superseded", "agent")


def test_resolve_rejects_an_unknown_status_and_leaves_the_row_unresolved(
    superseding_pair,
):
    older, newer = superseding_pair
    insert_supersession(newer.id, older.id, "extractor")
    (row,) = supersessions_for([newer.id])

    with pytest.raises(ValueError, match="unresolved"):
        resolve(row.id, "unresolved", "agent")

    assert supersessions_for([newer.id])[0].status == "unresolved"


def test_query_makes_no_embedding_or_model_call():
    source = Path(memeng.query.__file__).read_text()

    assert "memeng.embed" not in source
    assert "memeng.llm" not in source
    assert "httpx" not in source
    assert "subprocess" not in source


def test_the_package_exports_the_query_entry_points():
    assert (memeng.add, memeng.search, memeng.browse, memeng.dump, memeng.resolve) == (
        add,
        search,
        browse,
        dump,
        resolve,
    )
