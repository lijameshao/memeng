import random

import pytest

from memeng.database.utils import DB
from memeng.window import Window, build_window, render, tokens


def test_tokens_counts_four_characters_each():
    assert tokens("x" * 40) == 10


def test_build_window_truncates_only_once_the_render_passes_the_cap(message):
    msgs = [message("a" * 9), message("b" * 9)]
    at_the_cap = tokens(render(msgs))

    fits = build_window(msgs, at_the_cap)
    one_over = build_window(msgs, at_the_cap - 1)

    assert fits == Window(messages=msgs, text=render(msgs), truncated=False)
    assert one_over.truncated is True
    assert tokens(one_over.text) <= at_the_cap - 1


def test_build_window_over_the_cap_keeps_the_head_and_the_tail(message):
    msgs = [message(f"turn {i} " + "detail " * 5) for i in range(5)]

    window = build_window(msgs, token_cap=40)

    assert window.text.count("[... 3 messages truncated ...]") == 1
    head, _, tail = window.text.partition("[... 3 messages truncated ...]")
    assert head == f"user: {msgs[0].content}\n\n"
    assert tail == f"\n\nuser: {msgs[4].content}"
    assert tokens(window.text) <= 40


def test_build_window_hard_cuts_a_single_oversized_message(message):
    msgs = [message("START" + "x" * 400 + "END")]

    window = build_window(msgs, token_cap=20)

    assert window.text.startswith("user: START")
    assert window.text.endswith("END")
    assert "[... 1 messages truncated ...]" in window.text
    assert tokens(window.text) <= 20
    assert window.truncated is True
    assert window.messages == msgs


def test_build_window_cuts_to_the_cap_when_even_the_marker_does_not_fit(message):
    msgs = [message("x" * 400), message("the conclusion runs on and on")]

    window = build_window(msgs, token_cap=6)

    assert window.text == "user: the conclusion run"
    assert window.truncated is True


def test_build_window_keeps_a_short_message_beside_an_oversized_one(message):
    msgs = [message("x" * 400), message("the conclusion")]

    window = build_window(msgs, token_cap=20)

    assert window.text == "[... 1 messages truncated ...]\n\nuser: the conclusion"


@pytest.mark.parametrize("token_cap", [1, 5, 12, 40, 300])
def test_build_window_never_exceeds_the_cap_or_repeats_a_message(message, token_cap):
    rng = random.Random(0)

    for _ in range(100):
        msgs = [
            message(
                f"m{i} " + "detail " * rng.randint(0, 20),
                role=rng.choice(["user", "assistant"]),
            )
            for i in range(rng.randint(0, 8))
        ]

        window = build_window(msgs, token_cap)

        assert tokens(window.text) <= token_cap
        assert all(window.text.count(f"m{i} ") <= 1 for i in range(len(msgs)))


def test_build_window_reads_no_database(message):
    DB.close()

    window = build_window([message("we picked sqlite-vec")], token_cap=100)

    assert window.text == "user: we picked sqlite-vec"


def test_build_window_of_no_messages_is_empty_and_untruncated():
    assert build_window([], token_cap=100) == Window(
        messages=[], text="", truncated=False
    )


def test_build_window_cuts_the_gated_prefix_from_the_front_and_keeps_fresh_turns_whole(
    message, bind_config
):
    bind_config(gated_prefix_share=0.5)
    old = [message(f"g{i} " + "x" * 36, gate_count=1) for i in range(3)]
    new = [message("n0 " + "y" * 36), message("n1 " + "y" * 36)]
    cap = tokens(render(new)) + 12

    window = build_window(old + new, cap)

    assert "n0" in window.text and "n1" in window.text
    assert "g2" in window.text
    assert "g0" not in window.text and "g1" not in window.text
    assert window.truncated is True
    assert window.messages == old + new
    assert tokens(window.text) <= cap


def test_build_window_gives_the_whole_cap_to_the_prefix_when_nothing_is_fresh(message):
    old = [
        message("g0 " + "x" * 36, gate_count=1),
        message("g1 " + "x" * 36, gate_count=2),
    ]

    window = build_window(old, tokens(render(old)))

    assert window.text == render(old)
    assert window.truncated is False


def test_build_window_drops_a_gated_turn_that_fits_its_share_but_tips_the_join_over(
    message, bind_config
):
    bind_config(gated_prefix_share=0.5)
    old = [message("g0 " + "x" * 34, gate_count=1)]
    new = [message("n0 " + "y" * 30)]
    cap = 20

    window = build_window(old + new, cap)

    assert tokens(render(old)) == int(cap * 0.5)
    assert tokens(render(old + new)) == cap + 1
    assert window.text == render(new)
    assert window.truncated is True
