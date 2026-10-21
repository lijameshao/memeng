from dataclasses import dataclass

from memeng.config import CONFIG
from memeng.models.messages import Message

MARKER = "[... {n} messages truncated ...]"


@dataclass(frozen=True)
class Window:
    messages: list[Message]
    text: str
    truncated: bool


def tokens(s: str) -> int:
    return len(s) // 4


def render(messages: list[Message]) -> str:
    return "\n\n".join(_lines(messages))


def build_window(messages: list[Message], token_cap: int) -> Window:
    """`messages` is always the whole input: truncation reaches `text` only,
    because the caller still has to mark every message processed.
    """
    prefix = [m for m in messages if m.gate_count > 0]
    fresh = [m for m in messages if m.gate_count == 0]
    if not prefix:
        return _fit(messages, token_cap)
    if not fresh:
        return _fit(prefix, token_cap)

    fresh_window = _fit(fresh, token_cap)
    share = int(token_cap * CONFIG.gated_prefix_share)
    kept = list(prefix)
    while kept and (
        tokens(render(kept)) > share
        or tokens("\n\n".join((render(kept), fresh_window.text))) > token_cap
    ):
        kept.pop(0)
    text = "\n\n".join(part for part in (render(kept), fresh_window.text) if part)
    return Window(
        messages=messages,
        text=text,
        truncated=fresh_window.truncated or len(kept) < len(prefix),
    )


def _fit(messages: list[Message], token_cap: int) -> Window:
    if token_cap <= 0:
        raise ValueError(f"token_cap must be positive, not {token_cap}")

    text = render(messages)
    if tokens(text) <= token_cap:
        return Window(messages=messages, text=text, truncated=False)

    lines = _lines(messages)
    head, tail = _fit_head_and_tail(lines, token_cap)
    if head or tail:
        text = _joined(lines, head, tail)
    else:
        text = _hard_cut(lines[-1], token_cap, len(messages))
    return Window(messages=messages, text=text, truncated=True)


def _fit_head_and_tail(lines: list[str], token_cap: int) -> tuple[int, int]:
    head = tail = refused = 0
    from_head = True
    while head + tail < len(lines) and refused < 2:
        step_head = head + (1 if from_head else 0)
        step_tail = tail + (0 if from_head else 1)
        if tokens(_joined(lines, step_head, step_tail)) <= token_cap:
            head, tail, refused = step_head, step_tail, 0
        else:
            refused += 1
        from_head = not from_head
    return head, tail


def _joined(lines: list[str], head: int, tail: int) -> str:
    marker = MARKER.format(n=len(lines) - head - tail)
    return "\n\n".join([*lines[:head], marker, *lines[len(lines) - tail :]])


def _hard_cut(line: str, token_cap: int, truncated_count: int) -> str:
    marker = MARKER.format(n=truncated_count)
    budget = token_cap * 4 - len(marker) - 4
    if budget <= 0:
        return line[: token_cap * 4]
    parts = [line[: (budget + 1) // 2], marker, line[len(line) - budget // 2 :]]
    return "\n\n".join(part for part in parts if part)


def _lines(messages: list[Message]) -> list[str]:
    return [f"{m.role}: {m.content}" for m in messages]
