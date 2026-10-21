import json
import os
import subprocess


class LLMError(Exception):
    pass


class LLMTimeout(LLMError):
    pass


class LLMRateLimited(LLMError):
    pass


class LLMMalformed(LLMError):
    pass


def run_prompt(
    prompt: str, model: str, timeout_s: float = 120, *, claude_bin: str = "claude"
) -> str:
    return _envelope(prompt, model, timeout_s, claude_bin)["result"]


def model_id(model: str, timeout_s: float = 120, *, claude_bin: str = "claude") -> str:
    envelope = _envelope("Reply with one word: ok.", model, timeout_s, claude_bin)
    usage = envelope.get("modelUsage")
    if not isinstance(usage, dict) or len(usage) != 1:
        raise LLMMalformed(json.dumps(envelope)[:500])
    [resolved] = usage
    return resolved


def _envelope(prompt: str, model: str, timeout_s: float, claude_bin: str) -> dict:
    try:
        completed = subprocess.run(
            [
                claude_bin,
                "-p",
                "--model",
                model,
                "--output-format",
                "json",
                "--no-session-persistence",
                "--disallowed-tools",
                "*",
            ],
            # An extraction window runs to tens of thousands of characters, past
            # the argv limit.
            input=prompt,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            env={**os.environ, "MEMENG_GATE": "1"},
        )
    except subprocess.TimeoutExpired as expired:
        raise LLMTimeout(f"{claude_bin} exceeded {timeout_s}s") from expired
    except OSError as unusable:
        raise LLMError(f"could not run {claude_bin}: {unusable.strerror}") from unusable

    if completed.returncode != 0:
        raise _error_for(completed)

    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError as undecodable:
        raise LLMMalformed(completed.stdout[:500]) from undecodable

    if not isinstance(envelope, dict):
        raise LLMMalformed(completed.stdout[:500])
    if envelope.get("is_error"):
        raise _error_for(completed)
    if not isinstance(envelope.get("result"), str):
        raise LLMMalformed(completed.stdout[:500])
    return envelope


def _error_for(completed: subprocess.CompletedProcess) -> LLMError:
    reported = f"{completed.stderr}\n{completed.stdout}".strip()
    if "rate limit" in reported.lower() or "429" in reported:
        return LLMRateLimited(reported[:500])
    return LLMError(reported[:500])
