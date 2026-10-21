import httpx

from memeng.config import CONFIG


class EmbedUnavailable(Exception):
    """Typed so the read path can fail open instead of blocking the prompt."""


def embed(
    texts: list[str],
    timeout_s: float = 2.0,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    with httpx.Client(
        base_url=CONFIG.ollama_url, timeout=timeout_s, transport=transport
    ) as client:
        try:
            response = client.post(
                "/api/embed", json={"model": CONFIG.embed_model, "input": texts}
            )
        except httpx.TimeoutException as timeout:
            raise EmbedUnavailable(f"ollama timed out after {timeout_s}s") from timeout
        except httpx.ConnectError as refused:
            raise EmbedUnavailable(
                f"ollama not running at {CONFIG.ollama_url}"
            ) from refused
        except httpx.RequestError as failure:
            raise EmbedUnavailable(f"ollama request failed: {failure}") from failure

    if response.status_code == 404 and CONFIG.embed_model in response.text:
        raise EmbedUnavailable(
            f"model {CONFIG.embed_model} not pulled; "
            f"run: ollama pull {CONFIG.embed_model}"
        )
    if not response.is_success:
        raise EmbedUnavailable(
            f"ollama returned {response.status_code}: {response.text[:200]}"
        )
    try:
        vectors = response.json()["embeddings"]
    except (ValueError, KeyError, TypeError) as unreadable:
        raise EmbedUnavailable(
            f"ollama returned an unreadable body: {response.text[:200]}"
        ) from unreadable
    if len(vectors) != len(texts):
        raise EmbedUnavailable(
            f"ollama returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors


def embedding_dim(*, transport: httpx.BaseTransport | None = None) -> int:
    return len(embed(["probe"], transport=transport)[0])


def check_ollama(*, transport: httpx.BaseTransport | None = None) -> tuple[bool, str]:
    try:
        with httpx.Client(base_url=CONFIG.ollama_url, transport=transport) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
    except httpx.HTTPError:
        return False, f"Ollama not running at {CONFIG.ollama_url}"

    served = [model["name"] for model in response.json()["models"]]
    if not any(name.startswith(CONFIG.embed_model) for name in served):
        return False, f"model missing; run: ollama pull {CONFIG.embed_model}"
    return True, f"ollama ok: {CONFIG.embed_model}"
