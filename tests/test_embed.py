import httpx
import pytest

from memeng.embed import EmbedUnavailable, embed

BIG_BODY = "upstream exploded. " * 40


@pytest.fixture(autouse=True)
def ollama(bind_config):
    return bind_config(ollama_url="http://ollama.test:11434")


def test_embed_makes_no_request_when_there_is_nothing_to_embed():
    def handler(request):
        raise AssertionError("embed called ollama for an empty batch")

    assert embed([], transport=httpx.MockTransport(handler)) == []


@pytest.mark.parametrize(
    "failure, expected",
    [
        (
            httpx.ConnectError("[Errno 61] Connection refused"),
            "ollama not running at http://ollama.test:11434",
        ),
        (
            httpx.ReadTimeout("read timed out"),
            "ollama timed out after 0.5s",
        ),
        (
            httpx.ReadError("connection reset by peer"),
            "ollama request failed: connection reset by peer",
        ),
        (
            # An iterator body defers decoding to the read inside embed(); a bytes
            # body would decode, and raise, while this table is being built.
            httpx.Response(
                200, content=iter([b"not gzip"]), headers={"Content-Encoding": "gzip"}
            ),
            "ollama request failed: Error -3 while decompressing data: "
            "incorrect header check",
        ),
        (
            httpx.Response(
                404,
                json={
                    "error": 'model "embeddinggemma-2:270m" not found, try pulling it first'
                },
            ),
            "model embeddinggemma-2:270m not pulled; "
            "run: ollama pull embeddinggemma-2:270m",
        ),
        (
            httpx.Response(404, text="404 page not found"),
            "ollama returned 404: 404 page not found",
        ),
        (
            httpx.Response(500, text='model "embeddinggemma" failed to load'),
            'ollama returned 500: model "embeddinggemma" failed to load',
        ),
        (
            httpx.Response(500, text=BIG_BODY),
            f"ollama returned 500: {BIG_BODY[:200]}",
        ),
        (
            httpx.Response(200, text="<html>hello</html>"),
            "ollama returned an unreadable body: <html>hello</html>",
        ),
        (
            httpx.Response(200, json={"error": "something"}),
            'ollama returned an unreadable body: {"error":"something"}',
        ),
        (
            httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}),
            "ollama returned 1 vectors for 2 inputs",
        ),
    ],
)
def test_embed_reports_every_failure_as_embed_unavailable(
    failure, expected, transport_for
):
    with pytest.raises(EmbedUnavailable) as raised:
        embed(
            ["first", "second"],
            timeout_s=0.5,
            transport=transport_for(failure),
        )

    assert str(raised.value) == expected


def test_embed_bounds_the_request_with_the_given_timeout():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    embed(["fact"], timeout_s=0.5, transport=httpx.MockTransport(handler))

    assert requests[0].extensions["timeout"]["read"] == 0.5
