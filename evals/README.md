# Evals

Four measurements. None runs in the test suite against a model; each has a
smoke test with fakes so `uv run pytest` needs no Ollama and no `claude`.

Prerequisites for a real run: `ollama pull embeddinggemma-2:270m` for the retrieval and
rationale evals, and a working `claude` command-line interface for the gate,
no-gate and rationale evals.

| Eval | Command | What it measures |
|---|---|---|
| Gate | `uv run python evals/gate/run.py --model sonnet` | false-negative and false-positive rate of the gate prompt |
| No gate | `uv run python -m evals.gate.no_gate --model sonnet` | how often the extractor invents a fact on a window with no decision, and how often it returns nothing on a window with one |
| Rationale | `uv run python evals/rationale/run.py` | whether each stored reason is present in the turns it came from |
| Retrieval | `uv run python evals/retrieval/run.py [--sweep semantic_threshold=0.3,0.4,0.5,0.6] [--explain]` | recall@10, precision@10, negative silence, temporal pairing |

Results land under each eval's `results/` directory as `<timestamp>.json` and
are committed, so a number can be traced to the code that produced it.

The retrieval sweep rebinds the configuration per value in one process and
reuses embeddings across values, so a four-value sweep embeds each text once.
