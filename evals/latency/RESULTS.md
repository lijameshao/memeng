# Read-path latency

Apple M3 Pro, 36 GiB, macOS 26.6.2, Python 3.12, Ollama 0.33.3. Measured
2026-09-12 from the repository root with
`uv run python evals/latency/measure.py --runs 10`: median and p95 of 10 fresh
processes per step. The notes column is filled in by hand; the script leaves it
empty except for a skip reason.

| step | median ms | p95 ms | notes |
|---|---|---|---|
| python-start | 11.9 | 13.1 | bare venv interpreter, timed by the parent |
| uv-run-python-start | 101.0 | 130.7 | what the hook command pays before any import, timed by the parent |
| spacy-load-cold | 611.6 | 881.9 | `import spacy` plus `spacy.load(...)`, timed by the parent, so it includes python-start |
| lemmatise-cold | 535.8 | 660.6 | import, load and the first `lemmatise` of the 12-word query, timed in the child |
| lemmatise-after-load | 1.9 | 2.2 | a second `lemmatise` with the pipeline already loaded, timed in the child |
| ollama-embed-warm | 36.9 | 54.6 | second `embed` call, model resident in the daemon, timed in the child |
| ollama-embed-evicted | 832.5 | 869.5 | `ollama stop embeddinggemma` before each run, timed in the child |
| sentence-transformers-cold | - | - | skipped: `google/embeddinggemma-300m` is a gated Hugging Face repo and no token is set up |
| sentence-transformers-warm | - | - | skipped: same gate |

`sentence-transformers` 6.0.1 itself installs cleanly under
`uv run --with sentence-transformers`; only the model download is gated.

## Answers

1. **spaCy is not affordable against the 200 ms budget**: a fresh process pays
   660.6 ms p95 to import spaCy, load `en_core_web_sm` and lemmatise the query
   (`lemmatise-cold`), which is over three times the whole budget, and it is
   affordable only if the real tolerance is one to two seconds.
2. **The Ollama daemon earns its place**: a warm embed costs 54.6 ms p95 against
   869.5 ms p95 when the model is evicted first, so the resident daemon saves
   about 0.8 s per read; the in-process `sentence-transformers` comparison the
   issue asked for is missing because the model repo is gated, and the evicted
   number is a lower bound for it because it excludes the torch import.
3. **`search()` should target 1 s, not 200 ms**: `uv run` start (130.7) plus
   spaCy cold (660.6) plus a warm embed (54.6) is 845.9 ms p95 before the
   SQLite k-nearest-neighbour and BM25 queries, which this measurement does not
   cover.
