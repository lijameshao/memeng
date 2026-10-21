import argparse
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

from memeng.config import Config


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    before_each: tuple[str, ...] = ()
    timed_in_child: bool = False


@dataclass(frozen=True)
class Row:
    step: str
    samples: tuple[float, ...]
    note: str = ""


def steps_for(config: Config) -> list[Step]:
    python = sys.executable
    query = "where did we decide to keep the vectors and why was that"
    spacy_load = "spacy.load('en_core_web_sm', disable=['ner', 'parser'])"
    hf_model = "google/embeddinggemma-300m"

    lemmatise_cold = textwrap.dedent(f"""
        import time
        start = time.perf_counter()
        from memeng.text import lemmatise
        lemmatise({query!r})
        print((time.perf_counter() - start) * 1000)
    """)
    lemmatise_loaded = textwrap.dedent(f"""
        import time
        from memeng.text import lemmatise
        lemmatise('warm the pipeline')
        start = time.perf_counter()
        lemmatise({query!r})
        print((time.perf_counter() - start) * 1000)
    """)
    embed_warm = textwrap.dedent(f"""
        import time
        from memeng.config import Config
        from memeng.embed import embed
        config = Config.load()
        embed([{query!r}], config, timeout_s=60)
        start = time.perf_counter()
        embed([{query!r}], config, timeout_s=60)
        print((time.perf_counter() - start) * 1000)
    """)
    embed_evicted = textwrap.dedent(f"""
        import time
        from memeng.config import Config
        from memeng.embed import embed
        config = Config.load()
        start = time.perf_counter()
        embed([{query!r}], config, timeout_s=60)
        print((time.perf_counter() - start) * 1000)
    """)
    transformers_cold = textwrap.dedent(f"""
        import time
        start = time.perf_counter()
        from sentence_transformers import SentenceTransformer
        SentenceTransformer({hf_model!r}).encode([{query!r}])
        print((time.perf_counter() - start) * 1000)
    """)
    transformers_warm = textwrap.dedent(f"""
        import time
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer({hf_model!r})
        model.encode([{query!r}])
        start = time.perf_counter()
        model.encode([{query!r}])
        print((time.perf_counter() - start) * 1000)
    """)
    with_transformers = ("uv", "run", "--with", "sentence-transformers", "python", "-c")

    return [
        Step("python-start", (python, "-c", "pass")),
        Step("uv-run-python-start", ("uv", "run", "python", "-c", "pass")),
        Step("spacy-load-cold", (python, "-c", f"import spacy; {spacy_load}")),
        Step("lemmatise-cold", (python, "-c", lemmatise_cold), timed_in_child=True),
        Step(
            "lemmatise-after-load",
            (python, "-c", lemmatise_loaded),
            timed_in_child=True,
        ),
        Step("ollama-embed-warm", (python, "-c", embed_warm), timed_in_child=True),
        Step(
            "ollama-embed-evicted",
            (python, "-c", embed_evicted),
            before_each=("ollama", "stop", config.embed_model),
            timed_in_child=True,
        ),
        Step(
            "sentence-transformers-cold",
            (*with_transformers, transformers_cold),
            timed_in_child=True,
        ),
        Step(
            "sentence-transformers-warm",
            (*with_transformers, transformers_warm),
            timed_in_child=True,
        ),
    ]


def measure(step: Step, runs: int) -> Row:
    samples = []
    for _ in range(runs):
        try:
            if step.before_each:
                subprocess.run(
                    step.before_each, check=True, capture_output=True, text=True
                )
            start = time.perf_counter()
            completed = subprocess.run(
                step.argv, check=True, capture_output=True, text=True
            )
            wall_ms = (time.perf_counter() - start) * 1000
        except subprocess.CalledProcessError as failed:
            reported = (failed.stderr or "").strip().splitlines()
            return Row(
                step.name,
                (),
                reported[-1][:200] if reported else f"exit {failed.returncode}",
            )
        except OSError as unusable:
            return Row(step.name, (), f"{unusable.filename}: {unusable.strerror}")
        if step.timed_in_child:
            samples.append(float(completed.stdout.strip().splitlines()[-1]))
        else:
            samples.append(wall_ms)
    return Row(step.name, tuple(samples))


def render(rows: list[Row]) -> str:
    lines = ["| step | median ms | p95 ms | notes |", "|---|---|---|---|"]
    for row in rows:
        if row.samples:
            median = f"{statistics.median(row.samples):.1f}"
            p95 = f"{statistics.quantiles(row.samples, n=20)[18]:.1f}"
            note = row.note
        else:
            median = p95 = "-"
            note = f"skipped: {row.note}"
        lines.append(f"| {row.step} | {median} | {p95} | {note} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, steps: list[Step] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    rows = []
    for step in steps_for(Config.load()) if steps is None else steps:
        print(step.name, file=sys.stderr)
        rows.append(measure(step, args.runs))
    print(render(rows))


if __name__ == "__main__":
    main()
