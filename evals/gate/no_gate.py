import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from evals.gate.run import Case, load_cases
from memeng import extract, llm
from memeng.config import CONFIG, Config
from memeng.models.facts import ExtractedFact
from memeng.utils.common import now_iso


@dataclass(frozen=True)
class Outcome:
    case: Case
    facts: list[ExtractedFact] | None
    latency_s: float


def run_cases(
    cases: list[Case], model: str, *, run_prompt=llm.run_prompt
) -> list[Outcome]:
    outcomes = []
    for case in cases:
        started = time.perf_counter()
        try:
            facts = extract.parse_extraction(
                run_prompt(extract.render_extract_prompt(case.window, []), model), 0
            )
        except llm.LLMMalformed:
            facts = None
        outcomes.append(Outcome(case, facts, time.perf_counter() - started))
    return outcomes


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def report(model: str, model_id: str, outcomes: list[Outcome]) -> dict:
    positives = [o for o in outcomes if o.case.expected]
    negatives = [o for o in outcomes if not o.case.expected]
    invented = [o for o in negatives if o.facts]
    empty_yes = [o for o in positives if not o.facts]
    latencies = sorted(o.latency_s for o in outcomes)
    return {
        "model": model,
        "model_id": model_id,
        "n": len(outcomes),
        "invented_rate": _rate(len(invented), len(negatives)),
        "empty_yes_rate": _rate(len(empty_yes), len(positives)),
        "malformed": sum(o.facts is None for o in outcomes),
        "latency": {
            "mean_latency_s": statistics.mean(latencies) if latencies else None,
            "p95_latency_s": (
                latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
            ),
        },
        "invented": [
            {
                "id": o.case.id,
                "window_head": o.case.window.splitlines()[0],
                "facts": [
                    {"fact": f.fact, "rationale": f.rationale} for f in o.facts
                ],
            }
            for o in invented
        ],
        "empty_yes": [
            {"id": o.case.id, "window_head": o.case.window.splitlines()[0]}
            for o in empty_yes
        ],
    }


def main(
    argv: list[str] | None = None,
    *,
    run_prompt=llm.run_prompt,
    model_id=llm.model_id,
    load_config=Config.load,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).with_name("fixture.yaml")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).with_name("results")
    )
    args = parser.parse_args(argv)

    CONFIG.bind(load_config())
    outcomes = run_cases(load_cases(args.fixture), args.model, run_prompt=run_prompt)
    result = report(args.model, model_id(args.model), outcomes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{now_iso().replace(':', '')}-no-gate-{args.model}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
