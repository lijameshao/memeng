import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from memeng import extract, llm
from memeng.config import CONFIG, Config
from memeng.utils.common import now_iso


@dataclass(frozen=True)
class Case:
    id: str
    expected: bool
    window: str


@dataclass(frozen=True)
class Outcome:
    case: Case
    answer: bool | None
    latency_s: float


def load_cases(path: Path) -> list[Case]:
    labels = {True: True, "yes": True, False: False, "no": False}
    cases = []
    for item in yaml.safe_load(path.read_text()):
        if item["label"] not in labels:
            raise ValueError(f"{item['id']}: label must be yes or no, not {item['label']!r}")
        cases.append(
            Case(id=item["id"], expected=labels[item["label"]], window=item["window"])
        )
    return cases


def run_cases(
    cases: list[Case], model: str, *, run_prompt=llm.run_prompt
) -> list[Outcome]:
    outcomes = []
    for case in cases:
        started = time.perf_counter()
        try:
            answer = extract.parse_gate(
                run_prompt(extract.render_gate_prompt(case.window), model)
            )
        except (extract.LLMMalformed, llm.LLMMalformed):
            answer = None
        outcomes.append(Outcome(case, answer, time.perf_counter() - started))
    return outcomes


def report(model: str, model_id: str, outcomes: list[Outcome]) -> dict:
    positives = [o for o in outcomes if o.case.expected]
    negatives = [o for o in outcomes if not o.case.expected]
    latencies = sorted(o.latency_s for o in outcomes)
    return {
        "model": model,
        "model_id": model_id,
        "n": len(outcomes),
        "fn_rate": sum(o.answer is not True for o in positives) / len(positives),
        "fp_rate": sum(o.answer is True for o in negatives) / len(negatives),
        "latency": {
            "mean_latency_s": statistics.mean(latencies),
            "p95_latency_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
        },
        "misses": [
            {
                "id": o.case.id,
                "label": "yes" if o.case.expected else "no",
                "answer": {True: "yes", False: "no", None: "malformed"}[o.answer],
                "window_head": o.case.window.splitlines()[0],
            }
            for o in outcomes
            if o.answer != o.case.expected
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
    path = args.out_dir / f"{now_iso().replace(':', '')}-{args.model}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
