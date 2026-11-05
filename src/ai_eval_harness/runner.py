"""Run a system over a dataset, score every output, and collect the results."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from .models import Attempt, Case, CaseResult, Outcome, RunResult, SystemResult
from .scorers import Scorer

System = Callable[[Any], Any]
MAX_WORKERS = 32
MAX_REPEATS = 50


def call_system(system: System, case: Case) -> Outcome:
    """Call the system once. Any exception becomes an error outcome: one bad case must not stop the run."""
    started = time.perf_counter()
    try:
        raw = system(case.input)
    except Exception as exc:  # noqa: BLE001 - the system under test is untrusted
        detail = traceback.extract_tb(exc.__traceback__)[-1]
        return Outcome(error=f"{type(exc).__name__}: {str(exc)[:300]} ({detail.filename.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}:{detail.lineno})",
                       latency_s=time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    if isinstance(raw, SystemResult):
        return Outcome(value=raw.value, latency_s=elapsed, usage=dict(raw.usage))
    return Outcome(value=raw, latency_s=elapsed)


def recorded_system(outputs: dict[str, Any], cases: list[Case]) -> System:
    """Score outputs you already have (id -> output) instead of calling a live system. Free, instant and repeatable."""
    by_input = {id(c.input): c.id for c in cases}

    def system(value: Any) -> Any:
        case_id = by_input.get(id(value))
        if case_id is None or case_id not in outputs:
            raise LookupError(f"no recorded output for case '{case_id}'")
        return outputs[case_id]

    return system


def score_attempt(case: Case, outcome: Outcome, scorers: list[Scorer]) -> Attempt:
    if not outcome.ok:
        return Attempt(outcome, [])
    scores = []
    for scorer in scorers:
        try:
            scores.append(scorer.score(case, outcome))
        except Exception as exc:  # noqa: BLE001 - a broken scorer is reported as a failed score
            scores.append(scorer._result(0.0, False, f"the scorer crashed: {type(exc).__name__}: {exc}"))
    return Attempt(outcome, scores)


def run_suite(
    *,
    suite: str,
    system: System,
    system_name: str,
    cases: list[Case],
    scorers: list[Scorer],
    repeats: int = 1,
    workers: int = 4,
    config: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> RunResult:
    if not cases:
        raise ValueError("No cases to run (did a tag filter remove them all?)")
    if not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats must be between 1 and {MAX_REPEATS}")
    workers = max(1, min(workers, MAX_WORKERS))

    def one(job: tuple[Case, int]) -> tuple[str, int, Attempt]:
        case, repeat = job
        return case.id, repeat, score_attempt(case, call_system(system, case), scorers)

    jobs = [(case, repeat) for case in cases for repeat in range(repeats)]
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if workers == 1:
        done = [one(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(one, jobs))          # map keeps input order, so reports do not depend on thread timing
    grouped: dict[str, list[tuple[int, Attempt]]] = {}
    for case_id, repeat, attempt in done:
        grouped.setdefault(case_id, []).append((repeat, attempt))
    results = [CaseResult(case, [a for _, a in sorted(grouped[case.id], key=lambda pair: pair[0])]) for case in cases]
    return RunResult(suite, system_name, repeats, results, config or {}, started_at, list(warnings or []))
