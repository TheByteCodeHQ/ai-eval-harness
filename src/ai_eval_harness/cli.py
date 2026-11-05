"""Command line: aieval run | compare | show | validate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare_runs
from .dataset import DatasetError, load_dataset
from .llm import JudgeError, make_judge
from .report import load_run, render_comparison, render_html, render_markdown, render_text, save_run, summarize
from .runner import recorded_system, run_suite
from .scorers import ScorerConfigError, build_scorers
from .suite import load_suite

OK, GATE_FAILED, USAGE = 0, 1, 2


def _read_outputs(path: Path) -> dict:
    outputs = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            outputs[item["id"]] = item["output"]
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"{path.name} line {number}: each line must be {{\"id\": ..., \"output\": ...}}") from None
    return outputs


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    dataset = load_dataset(suite.dataset)
    cases = dataset.select(tags=args.tag, limit=args.limit)
    judge = None if args.skip_judge else make_judge(args.judge, args.judge_model)
    scorers = build_scorers(suite.scorers, judge)
    if args.skip_judge:
        scorers = [s for s in scorers if type(s).__name__ != "LLMJudge"]
        if not scorers:
            raise ScorerConfigError("--skip-judge removed every scorer")
    warnings = list(dataset.warnings)
    if args.outputs:
        system, name = recorded_system(_read_outputs(Path(args.outputs)), cases), f"recorded:{Path(args.outputs).name}"
    else:
        system, name = suite.load_system(), suite.system or "system"
    repeats = args.repeats or suite.repeats
    run = run_suite(suite=suite.name, system=system, system_name=name, cases=cases, scorers=scorers, repeats=repeats, workers=args.workers,
                    config={"suite_file": suite.path.name if suite.path else "", "tags": args.tag or [], "judge": args.judge}, warnings=warnings)
    print(render_text(run))
    if args.save:
        save_run(run, Path(args.save))
        print(f"\nSaved run to {args.save}")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(run), encoding="utf-8")
    if args.html:
        Path(args.html).write_text(render_html(run), encoding="utf-8")
    gate = args.min_pass_rate if args.min_pass_rate is not None else suite.min_pass_rate
    if gate is not None:
        rate = summarize(run)["pass_rate"]
        if rate + 1e-12 < gate:
            print(f"\nGATE FAILED: pass rate {rate:.1%} is below the required {gate:.1%}", file=sys.stderr)
            return GATE_FAILED
        print(f"\nGate passed: pass rate {rate:.1%} meets the required {gate:.1%}")
    return OK


def cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(load_run(Path(args.baseline)), load_run(Path(args.candidate)))
    print(render_comparison(comparison))
    if args.fail_on_regression and comparison.regressed:
        print(f"\nFAILED: {len(comparison.regressed)} case(s) that used to pass now fail", file=sys.stderr)
        return GATE_FAILED
    return OK


def cmd_show(args: argparse.Namespace) -> int:
    print(render_text(load_run(Path(args.run)), limit=args.limit))
    return OK


def cmd_validate(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    dataset = load_dataset(suite.dataset)
    scorers = build_scorers(suite.scorers, None)
    print(f"Suite '{suite.name}' is valid: {len(dataset)} cases, {len(scorers)} scorers ({', '.join(s.name for s in scorers)}).")
    for warning in dataset.warnings:
        print(f"Warning: {warning}")
    return OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aieval", description="Evaluate an AI system: score it, repeat it, compare it, gate on it.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and score the results")
    run.add_argument("suite")
    run.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    run.add_argument("--limit", type=int, help="only the first N matching cases")
    run.add_argument("--repeats", type=int, help="run each case N times (overrides the suite)")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--outputs", help="score recorded outputs (JSONL of {id, output}) instead of calling the system")
    run.add_argument("--judge", choices=["none", "gemini"], default="none")
    run.add_argument("--judge-model")
    run.add_argument("--skip-judge", action="store_true", help="leave out llm_judge scorers")
    run.add_argument("--min-pass-rate", type=float, help="exit 1 if the pass rate is lower (overrides the suite)")
    run.add_argument("--save", help="write the full run as JSON")
    run.add_argument("--markdown")
    run.add_argument("--html")
    run.set_defaults(func=cmd_run)

    comp = sub.add_parser("compare", help="compare two saved runs")
    comp.add_argument("baseline")
    comp.add_argument("candidate")
    comp.add_argument("--fail-on-regression", action="store_true")
    comp.set_defaults(func=cmd_compare)

    show = sub.add_parser("show", help="print a saved run")
    show.add_argument("run")
    show.add_argument("--limit", type=int, default=15)
    show.set_defaults(func=cmd_show)

    val = sub.add_parser("validate", help="check a suite and its dataset without running anything")
    val.add_argument("suite")
    val.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, ScorerConfigError, JudgeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    raise SystemExit(main())
