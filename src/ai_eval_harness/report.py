"""Turn a run into numbers, text and JSON, and read a saved run back."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .compare import Comparison
from .models import Attempt, Case, CaseResult, Outcome, RunResult, Score
from .stats import percentile, wilson_interval

FORMAT_VERSION = 1


def summarize(run: RunResult) -> dict[str, Any]:
    results = run.results
    passed = sum(r.passed for r in results)
    low, high = wilson_interval(passed, len(results))
    attempts = [a for r in results for a in r.attempts]
    latencies = [a.outcome.latency_s for a in attempts if a.outcome.ok]
    scorers: dict[str, list[Score]] = defaultdict(list)
    for attempt in attempts:
        for score in attempt.scores:
            scorers[score.scorer].append(score)
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in run.tags():
        group = [r for r in results if tag in r.case.tags]
        ok = sum(r.passed for r in group)
        t_low, t_high = wilson_interval(ok, len(group))
        by_tag[tag] = {"cases": len(group), "passed": ok, "pass_rate": ok / len(group), "ci": [t_low, t_high]}
    usage: dict[str, float] = defaultdict(float)
    for attempt in attempts:
        for key, number in attempt.outcome.usage.items():
            usage[key] += number
    return {
        "cases": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "pass_rate_ci": [low, high],
        "mean_score": sum(r.value for r in results) / len(results) if results else 0.0,
        "errors": sum(not a.outcome.ok for a in attempts),
        "inconsistent": sum(r.inconsistent for r in results),
        "judge_unavailable": sum(bool(s.details.get("unavailable")) for a in attempts for s in a.scores),
        "latency_p50": percentile(latencies, 0.5),
        "latency_p95": percentile(latencies, 0.95),
        "usage": dict(usage),
        "scorers": {
            name: {
                "mean": sum(s.value for s in scs) / len(scs),
                "pass_rate": sum(s.passed for s in scs) / len(scs),
                "gating": scs[0].gating,
            }
            for name, scs in scorers.items()
        },
        "by_tag": by_tag,
    }


def failures(run: RunResult) -> list[CaseResult]:
    return [r for r in run.results if not r.passed]


def _reason(result: CaseResult) -> str:
    for attempt in result.attempts:
        if not attempt.outcome.ok:
            return f"system error: {attempt.outcome.error}"
        for score in attempt.scores:
            if score.gating and not score.passed:
                return f"{score.scorer}: {score.reason or 'failed'}"
    return "failed"


def render_text(run: RunResult, *, limit: int = 15) -> str:
    s = summarize(run)
    low, high = s["pass_rate_ci"]
    lines = [
        f"Suite: {run.suite}    System: {run.system}    Repeats: {run.repeats}",
        f"Cases: {s['cases']}   Passed: {s['passed']}   Pass rate: {s['pass_rate']:.1%}  (95% interval {low:.0%} to {high:.0%})",
        f"Mean score: {s['mean_score']:.3f}   Latency p50/p95: {s['latency_p50']:.3f}s / {s['latency_p95']:.3f}s",
    ]
    if s["errors"]:
        lines.append(f"System errors: {s['errors']}")
    if s["inconsistent"]:
        lines.append(f"Inconsistent cases (passed on some repeats, failed on others): {s['inconsistent']}")
    if s["judge_unavailable"]:
        lines.append(f"WARNING: the judge was unavailable for {s['judge_unavailable']} score(s); they count as failures, not passes")
    if s["usage"]:
        lines.append("Usage: " + ", ".join(f"{k}={v:g}" for k, v in sorted(s["usage"].items())))
    lines += ["", "Scorers"]
    for name, data in s["scorers"].items():
        lines.append(f"  {name:<24} pass {data['pass_rate']:.0%}   mean {data['mean']:.3f}" + ("" if data["gating"] else "   (informational)"))
    if s["by_tag"]:
        lines += ["", "By tag"]
        for tag, data in sorted(s["by_tag"].items(), key=lambda kv: kv[1]["pass_rate"]):
            lines.append(f"  {tag:<24} {data['passed']}/{data['cases']}  {data['pass_rate']:.0%}  (95% {data['ci'][0]:.0%} to {data['ci'][1]:.0%})")
    bad = failures(run)
    if bad:
        lines += ["", f"Failing cases ({len(bad)})"]
        for result in bad[:limit]:
            lines.append(f"  {result.case.id}: {_reason(result)[:160]}")
        if len(bad) > limit:
            lines.append(f"  ... and {len(bad) - limit} more (see the JSON report)")
    for warning in run.warnings:
        lines.append(f"Note: {warning}")
    return "\n".join(lines)


def render_comparison(c: Comparison) -> str:
    lines = [
        f"Baseline: {c.baseline}    Candidate: {c.candidate}    Shared cases: {c.shared}",
        f"Pass rate: {c.baseline_rate:.1%} -> {c.candidate_rate:.1%}",
        f"Mean score change: {c.score_diff:+.3f}  (95% interval {c.score_low:+.3f} to {c.score_high:+.3f})",
        f"Fixed: {len(c.fixed)}   Regressed: {len(c.regressed)}   McNemar p = {c.p_value:.3f}",
        f"Verdict: {c.verdict}",
    ]
    if c.regressed:
        lines.append("Regressed cases: " + ", ".join(c.regressed[:15]) + (" ..." if len(c.regressed) > 15 else ""))
    if c.fixed:
        lines.append("Fixed cases: " + ", ".join(c.fixed[:15]) + (" ..." if len(c.fixed) > 15 else ""))
    lines += [f"Note: {w}" for w in c.warnings]
    return "\n".join(lines)


def render_markdown(run: RunResult) -> str:
    s = summarize(run)
    out = [f"# Evaluation: {run.suite}", "", f"System `{run.system}`, {s['cases']} cases, {run.repeats} repeat(s).", "",
           f"**Pass rate {s['pass_rate']:.1%}** (95% interval {s['pass_rate_ci'][0]:.0%} to {s['pass_rate_ci'][1]:.0%}), mean score {s['mean_score']:.3f}.", "",
           "| Scorer | Pass | Mean |", "|---|---|---|"]
    out += [f"| {n} | {d['pass_rate']:.0%} | {d['mean']:.3f} |" for n, d in s["scorers"].items()]
    if s["by_tag"]:
        out += ["", "| Tag | Passed | Rate |", "|---|---|---|"]
        out += [f"| {t} | {d['passed']}/{d['cases']} | {d['pass_rate']:.0%} |" for t, d in sorted(s["by_tag"].items())]
    bad = failures(run)
    if bad:
        out += ["", "## Failing cases", ""]
        out += [f"- `{r.case.id}`: {_reason(r)[:160]}" for r in bad[:30]]
    return "\n".join(out) + "\n"


def render_html(run: RunResult) -> str:
    """One self-contained file, safe to open or attach: every value from the run is escaped."""
    s = summarize(run)
    rows = "".join(
        f"<tr class='{'ok' if r.passed else 'bad'}'><td>{html.escape(r.case.id)}</td><td>{'pass' if r.passed else 'FAIL'}</td>"
        f"<td>{r.value:.2f}</td><td>{html.escape('' if r.passed else _reason(r)[:200])}</td></tr>"
        for r in run.results
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Evaluation report</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:60rem}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #ddd;padding:.35rem .6rem;text-align:left}.bad td{background:#fdecea}.ok td{color:#555}</style>"
        f"<h1>{html.escape(run.suite)}</h1><p>System <code>{html.escape(run.system)}</code>. "
        f"Pass rate <b>{s['pass_rate']:.1%}</b> ({s['passed']}/{s['cases']}), 95% interval "
        f"{s['pass_rate_ci'][0]:.0%} to {s['pass_rate_ci'][1]:.0%}.</p>"
        f"<table><tr><th>Case</th><th>Result</th><th>Score</th><th>Why</th></tr>{rows}</table>"
    )


# -- saving and loading --------------------------------------------------------------------------------

def to_dict(run: RunResult) -> dict[str, Any]:
    def attempt(a: Attempt) -> dict[str, Any]:
        return {
            "value": a.outcome.value, "error": a.outcome.error, "latency_s": round(a.outcome.latency_s, 6), "usage": a.outcome.usage,
            "scores": [{"scorer": s.scorer, "value": s.value, "passed": s.passed, "reason": s.reason, "details": s.details, "gating": s.gating}
                       for s in a.scores],
        }

    return {
        "format": FORMAT_VERSION, "suite": run.suite, "system": run.system, "repeats": run.repeats, "started_at": run.started_at,
        "config": run.config, "warnings": run.warnings, "summary": summarize(run),
        "results": [
            {"id": r.case.id, "input": r.case.input, "expected": r.case.expected, "tags": list(r.case.tags), "metadata": r.case.metadata,
             "passed": r.passed, "attempts": [attempt(a) for a in r.attempts]}
            for r in run.results
        ],
    }


def save_run(run: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(run), indent=2, default=str), encoding="utf-8")


def load_run(path: Path) -> RunResult:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read run file {path}: {exc}") from None
    if not isinstance(data, dict) or data.get("format") != FORMAT_VERSION or "results" not in data:
        raise ValueError(f"{path} is not an ai-eval-harness run file (format {FORMAT_VERSION})")
    results = []
    for item in data["results"]:
        case = Case(item["id"], item.get("input"), item.get("expected"), tuple(item.get("tags", [])), item.get("metadata", {}))
        attempts = [
            Attempt(Outcome(a.get("value"), a.get("error"), a.get("latency_s", 0.0), a.get("usage", {})),
                    [Score(s["scorer"], s["value"], s["passed"], s.get("reason", ""), s.get("details", {}), s.get("gating", True))
                     for s in a.get("scores", [])])
            for a in item["attempts"]
        ]
        results.append(CaseResult(case, attempts))
    return RunResult(data["suite"], data["system"], data["repeats"], results, data.get("config", {}), data.get("started_at", ""),
                     data.get("warnings", []))
