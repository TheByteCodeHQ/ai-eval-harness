import json

import pytest

from ai_eval_harness.compare import compare_runs
from ai_eval_harness.dataset import DatasetError, load_dataset
from ai_eval_harness.models import Case, SystemResult
from ai_eval_harness.paths import PathError, get_path
from ai_eval_harness.report import load_run, render_comparison, render_html, render_text, save_run, summarize
from ai_eval_harness.runner import recorded_system, run_suite
from ai_eval_harness.scorers import build_scorers
from ai_eval_harness.stats import bootstrap_mean_diff, mcnemar_exact, percentile, wilson_interval

# -- paths ---------------------------------------------------------------------------------------------

def test_paths():
    value = {"a": [{"b": 1}, {"b": 2}, {"c": 3}]}
    assert get_path(value, "a.1.b") == 2
    assert get_path(value, "a.*.b") == [1, 2]          # an item without the field is skipped
    assert get_path(value, "") is value
    assert get_path(value, "a.9.b", default="d") == "d"
    with pytest.raises(PathError):
        get_path(value, "a.9.b")
    with pytest.raises(PathError):
        get_path({"a": 1}, "a.*")


# -- dataset -------------------------------------------------------------------------------------------

def write(tmp_path, *lines):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dataset_loads_skips_comments_and_selects(tmp_path):
    ds = load_dataset(write(tmp_path, "# header", "", '{"id":"a","input":1,"expected":2,"tags":["x"]}', '{"id":"b","input":3,"tags":["y"]}'))
    assert len(ds) == 2
    assert [c.id for c in ds.select(tags=["x"])] == ["a"]
    assert len(ds.select(limit=1)) == 1


@pytest.mark.parametrize("line,message", [
    ("not json", "line 1: not valid JSON"),
    ('{"input":1}', "'id'"),
    ('{"id":"a"}', "'input' is required"),
    ('{"id":"a","input":1,"extra":1}', "unknown field"),
    ('{"id":"a","input":1,"tags":"x"}', "'tags'"),
    ('{"id":"a","input":1,"metadata":[]}', "'metadata'"),
    ("[1]", "JSON object"),
])
def test_dataset_errors_name_the_line(tmp_path, line, message):
    with pytest.raises(DatasetError, match=message):
        load_dataset(write(tmp_path, line))


def test_dataset_duplicate_ids_and_empty(tmp_path):
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(write(tmp_path, '{"id":"a","input":1}', '{"id":"a","input":2}'))
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(write(tmp_path, "# nothing"))
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_dataset_warns_about_weak_datasets(tmp_path):
    ds = load_dataset(write(tmp_path, '{"id":"a","input":1}', '{"id":"b","input":1}'))
    text = " ".join(ds.warnings)
    assert "more than one case" in text and "expected" in text and "tags" in text


# -- stats ---------------------------------------------------------------------------------------------

def test_wilson_interval_behaves_at_the_edges():
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.65 < low < 0.75
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.01) and high == pytest.approx(0.596, abs=0.01)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(1, 9) == mcnemar_exact(9, 1)


def test_bootstrap_is_repeatable_and_brackets_the_mean():
    a, b = [0.0] * 20, [1.0, 0.0] * 10
    first = bootstrap_mean_diff(a, b)
    assert first == bootstrap_mean_diff(a, b)
    diff, low, high = first
    assert diff == 0.5 and low <= diff <= high
    with pytest.raises(ValueError):
        bootstrap_mean_diff([1], [1, 2])


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.9) == 0.0


# -- runner --------------------------------------------------------------------------------------------

CASES = [Case(f"c{i}", i, i * 2, ("even",) if i % 2 == 0 else ("odd",)) for i in range(10)]
EXACT = build_scorers([{"type": "exact"}])


def double(x):
    return x * 2


def test_run_scores_every_case_in_order_regardless_of_workers():
    run = run_suite(suite="s", system=double, system_name="double", cases=CASES, scorers=EXACT, workers=8)
    assert [r.case.id for r in run.results] == [c.id for c in CASES]
    assert all(r.passed for r in run.results)


def test_exceptions_become_error_outcomes_not_a_crashed_run():
    def flaky(x):
        if x == 3:
            raise ValueError("boom")
        return x * 2

    run = run_suite(suite="s", system=flaky, system_name="f", cases=CASES, scorers=EXACT)
    bad = [r for r in run.results if not r.passed]
    assert [r.case.id for r in bad] == ["c3"]
    assert "ValueError: boom" in bad[0].attempts[0].outcome.error
    assert summarize(run)["errors"] == 1
    assert bad[0].attempts[0].scores == []     # scorers are not run on an error, so no misleading score is recorded
    assert bad[0].value == 0.0                 # an error scores zero rather than vanishing from the average


def test_repeats_expose_inconsistent_systems():
    calls = {"n": 0}

    def alternating(x):
        calls["n"] += 1
        return x * 2 if calls["n"] % 2 else -1

    run = run_suite(suite="s", system=alternating, system_name="a", cases=CASES[:1], scorers=EXACT, repeats=2, workers=1)
    result = run.results[0]
    assert result.inconsistent and not result.passed and result.pass_rate == 0.5
    assert summarize(run)["inconsistent"] == 1


def test_repeat_and_empty_limits():
    with pytest.raises(ValueError):
        run_suite(suite="s", system=double, system_name="d", cases=CASES, scorers=EXACT, repeats=0)
    with pytest.raises(ValueError):
        run_suite(suite="s", system=double, system_name="d", cases=[], scorers=EXACT)


def test_usage_is_summed():
    run = run_suite(suite="s", system=lambda x: SystemResult(x * 2, {"tokens": 10}), system_name="u", cases=CASES, scorers=EXACT)
    assert summarize(run)["usage"] == {"tokens": 100}


def test_recorded_outputs_are_scored_without_calling_anything():
    system = recorded_system({c.id: c.expected for c in CASES}, CASES)
    run = run_suite(suite="s", system=system, system_name="r", cases=CASES, scorers=EXACT)
    assert all(r.passed for r in run.results)
    missing = run_suite(suite="s", system=recorded_system({}, CASES), system_name="r", cases=CASES[:1], scorers=EXACT)
    assert "no recorded output" in missing.results[0].attempts[0].outcome.error


def test_non_gating_scores_cannot_fail_a_case():
    scorers = build_scorers([{"type": "exact"}, {"type": "latency", "max_seconds": -1}])
    run = run_suite(suite="s", system=double, system_name="d", cases=CASES[:2], scorers=scorers)
    assert all(r.passed for r in run.results)
    assert summarize(run)["scorers"]["latency"]["gating"] is False


def test_by_tag_breakdown_keeps_a_weak_area_visible():
    run = run_suite(suite="s", system=lambda x: x * 2 if x % 2 else -1, system_name="s", cases=CASES, scorers=EXACT)
    tags = summarize(run)["by_tag"]
    assert tags["odd"]["pass_rate"] == 1.0 and tags["even"]["pass_rate"] == 0.0


# -- save / load / render ------------------------------------------------------------------------------

def test_saved_run_round_trips(tmp_path):
    run = run_suite(suite="s", system=lambda x: x * 2 if x else -1, system_name="d", cases=CASES, scorers=EXACT, repeats=2)
    path = tmp_path / "out" / "run.json"
    save_run(run, path)
    loaded = load_run(path)
    assert summarize(loaded)["pass_rate"] == summarize(run)["pass_rate"]
    assert [r.passed for r in loaded.results] == [r.passed for r in run.results]
    assert loaded.results[1].case.tags == run.results[1].case.tags


def test_load_run_rejects_foreign_files(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an ai-eval-harness run"):
        load_run(path)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read"):
        load_run(path)


def test_text_report_names_failures_and_judge_outages():
    run = run_suite(suite="s", system=lambda x: -1, system_name="bad", cases=CASES[:3], scorers=EXACT)
    text = render_text(run)
    assert "Pass rate: 0.0%" in text and "c0: exact:" in text
    judged = build_scorers([{"type": "llm_judge", "rubric": "r"}], None)
    run = run_suite(suite="s", system=double, system_name="d", cases=CASES[:2], scorers=judged)
    assert "judge was unavailable" in render_text(run)


def test_html_report_escapes_everything():
    evil = [Case("<script>alert(1)</script>", 1, 2)]
    run = run_suite(suite="<b>s</b>", system=lambda x: "<img src=x onerror=1>", system_name="<i>", cases=evil, scorers=EXACT)
    page = render_html(run)
    assert "<script>" not in page and "<img" not in page and "<b>s</b>" not in page
    assert "&lt;script&gt;" in page


# -- compare -------------------------------------------------------------------------------------------

def runs_with(base_pass, cand_pass):
    cases = [Case(f"c{i}", i, 1) for i in range(len(base_pass))]
    outputs = []
    for passes in (base_pass, cand_pass):
        outputs.append(run_suite(suite="s", system=lambda x, p=passes: 1 if p[x] else 0, system_name=str(id(passes)), cases=cases, scorers=EXACT))
    return outputs


def test_compare_flags_regressions_even_when_the_average_improves():
    base, cand = runs_with([True] * 10 + [False] * 10, [False] + [True] * 9 + [True] * 10)
    result = compare_runs(base, cand)
    assert result.regressed == ["c0"] and len(result.fixed) == 10
    assert result.candidate_rate > result.baseline_rate
    assert "Regressed cases: c0" in render_comparison(result)


def test_compare_calls_a_tiny_gain_unproven():
    base, cand = runs_with([False, True, True], [True, True, True])
    result = compare_runs(base, cand)
    assert result.verdict == "improved, not proven"
    assert any("Only 3 shared" in w for w in result.warnings)


def test_compare_identical_and_disjoint():
    base, cand = runs_with([True] * 5, [True] * 5)
    assert compare_runs(base, cand).verdict == "no change"
    other = run_suite(suite="s", system=double, system_name="o", cases=[Case("zzz", 1, 2)], scorers=EXACT)
    result = compare_runs(base, other)
    assert result.verdict == "no comparable cases" and result.skipped


def test_compare_significant_improvement():
    base, cand = runs_with([False] * 12 + [True] * 8, [True] * 20)
    result = compare_runs(base, cand)
    assert result.significant and result.verdict == "better"
