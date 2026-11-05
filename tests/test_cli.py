import json

import pytest

from ai_eval_harness.cli import main

SYSTEM = "def answer(x):\n    return x * 2 if x != 2 else -1\n"


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "mysystem.py").write_text(SYSTEM, encoding="utf-8")
    lines = [json.dumps({"id": f"c{i}", "input": i, "expected": i * 2, "tags": ["t"]}) for i in range(1, 5)]
    (tmp_path / "cases.jsonl").write_text("\n".join(lines), encoding="utf-8")
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"name": "demo", "dataset": "cases.jsonl", "system": "mysystem:answer", "scorers": [{"type": "exact"}]}), encoding="utf-8")
    return path


def test_run_prints_report_and_saves(suite, tmp_path, capsys):
    out = tmp_path / "run.json"
    assert main(["run", str(suite), "--save", str(out), "--html", str(tmp_path / "r.html"), "--markdown", str(tmp_path / "r.md")]) == 0
    text = capsys.readouterr().out
    assert "Pass rate: 75.0%" in text and "c2:" in text
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["passed"] == 3
    assert (tmp_path / "r.html").exists() and (tmp_path / "r.md").exists()


def test_gate_fails_with_exit_code_1(suite, capsys):
    assert main(["run", str(suite), "--min-pass-rate", "0.9"]) == 1
    assert "GATE FAILED" in capsys.readouterr().err
    assert main(["run", str(suite), "--min-pass-rate", "0.7"]) == 0


def test_suite_level_gate_is_used_when_flag_absent(suite, tmp_path):
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["min_pass_rate"] = 1.0
    suite.write_text(json.dumps(data), encoding="utf-8")
    assert main(["run", str(suite)]) == 1


def test_tag_filter_that_matches_nothing_is_an_error(suite, capsys):
    assert main(["run", str(suite), "--tag", "nope"]) == 2
    assert "No cases" in capsys.readouterr().err


def test_bad_suite_is_exit_2_with_a_message(tmp_path, capsys):
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps({"name": "x", "dataset": "d.jsonl", "scorers": [], "bogus": 1}), encoding="utf-8")
    assert main(["run", str(bad)]) == 2
    assert "bogus" in capsys.readouterr().err


def test_recorded_outputs_need_no_system(suite, tmp_path, capsys):
    data = json.loads(suite.read_text(encoding="utf-8"))
    del data["system"]
    suite.write_text(json.dumps(data), encoding="utf-8")
    assert main(["run", str(suite)]) == 2
    assert "names no 'system'" in capsys.readouterr().err
    outputs = tmp_path / "out.jsonl"
    outputs.write_text("\n".join(json.dumps({"id": f"c{i}", "output": i * 2}) for i in range(1, 5)), encoding="utf-8")
    assert main(["run", str(suite), "--outputs", str(outputs), "--min-pass-rate", "1"]) == 0


def test_judge_scorers_without_a_judge_fail_visibly_and_skip_judge_removes_them(suite, tmp_path, capsys):
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["scorers"].append({"type": "llm_judge", "rubric": "r"})
    suite.write_text(json.dumps(data), encoding="utf-8")
    main(["run", str(suite)])
    assert "judge was unavailable" in capsys.readouterr().out
    assert main(["run", str(suite), "--skip-judge"]) == 0
    assert "unavailable" not in capsys.readouterr().out


def test_gemini_judge_without_key_is_a_clean_error(suite, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert main(["run", str(suite), "--judge", "gemini"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_compare_and_show_round_trip(suite, tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["run", str(suite), "--save", str(a)])
    main(["run", str(suite), "--save", str(b)])
    capsys.readouterr()
    assert main(["compare", str(a), str(b), "--fail-on-regression"]) == 0
    assert "no change" in capsys.readouterr().out
    assert main(["show", str(a)]) == 0
    assert main(["compare", str(a), str(tmp_path / "missing.json")]) == 2


def test_validate(suite, capsys):
    assert main(["validate", str(suite)]) == 0
    assert "is valid: 4 cases" in capsys.readouterr().out
