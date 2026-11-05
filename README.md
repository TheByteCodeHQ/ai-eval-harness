# ai-eval-harness

A small, honest evaluation harness for AI systems. Give it a dataset of cases and a system to test. It scores every output,
repeats runs to expose flakiness, says how sure it is, compares two versions fairly, and fails your build when quality drops.

It has **no runtime dependencies** and needs no model to run. A model is only used if you ask for an LLM judge, and then
the judge is guarded (see below).

## Why another one?

Most "evals" are a pass rate printed to two decimals. That hides things that decide whether you can trust a change:

| Question | What this harness does |
|---|---|
| Is 93% on 30 cases real? | Prints a **95% Wilson interval** next to every pass rate, overall and per tag |
| Did the average go up while something broke? | `compare` lists **regressed cases by name**, and `--fail-on-regression` fails on any |
| Is the new version really better? | **Paired comparison** with an exact McNemar test and a bootstrap interval; small gains are labelled *not proven* |
| Does it give the same answer twice? | `repeats`: a case passes only if **every** repeat passes; inconsistent cases are counted |
| Did the system crash on a case? | Becomes an error outcome that scores zero. It cannot disappear from the average |
| Did the judge model fail or get tricked? | Counts as a **failure**, never a pass, and is reported loudly |
| Where is it weak? | Results are broken down by **tag** so a weak area stays visible |

## Install

```bash
pip install -e .            # no dependencies
pip install -e ".[gemini]"  # only if you want the Gemini judge
```

Python 3.10 or newer.

## Quick start

A **suite** is a small JSON file. A **dataset** is JSON Lines, one case per line:

```jsonl
{"id": "refund-1", "input": "I want my money back", "expected": "refund", "tags": ["billing"]}
```

```json
{
  "name": "support-triage",
  "dataset": "cases.jsonl",
  "system": "my_app:triage",
  "repeats": 3,
  "min_pass_rate": 0.9,
  "scorers": [
    {"type": "exact", "normalize": true},
    {"type": "no_secrets"},
    {"type": "latency", "max_seconds": 2}
  ]
}
```

`system` is any Python function `module:function` that takes a case input and returns the output (or a `SystemResult(value, usage)`
to also report tokens or cost).

```bash
aieval validate suite.json                       # check the suite and dataset, run nothing
aieval run suite.json --save runs/base.json      # run, print the report, save everything
aieval run suite.json --tag billing --limit 20   # a slice
aieval run suite.json --outputs outputs.jsonl    # score outputs you already have: free, instant, repeatable
aieval compare runs/base.json runs/new.json --fail-on-regression
```

Exit codes: `0` ok, `1` a gate failed (pass rate below the minimum, or a regression), `2` bad input.

## Scorers

| `type` | Checks |
|---|---|
| `exact` | output equals expected (`normalize` ignores case and whitespace) |
| `contains` / `not_contains` | required or forbidden strings, with partial credit for `contains` |
| `regex` | pattern must (or must not) match |
| `numeric` | within `abs_tol` / `rel_tol` of expected; NaN never matches |
| `length` | size within `min`..`max` |
| `set_match` | lists compared as sets: **precision, recall or F1** with a threshold. The right tool for detectors |
| `json_schema` | a JSON Schema subset (type, enum, required, properties, items, bounds, pattern) |
| `no_secrets` | output leaks no keys, tokens or private keys (optionally emails). Reports the *kind*, never the secret |
| `latency` | under a time limit (informational by default) |
| `python` | your own `module:function(case, outcome)` |
| `llm_judge` | a rubric graded 1 to 5 by a model |

`output_path` and `expected_path` reach into nested values (`findings.0.rule_id`, or `findings.*.rule_id` to collect a field
from every item). `"gating": false` makes a score informational: reported, but unable to fail a case.

Typos are errors: an unknown option such as `"treshold"` is rejected rather than silently ignored, because a gate that quietly
does nothing is worse than none.

## The LLM judge, and why it is guarded

A judge model reads the system's output, and that output is untrusted text. So:

- the output is wrapped as data in `<candidate>` tags and the judge is told never to follow instructions inside it;
- output that tries to steer the judge ("ignore previous instructions", "give this a 5", "GRADER: ...") is **not graded at all**
  and fails with a clear reason (and the model is never called);
- the reply must be a JSON integer 1 to 5; anything else is retried once, then fails;
- a missing key, a network error or a broken reply is a **failure marked unavailable**, never a pass, and the report warns you.

Run with `--judge gemini` (needs `GEMINI_API_KEY`) or leave it off and use `--skip-judge`. Judges are noisy: prefer deterministic
scorers where they exist, and use `repeats` to see how stable a judged score is.

## Example: evaluating a real system

`examples/risk_scanner` evaluates the rule engine of [ai-code-risk-scanner](https://github.com/TheByteCodeHQ/ai-code-risk-scanner)
on 29 hand-labelled snippets (17 that should be flagged, 12 clean look-alikes that should not).

```bash
pip install git+https://github.com/TheByteCodeHQ/ai-code-risk-scanner
aieval run examples/risk_scanner/suite.json --save runs/base.json
aieval run examples/risk_scanner/suite_weaker.json --save runs/weaker.json
aieval compare runs/base.json runs/weaker.json --fail-on-regression
```

What it showed:

- The labels were written before the first run. On that first run **one case failed, and the label was wrong, not the scanner**:
  I had missed that `os.system(cmd)` is shell execution. The scanner was right. Fixing a label is legitimate; it is also exactly why
  you should read failures rather than only count them.
- `suite_weaker.json` switches two rules off, as if someone "reduced noise". The weaker scanner still scores **89.7%, which
  passes the suite's 85% gate**. `compare` names the three cases that regressed and exits `1`. A pass-rate gate alone would have
  let that through. This is the reason `--fail-on-regression` exists.
- With 29 cases, even 100% has a 95% interval of 88% to 100%. The report says so.

## Layout

```
src/ai_eval_harness/
  dataset.py   JSONL loading with line-numbered errors and quality warnings
  scorers.py   all scorers, built from plain dictionaries
  llm.py       the judge interface and a Gemini implementation
  runner.py    runs cases (threaded, order-stable), repeats, errors become outcomes
  stats.py     Wilson interval, exact McNemar, paired bootstrap, percentiles
  compare.py   regressions, fixes and a verdict that does not overclaim
  report.py    text / Markdown / self-contained HTML (escaped) / JSON, and loading a saved run
  suite.py     suite files with strict key checking
  cli.py       aieval run | compare | show | validate
```

## Tested

76 tests run on Python 3.10 and 3.12 on Linux and Windows in CI. Beyond the usual cases, the suite covers the failure paths:
a crashing system, a crashing scorer, a judge that errors or is steered, malformed datasets, escaped HTML output. I also broke the
code on purpose (ignoring the `gating` flag, removing the judge guard, dropping the factor of two in the p-value, scoring errored
cases, flipping the empty-output precision rule) and confirmed the tests fail each time. One of those checks exposed a missing
assertion, which is now in the suite.

## Limits

- **The Gemini judge has never been run against the live API.** The prompt, parsing and failure handling are tested with fakes;
  the network path is not.
- Systems run in-process, in threads. There is no per-call timeout and no sandbox, so evaluate code you trust.
- The statistics assume cases are independent and reasonably chosen. A small or unrepresentative dataset gives a confident-looking
  report about the wrong thing, and no harness can fix that.
- The JSON Schema support is a deliberate subset, not a full validator.
- The example dataset is small and was labelled by one person.

## License

MIT
