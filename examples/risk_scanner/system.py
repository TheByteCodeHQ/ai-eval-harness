"""The system under test: ai-code-risk-scanner's rule engine (no model), wrapped as a function the harness can call.

Install the scanner first:  pip install git+https://github.com/TheByteCodeHQ/ai-code-risk-scanner
"""

from code_risk_scanner.scanner import scan


def _diff(path: str, lines: list[str]) -> str:
    header = [f"diff --git a/{path} b/{path}", "new file mode 100644", "index 0000000..1111111", "--- /dev/null", f"+++ b/{path}",
              f"@@ -0,0 +1,{len(lines)} @@"]
    return "\n".join(header + [f"+{line}" for line in lines]) + "\n"


def scan_snippet(case_input: dict) -> dict:
    """Input is {"path": ..., "lines": [...]}. Only content rules are reported: the change-shape rules (CHG*) judge a whole
    merge request, which a three-line snippet is not."""
    result = scan(_diff(case_input["path"], case_input["lines"]))
    return {"findings": [{"rule_id": f.rule_id, "line": f.line, "severity": str(f.severity)} for f in result.findings
                         if not f.rule_id.startswith("CHG")]}
