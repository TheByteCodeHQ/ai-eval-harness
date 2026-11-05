"""A deliberately worse candidate: the same scanner with two rules switched off, as if someone 'reduced noise'."""

from system import scan_snippet

SWITCHED_OFF = {"SEC004", "INJ001"}


def scan_snippet_quieter(case_input: dict) -> dict:
    result = scan_snippet(case_input)
    return {"findings": [f for f in result["findings"] if f["rule_id"] not in SWITCHED_OFF]}
