"""A suite is a small JSON file: which dataset, which system, and how to score it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .scorers import ScorerConfigError, load_callable

KEYS = {"name", "dataset", "system", "repeats", "scorers", "min_pass_rate", "description"}


@dataclass
class Suite:
    name: str
    dataset: Path
    scorers: list[dict[str, Any]]
    system: str | None = None
    repeats: int = 1
    min_pass_rate: float | None = None
    description: str = ""
    path: Path | None = field(default=None, repr=False)

    def load_system(self):
        if not self.system:
            raise ScorerConfigError("This suite names no 'system'. Add one (module:function) or score recorded outputs with --outputs")
        base = str(self.path.parent) if self.path else ""
        import sys

        if base and base not in sys.path:
            sys.path.insert(0, base)             # the system module may sit beside the suite file
        return load_callable(self.system)


def load_suite(path: Path) -> Suite:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScorerConfigError(f"Cannot read suite {path}: {exc}") from None
    except ValueError as exc:
        raise ScorerConfigError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ScorerConfigError(f"{path.name} must contain a JSON object")
    unknown = set(data) - KEYS
    if unknown:
        raise ScorerConfigError(f"{path.name}: unknown key(s) {sorted(unknown)}. Allowed: {sorted(KEYS)}")
    for required in ("name", "dataset", "scorers"):
        if required not in data:
            raise ScorerConfigError(f"{path.name}: '{required}' is required")
    repeats = data.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ScorerConfigError(f"{path.name}: 'repeats' must be a whole number of at least 1")
    gate = data.get("min_pass_rate")
    if gate is not None and (isinstance(gate, bool) or not isinstance(gate, (int, float)) or not 0 <= gate <= 1):
        raise ScorerConfigError(f"{path.name}: 'min_pass_rate' must be between 0 and 1")
    if not isinstance(data["scorers"], list):
        raise ScorerConfigError(f"{path.name}: 'scorers' must be a list")
    return Suite(str(data["name"]), (path.parent / data["dataset"]).resolve(), data["scorers"], data.get("system"), repeats, gate,
                 str(data.get("description", "")), path.resolve())
