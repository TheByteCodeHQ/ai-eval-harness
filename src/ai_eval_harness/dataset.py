"""Load and check an evaluation dataset (JSON Lines: one case per line)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import Case

MAX_CASES = 100_000
MAX_LINE_BYTES = 2_000_000


class DatasetError(ValueError):
    pass


@dataclass
class Dataset:
    cases: list[Case]
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def select(self, *, tags: list[str] | None = None, limit: int | None = None) -> list[Case]:
        chosen = [c for c in self.cases if not tags or set(tags) & set(c.tags)]
        return chosen[:limit] if limit else chosen


def _case(raw: object, line: int) -> Case:
    if not isinstance(raw, dict):
        raise DatasetError(f"line {line}: each case must be a JSON object")
    unknown = set(raw) - {"id", "input", "expected", "tags", "metadata"}
    if unknown:
        raise DatasetError(f"line {line}: unknown field(s) {sorted(unknown)}. Allowed: id, input, expected, tags, metadata")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetError(f"line {line}: 'id' must be a non-empty string")
    if "input" not in raw:
        raise DatasetError(f"line {line} ({case_id}): 'input' is required")
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        raise DatasetError(f"line {line} ({case_id}): 'tags' must be a list of non-empty strings")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"line {line} ({case_id}): 'metadata' must be an object")
    return Case(id=case_id.strip(), input=raw["input"], expected=raw.get("expected"), tags=tuple(tags), metadata=metadata)


def load_dataset(path: Path) -> Dataset:
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise DatasetError(f"line {number}: larger than {MAX_LINE_BYTES // 1_000_000} MB")
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise DatasetError(f"line {number}: not valid JSON ({exc})") from None
        cases.append(_case(raw, number))
        if len(cases) > MAX_CASES:
            raise DatasetError(f"More than {MAX_CASES} cases")
    if not cases:
        raise DatasetError(f"{path.name} contains no cases")

    warnings: list[str] = []
    duplicates = sorted(i for i, n in Counter(c.id for c in cases).items() if n > 1)
    if duplicates:
        raise DatasetError(f"Duplicate case id(s): {', '.join(duplicates[:5])}")
    same_input = Counter(json.dumps(c.input, sort_keys=True, default=str) for c in cases)
    repeated = sum(1 for n in same_input.values() if n > 1)
    if repeated:
        warnings.append(f"{repeated} input(s) appear in more than one case. Duplicates inflate scores and can hide a weak spot")
    if all(c.expected is None for c in cases):
        warnings.append("No case has an 'expected' value, so only scorers that need no reference can be used")
    untagged = sum(1 for c in cases if not c.tags)
    if untagged == len(cases):
        warnings.append("No case has tags. Tags are how results are broken down, so a weak area stays visible")
    return Dataset(cases, warnings)
