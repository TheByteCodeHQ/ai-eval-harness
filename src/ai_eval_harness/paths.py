"""Reach into nested outputs with a dotted path such as ``findings.0.rule_id``."""

from __future__ import annotations

from typing import Any

_MISSING = object()


class PathError(KeyError):
    pass


def get_path(value: Any, path: str | None, default: Any = _MISSING) -> Any:
    """``""`` or ``None`` means the whole value. Lists are indexed by number, and ``*`` collects a field from every item."""
    if not path:
        return value
    current = [value]
    wildcard = False
    for part in path.split("."):
        nxt: list[Any] = []
        for item in current:
            if part == "*":
                if not isinstance(item, list):
                    return _fail(default, path)
                nxt.extend(item)
                wildcard = True
            elif isinstance(item, dict) and part in item:
                nxt.append(item[part])
            elif isinstance(item, list) and part.isdigit() and int(part) < len(item):
                nxt.append(item[int(part)])
            elif wildcard:
                continue                      # a field missing from one item of a collected list is skipped
            else:
                return _fail(default, path)
        current = nxt
    return current if wildcard else current[0]


def _fail(default: Any, path: str) -> Any:
    if default is _MISSING:
        raise PathError(f"Path '{path}' does not exist in the value")
    return default
