"""The vocabulary of an evaluation: cases go in, outputs come out, scorers judge them, and results add up."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    expected: Any = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """What the system under test returned for one case, or why it did not."""

    value: Any = None
    error: str | None = None
    latency_s: float = 0.0
    usage: dict[str, float] = field(default_factory=dict)      # optional: tokens, cost, anything numeric the system reports

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class SystemResult:
    """Return this from a system that also wants to report usage (tokens, cost). Returning a bare value works too."""

    value: Any
    usage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    scorer: str
    value: float                 # 0.0 to 1.0
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    gating: bool = True          # an informational score is reported but cannot fail a case


@dataclass
class Attempt:
    outcome: Outcome
    scores: list[Score]

    @property
    def passed(self) -> bool:
        return self.outcome.ok and all(s.passed for s in self.scores if s.gating)

    @property
    def value(self) -> float:
        """Mean of the scores. A system that errored scores zero, so errors cannot hide in an average."""
        if not self.outcome.ok or not self.scores:
            return 0.0
        return sum(s.value for s in self.scores) / len(self.scores)


@dataclass
class CaseResult:
    case: Case
    attempts: list[Attempt]

    @property
    def pass_rate(self) -> float:
        return sum(a.passed for a in self.attempts) / len(self.attempts) if self.attempts else 0.0

    @property
    def passed(self) -> bool:
        """A case passes only if every repeat passed. For a system that should be reliable, one bad sample is a failure."""
        return bool(self.attempts) and all(a.passed for a in self.attempts)

    @property
    def value(self) -> float:
        return sum(a.value for a in self.attempts) / len(self.attempts) if self.attempts else 0.0

    @property
    def inconsistent(self) -> bool:
        """Some repeats passed and some failed: the system gave different answers to the same input."""
        return len({a.passed for a in self.attempts}) > 1


@dataclass
class RunResult:
    suite: str
    system: str
    repeats: int
    results: list[CaseResult]
    config: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def tags(self) -> list[str]:
        return sorted({t for r in self.results for t in r.case.tags})
