"""Compare two runs of the same suite: did the change help, hurt, or is the difference just noise?"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import RunResult
from .stats import bootstrap_mean_diff, mcnemar_exact

ALPHA = 0.05


@dataclass
class Comparison:
    baseline: str
    candidate: str
    shared: int
    fixed: list[str] = field(default_factory=list)          # failed in baseline, pass in candidate
    regressed: list[str] = field(default_factory=list)      # passed in baseline, fail in candidate
    baseline_rate: float = 0.0
    candidate_rate: float = 0.0
    score_diff: float = 0.0
    score_low: float = 0.0
    score_high: float = 0.0
    p_value: float = 1.0
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def verdict(self) -> str:
        if not self.shared:
            return "no comparable cases"
        if self.regressed and (not self.fixed or (self.significant and len(self.regressed) > len(self.fixed))):
            return "worse" if self.significant else "regressions, not proven"
        if self.fixed and not self.regressed:
            return "better" if self.significant else "improved, not proven"
        if not self.fixed and not self.regressed:
            return "no change"
        return "mixed" if not self.significant else ("better" if len(self.fixed) > len(self.regressed) else "worse")


def compare_runs(baseline: RunResult, candidate: RunResult) -> Comparison:
    base = {r.case.id: r for r in baseline.results}
    cand = {r.case.id: r for r in candidate.results}
    shared = [cid for cid in base if cid in cand]
    out = Comparison(baseline.system, candidate.system, len(shared))
    out.skipped = sorted(set(base) ^ set(cand))
    if out.skipped:
        out.warnings.append(f"{len(out.skipped)} case(s) are in only one run and were left out of the comparison")
    if baseline.suite != candidate.suite:
        out.warnings.append(f"The runs are of different suites ('{baseline.suite}' and '{candidate.suite}'), so the comparison may not be fair")
    if not shared:
        return out
    for cid in shared:
        b, c = base[cid].passed, cand[cid].passed
        if c and not b:
            out.fixed.append(cid)
        elif b and not c:
            out.regressed.append(cid)
    out.baseline_rate = sum(base[c].passed for c in shared) / len(shared)
    out.candidate_rate = sum(cand[c].passed for c in shared) / len(shared)
    out.p_value = mcnemar_exact(len(out.regressed), len(out.fixed))
    out.score_diff, out.score_low, out.score_high = bootstrap_mean_diff([base[c].value for c in shared], [cand[c].value for c in shared])
    if len(shared) < 20:
        out.warnings.append(f"Only {len(shared)} shared cases. Differences this small are rarely trustworthy, whatever the p-value says")
    return out
