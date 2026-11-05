"""Small, dependency-free statistics. A pass rate from 30 cases is not a precise number, so say how imprecise it is."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate. Unlike the textbook formula it behaves at 0% and 100% and with few cases."""
    if total <= 0:
        return (0.0, 1.0)
    p = passed / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value for 'A and B pass different cases'. only_a = A passed and B failed; only_b = the reverse."""
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(only_a, only_b) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], *, rounds: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Paired bootstrap of mean(b - a). Returns (difference, low, high) for a 95% interval. The seed makes reports repeatable."""
    if len(a) != len(b) or not a:
        raise ValueError("Paired samples must be the same non-zero length")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return (mean, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1])


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
