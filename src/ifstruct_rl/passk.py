"""Unbiased pass@k (Chen et al. 2021)."""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k draws from n samples with c passes is a pass."""
    if n <= 0 or k <= 0:
        return 0.0
    if c < 0 or c > n:
        raise ValueError(f"c={c} must be in [0, n={n}]")
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
