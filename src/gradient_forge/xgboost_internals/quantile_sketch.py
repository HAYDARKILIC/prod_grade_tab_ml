"""Weighted Quantile Sketch (Chen & Guestrin, 2016, §3.3).

Computes ε-approximate weighted quantiles of a univariate distribution,
weighting each observation by its Hessian h_i.  The rationale is that the
second-order Taylor expansion of the objective is locally a weighted MSE
with weights h_i:

        Σ 1/2 · h_i · (f(x_i) - (-g_i / h_i))²  +  constant

so candidate split points should be quantiles of x_i weighted by h_i, not
of x_i alone.  This implementation uses a simple sorted-array sketch that
is O(n log n); production XGBoost uses a streaming variant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class WeightedQuantileSketch:
    """Sorted-array weighted quantile sketch with ε-approximate output."""

    def __init__(self, eps: float = 0.03) -> None:
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must lie in (0, 1)")
        self.eps = eps

    def candidates(self, x: Array, h: Array) -> Array:
        """Return ~1/eps candidate split points along feature x."""
        x = np.asarray(x, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        if x.shape != h.shape:
            raise ValueError("x and h must share the same shape")

        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        h_sorted = h[order]

        total = float(h_sorted.sum())
        if total <= 0:
            return np.unique(x_sorted)
        cdf = np.cumsum(h_sorted) / total

        n_buckets = max(1, int(np.ceil(1.0 / self.eps)))
        targets = (np.arange(1, n_buckets) / n_buckets).astype(np.float64)
        idx = np.searchsorted(cdf, targets, side="left")
        idx = np.clip(idx, 0, x_sorted.size - 1)
        return np.unique(x_sorted[idx])
