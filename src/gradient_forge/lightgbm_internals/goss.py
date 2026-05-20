"""Gradient-based One-Side Sampling (Ke et al., 2017, §4.1).

Keep the top-a fraction of instances by |g_i| (large-gradient, hence
poorly-fit), and randomly sample b · (1 - a) of the remainder.  The
small-gradient sample's gradients/Hessians are amplified by
(1 - a) / b  to keep the population estimate unbiased.

Pseudo-code::

    sort instances by |g_i| descending
    A = first a · n indices                          # top
    B = sample b · (1 - a) · n from the rest        # random small-grad
    multiply gradients and Hessians on B by (1 - a) / b
    use union(A, B) for the next iteration's histogram

This trades a small variance penalty for an O(1 / (a + b)) speed-up in
histogram construction, the dominant cost of GBDT training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GradientBasedOneSideSampling:
    """GOSS sampler reproducing LightGBM's sampling behaviour."""

    top_rate: float = 0.2     # a — fraction of large-gradient kept
    other_rate: float = 0.1   # b — fraction of small-gradient sampled
    random_state: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.top_rate < 1.0):
            raise ValueError("top_rate must lie in (0, 1)")
        if not (0.0 < self.other_rate < 1.0):
            raise ValueError("other_rate must lie in (0, 1)")
        if self.top_rate + self.other_rate >= 1.0:
            raise ValueError("top_rate + other_rate must be < 1")

    def sample(
        self,
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """Return (selected indices, reweighted g, reweighted h)."""
        rng = np.random.default_rng(self.random_state)
        n = gradients.size

        abs_g = np.abs(gradients)
        sorted_idx = np.argsort(-abs_g, kind="stable")  # descending |g|

        top_n = int(self.top_rate * n)
        other_pool_n = n - top_n
        other_n = int(self.other_rate * n)

        top_idx = sorted_idx[:top_n]
        other_pool = sorted_idx[top_n:]
        if other_n > 0 and other_pool_n > 0:
            other_idx = rng.choice(other_pool, size=min(other_n, other_pool_n), replace=False)
        else:
            other_idx = np.empty(0, dtype=np.intp)

        selected = np.concatenate([top_idx, other_idx])
        amplify = (1.0 - self.top_rate) / max(self.other_rate, 1e-12)
        g_out = gradients.copy()
        h_out = hessians.copy()
        g_out[other_idx] *= amplify
        h_out[other_idx] *= amplify
        return selected, g_out[selected], h_out[selected]
