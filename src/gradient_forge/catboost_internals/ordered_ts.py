"""Ordered Target Statistics (Prokhorenkova et al., 2018, §3-4).

Naive target encoding leaks because the encoding of row i uses y_i.
CatBoost fixes this with a random permutation σ of the rows and uses,
for every row i, only the rows j that appear *before* i in σ:

        x̂_{σ(i), j} =
              ( Σ_{k=1..i-1} 𝟙{x_{σ(k), j} = x_{σ(i), j}} · y_{σ(k)}  +  a · prior )
            / ( Σ_{k=1..i-1} 𝟙{x_{σ(k), j} = x_{σ(i), j}}             +  a         )

The prior is typically the global mean of y, and  a > 0  is a smoothing
parameter that controls bias for rare categories.  At inference time the
entire training set is available, so the standard target statistic is
used.

This module is *didactic*: it shows the unbiased encoding and contrasts
it with mean-target encoding so the leakage can be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class OrderedTargetStatistics:
    """Ordered target statistic encoder."""

    smoothing: float = 1.0          # ``a`` in the formula above
    random_state: int | None = None
    prior_: float = field(default=0.0, init=False)
    stats_: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        x_cat: NDArray[np.object_],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Encode training rows in a single permuted pass.

        Stores per-category (Σy, n) over the *whole* training set after
        the ordered pass, for use at inference time.
        """
        rng = np.random.default_rng(self.random_state)
        n = x_cat.size
        perm = rng.permutation(n)
        self.prior_ = float(np.mean(y))

        running: dict[str, list[float]] = {}   # cat -> [Σy_so_far, n_so_far]
        encoded = np.empty(n, dtype=np.float64)

        for pos in perm:
            cat = str(x_cat[pos])
            s, c = running.get(cat, [0.0, 0.0])
            encoded[pos] = (s + self.smoothing * self.prior_) / (c + self.smoothing)
            running[cat] = [s + float(y[pos]), c + 1.0]

        # final stats over the whole training set for inference use
        self.stats_ = {cat: (s, c) for cat, (s, c) in running.items()}
        return encoded

    def transform(self, x_cat: NDArray[np.object_]) -> NDArray[np.float64]:
        out = np.empty(x_cat.size, dtype=np.float64)
        for i, raw in enumerate(x_cat):
            cat = str(raw)
            s, c = self.stats_.get(cat, (0.0, 0.0))
            out[i] = (s + self.smoothing * self.prior_) / (c + self.smoothing)
        return out


def naive_mean_target_encode(
    x_cat: NDArray[np.object_], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Leaky mean-target encoding for didactic comparison.

    Replaces every level by  E[y | x = level]  computed on the *same*
    rows being encoded.  Used in notebook 04 to measure the train–val
    gap induced by leakage.
    """
    n = x_cat.size
    out = np.empty(n, dtype=np.float64)
    for cat in np.unique(x_cat):
        mask = x_cat == cat
        out[mask] = float(y[mask].mean())
    return out
