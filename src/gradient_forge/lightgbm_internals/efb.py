"""Exclusive Feature Bundling (Ke et al., 2017, §4.2).

EFB groups *mutually exclusive* sparse features into a single dense
"bundle".  Two features are exclusive if they rarely take non-zero values
at the same row; one-hot encoded categorical columns are the canonical
example.  The greedy bundling problem is equivalent to graph coloring on
the conflict graph, which is NP-hard — but a greedy approximation
(sort by degree, place into the first bundle that does not exceed a
conflict budget) is sufficient in practice.

The bundled values must be remapped so that the offsets prevent two
features in the same bundle from overlapping their value ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FeatureBundle:
    feature_ids: list[int]
    offsets: list[float]
    ranges: list[float]


class ExclusiveFeatureBundling:
    """Greedy approximate EFB."""

    def __init__(self, max_conflict_rate: float = 0.0) -> None:
        if max_conflict_rate < 0.0 or max_conflict_rate > 1.0:
            raise ValueError("max_conflict_rate must lie in [0, 1]")
        self.max_conflict_rate = max_conflict_rate
        self.bundles_: list[FeatureBundle] = []

    @staticmethod
    def _conflict(a: NDArray[np.float64], b: NDArray[np.float64]) -> int:
        return int(np.sum((a != 0) & (b != 0)))

    def fit(self, X: NDArray[np.float64]) -> "ExclusiveFeatureBundling":
        n, d = X.shape
        max_conflict = int(self.max_conflict_rate * n)

        # sort by descending number of non-zeros (degree proxy)
        nnz = (X != 0).sum(axis=0)
        order = np.argsort(-nnz, kind="stable")

        bundles: list[FeatureBundle] = []
        for j in order:
            placed = False
            for bundle in bundles:
                conflict = sum(
                    self._conflict(X[:, j], X[:, k]) for k in bundle.feature_ids
                )
                if conflict <= max_conflict:
                    offset = sum(bundle.ranges)
                    rng = float(X[:, j].max() - X[:, j].min() + 1e-9)
                    bundle.feature_ids.append(int(j))
                    bundle.offsets.append(offset)
                    bundle.ranges.append(rng)
                    placed = True
                    break
            if not placed:
                rng = float(X[:, j].max() - X[:, j].min() + 1e-9)
                bundles.append(
                    FeatureBundle(feature_ids=[int(j)], offsets=[0.0], ranges=[rng])
                )
        self.bundles_ = bundles
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.bundles_:
            raise RuntimeError("fit() must be called before transform()")
        n = X.shape[0]
        out = np.zeros((n, len(self.bundles_)), dtype=np.float64)
        for b_idx, bundle in enumerate(self.bundles_):
            for f_idx, off in zip(bundle.feature_ids, bundle.offsets, strict=True):
                vals = X[:, f_idx]
                mask = vals != 0
                out[mask, b_idx] = vals[mask] + off
        return out

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit(X).transform(X)
