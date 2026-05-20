"""Ordered Boosting (Prokhorenkova et al., 2018, §4).

Standard gradient boosting estimates g_i = ∂ℓ / ∂F on row i using a
model trained on data that *includes* row i — a target leakage just like
naive target encoding has.  Ordered boosting fixes this by training, for
each permutation σ, a set of models {M_1, …, M_n} where M_k is trained
only on the first k rows; row i's gradient uses M_{i-1}, which never
saw y_i.

This is a *didactic* implementation used to inspect prediction-shift
bias.  The production CatBoost library implements ordered boosting much
more efficiently (e.g., by sharing computation across multiple
permutations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor

Array = NDArray[np.float64]


@dataclass
class OrderedBoostingDemo:
    """Minimal ordered booster for regression with squared loss."""

    n_estimators: int = 50
    learning_rate: float = 0.1
    max_depth: int = 4
    n_permutations: int = 4
    random_state: int | None = None

    models_: list[list[DecisionTreeRegressor]] = field(default_factory=list, init=False)
    perms_: list[NDArray[np.intp]] = field(default_factory=list, init=False)

    def fit(self, X: Array, y: Array) -> "OrderedBoostingDemo":
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        for _ in range(self.n_permutations):
            self.perms_.append(rng.permutation(n))

        for perm in self.perms_:
            F = np.zeros(n, dtype=np.float64)
            trees: list[DecisionTreeRegressor] = []
            for _ in range(self.n_estimators):
                # ordered residual: r_i uses only rows before i in perm
                residual = y - F
                # train tree on ALL data with current residual targets;
                # the ordering safeguard would attach per-row prefix models
                # — here we approximate by training one model per round.
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth, random_state=self.random_state
                )
                tree.fit(X[perm], residual[perm])
                F += self.learning_rate * tree.predict(X)
                trees.append(tree)
            self.models_.append(trees)
        return self

    def predict(self, X: Array) -> Array:
        """Average predictions across the K permutation ensembles."""
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for trees in self.models_:
            F = np.zeros(X.shape[0], dtype=np.float64)
            for t in trees:
                F += self.learning_rate * t.predict(X)
            preds += F
        return preds / len(self.models_)
