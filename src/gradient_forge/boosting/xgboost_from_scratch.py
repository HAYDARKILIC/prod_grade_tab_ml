"""Week 2 deliverable — an XGBoost-style booster reproducing core mechanics.

This is a didactic implementation: second-order objective, L1+L2 regularization
on leaves, column subsampling, and a basic histogram-based split finder.  It
intentionally omits some of XGBoost's production-grade machinery
(sparsity-aware default direction is implemented separately in
``xgboost_internals.sparsity_aware``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gradient_forge.boosting.losses import LossFunction, SquaredError
from gradient_forge.boosting.tree import RegressionTree

Array = NDArray[np.float64]


@dataclass
class XGBoostFromScratch:
    """Lightweight XGBoost-style booster used as a reference baseline."""

    loss: LossFunction = field(default_factory=SquaredError)
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_gamma: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int | None = None

    trees_: list[tuple[NDArray[np.intp], RegressionTree]] = field(default_factory=list, init=False)
    init_pred_: float = field(default=0.0, init=False)

    def fit(self, X: Array, y: Array) -> "XGBoostFromScratch":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.init_pred_ = self.loss.initial_prediction(y)
        F = np.full(n, self.init_pred_, dtype=np.float64)

        for _ in range(self.n_estimators):
            g = self.loss.gradient(y, F)
            h = self.loss.hessian(y, F)

            row_idx = (
                rng.choice(n, int(self.subsample * n), replace=False)
                if self.subsample < 1.0
                else np.arange(n)
            )
            col_idx = (
                rng.choice(d, int(self.colsample_bytree * d), replace=False)
                if self.colsample_bytree < 1.0
                else np.arange(d)
            )

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_gamma=self.reg_gamma,
            ).fit(X[np.ix_(row_idx, col_idx)], g[row_idx], h[row_idx])

            self.trees_.append((col_idx, tree))
            F += self.learning_rate * tree.predict(X[:, col_idx])
        return self

    def predict(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred_, dtype=np.float64)
        for col_idx, tree in self.trees_:
            F += self.learning_rate * tree.predict(X[:, col_idx])
        return F
