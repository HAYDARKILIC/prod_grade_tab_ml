"""From-scratch gradient booster (Week 1 deliverable).

Implements F_t(x) = F_{t-1}(x) + η · f_t(x) with second-order Taylor-expanded
objective and L2 regularization on leaf weights.  No scikit-learn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gradient_forge.boosting.losses import LossFunction, SquaredError
from gradient_forge.boosting.tree import RegressionTree

Array = NDArray[np.float64]


@dataclass
class GradientBooster:
    """Pure-NumPy second-order gradient booster."""

    loss: LossFunction = field(default_factory=SquaredError)
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    min_samples_split: int = 2
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_gamma: float = 0.0
    subsample: float = 1.0
    random_state: int | None = None
    verbose: bool = False

    # learned state
    trees_: list[RegressionTree] = field(default_factory=list, init=False)
    init_pred_: float = field(default=0.0, init=False)
    history_: list[dict[str, float]] = field(default_factory=list, init=False)

    # ---------------------------------------------------------------- API
    def fit(
        self,
        X: Array,
        y: Array,
        eval_set: tuple[Array, Array] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "GradientBooster":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]

        self.init_pred_ = self.loss.initial_prediction(y)
        F = np.full(n, self.init_pred_, dtype=np.float64)

        if eval_set is not None:
            X_val, y_val = (np.asarray(eval_set[0], dtype=np.float64),
                            np.asarray(eval_set[1], dtype=np.float64))
            F_val = np.full(X_val.shape[0], self.init_pred_, dtype=np.float64)
        else:
            X_val = y_val = F_val = None

        best_val = np.inf
        rounds_since_best = 0

        for t in range(self.n_estimators):
            g = self.loss.gradient(y, F)
            h = self.loss.hessian(y, F)

            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(self.subsample * n), replace=False)
                Xs, gs, hs = X[idx], g[idx], h[idx]
            else:
                Xs, gs, hs = X, g, h

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_gamma=self.reg_gamma,
            ).fit(Xs, gs, hs)
            self.trees_.append(tree)

            F += self.learning_rate * tree.predict(X)
            train_loss = float(self.loss.loss(y, F).mean())

            record: dict[str, float] = {"iter": float(t), "train_loss": train_loss}

            if X_val is not None and y_val is not None and F_val is not None:
                F_val += self.learning_rate * tree.predict(X_val)
                val_loss = float(self.loss.loss(y_val, F_val).mean())
                record["val_loss"] = val_loss

                if early_stopping_rounds is not None:
                    if val_loss < best_val - 1e-9:
                        best_val = val_loss
                        rounds_since_best = 0
                    else:
                        rounds_since_best += 1
                    if rounds_since_best >= early_stopping_rounds:
                        if self.verbose:
                            print(f"[GradientBooster] early stop at round {t}")
                        self.history_.append(record)
                        break

            if self.verbose and (t % 10 == 0 or t == self.n_estimators - 1):
                msg = f"[iter {t:4d}] train={train_loss:.6f}"
                if "val_loss" in record:
                    msg += f"  val={record['val_loss']:.6f}"
                print(msg)
            self.history_.append(record)

        return self

    def predict(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred_, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X: Array) -> Any:
        """Yield predictions after each boosting round (useful for diagnostics)."""
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred_, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()
