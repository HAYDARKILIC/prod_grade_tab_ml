"""Weighted ensemble via constrained optimization.

Given out-of-fold predictions  P ∈ ℝ^{n × m}  from m base models, find
non-negative weights  w ∈ ℝ^m  with  Σ w_j = 1  that minimize a
differentiable surrogate of the chosen metric. Solved with scipy's
SLSQP.

Note on the AUC surrogate: AUC is *piecewise constant* in the weights
(it only changes when two predictions swap ranks), so its gradient is
zero almost everywhere and quasi-Newton optimizers cannot move. We
optimize the binary cross-entropy instead — Spearman-aligned with AUC
on well-calibrated base learners but smooth, with non-zero gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

Array = NDArray[np.float64]


@dataclass
class WeightedEnsemble:
    """Constrained-optimization blender of base-model predictions."""

    metric: str = "auc"          # "auc" | "mse"
    weights_: Array | None = field(default=None, init=False)

    _EPS: float = 1e-15

    def _objective(self, w: Array, P: Array, y: Array) -> float:
        blend = P @ w
        if self.metric == "auc":
            # Smooth log-loss surrogate; AUC itself is piecewise constant.
            blend = np.clip(blend, self._EPS, 1.0 - self._EPS)
            return -float(np.mean(y * np.log(blend) + (1.0 - y) * np.log(1.0 - blend)))
        if self.metric == "mse":
            return float(mean_squared_error(y, blend))
        raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, P: Array, y: Array) -> "WeightedEnsemble":
        m = P.shape[1]
        w0 = np.full(m, 1.0 / m, dtype=np.float64)
        bounds = [(0.0, 1.0)] * m
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        res = minimize(
            self._objective,
            w0,
            args=(P, y),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        self.weights_ = np.asarray(res.x, dtype=np.float64)
        return self

    def predict(self, P: Array) -> Array:
        if self.weights_ is None:
            raise RuntimeError("fit() must be called before predict()")
        return P @ self.weights_

    def report(self) -> dict[str, float]:
        if self.weights_ is None:
            raise RuntimeError("fit() must be called before report()")
        return {f"w_{i}": float(v) for i, v in enumerate(self.weights_)}
