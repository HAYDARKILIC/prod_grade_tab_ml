"""Loss functions for gradient boosting.

Each loss defines:
    - L(y, p):      the per-instance loss
    - gradient(y, p):  g_i = ∂L/∂p
    - hessian(y, p):   h_i = ∂²L/∂p²

These are exactly the quantities that appear in the second-order Taylor
expansion of the regularized objective used by XGBoost-style boosting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class LossFunction(ABC):
    """Abstract base class for differentiable loss functions."""

    @abstractmethod
    def loss(self, y: Array, pred: Array) -> Array:
        """Per-instance loss L(y_i, p_i)."""

    @abstractmethod
    def gradient(self, y: Array, pred: Array) -> Array:
        """First derivative g_i = ∂L/∂p evaluated at p_i."""

    @abstractmethod
    def hessian(self, y: Array, pred: Array) -> Array:
        """Second derivative h_i = ∂²L/∂p² evaluated at p_i."""

    @abstractmethod
    def initial_prediction(self, y: Array) -> float:
        """Optimal constant predictor F_0(x) = argmin_c Σ L(y_i, c)."""


class SquaredError(LossFunction):
    """L(y, p) = 1/2 · (y - p)²

    Gradient :  g_i = p_i - y_i
    Hessian  :  h_i = 1
    F_0(x)   :  mean(y)
    """

    def loss(self, y: Array, pred: Array) -> Array:
        return 0.5 * (y - pred) ** 2

    def gradient(self, y: Array, pred: Array) -> Array:
        return pred - y

    def hessian(self, y: Array, pred: Array) -> Array:
        return np.ones_like(y, dtype=np.float64)

    def initial_prediction(self, y: Array) -> float:
        return float(np.mean(y))


class BinaryCrossEntropy(LossFunction):
    """Logistic loss for binary classification.

    Working in raw-score space  p ∈ ℝ , let σ(p) = 1 / (1 + e⁻ᵖ).

        L(y, p) = -y · log σ(p) - (1 - y) · log(1 - σ(p))
        g_i     =  σ(p_i) - y_i
        h_i     =  σ(p_i) · (1 - σ(p_i))
        F_0(x)  =  log( p̄ / (1 - p̄) )    where p̄ = mean(y)

    Note: ``pred`` is the raw additive logit, not a probability.
    """

    _EPS: float = 1e-15

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, y: Array, pred: Array) -> Array:
        p = np.clip(self._sigmoid(pred), self._EPS, 1.0 - self._EPS)
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    def gradient(self, y: Array, pred: Array) -> Array:
        return self._sigmoid(pred) - y

    def hessian(self, y: Array, pred: Array) -> Array:
        p = self._sigmoid(pred)
        return p * (1.0 - p)

    def initial_prediction(self, y: Array) -> float:
        p_bar = float(np.clip(np.mean(y), self._EPS, 1.0 - self._EPS))
        return float(np.log(p_bar / (1.0 - p_bar)))
