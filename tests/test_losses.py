"""Unit tests for loss functions and their derivatives."""

from __future__ import annotations

import numpy as np
import pytest

from gradient_forge.boosting.losses import BinaryCrossEntropy, SquaredError


def _finite_diff_grad(loss_fn, y: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    plus = loss_fn.loss(y, pred + eps)
    minus = loss_fn.loss(y, pred - eps)
    return (plus - minus) / (2.0 * eps)


def _finite_diff_hess(loss_fn, y: np.ndarray, pred: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    plus = loss_fn.loss(y, pred + eps)
    base = loss_fn.loss(y, pred)
    minus = loss_fn.loss(y, pred - eps)
    return (plus - 2.0 * base + minus) / (eps ** 2)


def test_squared_error_gradient_matches_finite_diff() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=64)
    pred = rng.normal(size=64)
    loss = SquaredError()
    g_analytic = loss.gradient(y, pred)
    g_numeric = _finite_diff_grad(loss, y, pred)
    np.testing.assert_allclose(g_analytic, g_numeric, atol=1e-4)


def test_squared_error_hessian_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([0.5, 1.5, 2.5])
    h = SquaredError().hessian(y, pred)
    np.testing.assert_allclose(h, np.ones_like(h))


def test_squared_error_initial_prediction_is_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert SquaredError().initial_prediction(y) == pytest.approx(2.5)


def test_binary_crossentropy_gradient_matches_finite_diff() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=64).astype(np.float64)
    pred = rng.normal(size=64)
    loss = BinaryCrossEntropy()
    g_analytic = loss.gradient(y, pred)
    g_numeric = _finite_diff_grad(loss, y, pred)
    np.testing.assert_allclose(g_analytic, g_numeric, atol=1e-4)


def test_binary_crossentropy_hessian_matches_finite_diff() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=32).astype(np.float64)
    pred = rng.normal(scale=0.5, size=32)
    loss = BinaryCrossEntropy()
    h_analytic = loss.hessian(y, pred)
    h_numeric = _finite_diff_hess(loss, y, pred)
    np.testing.assert_allclose(h_analytic, h_numeric, atol=1e-2)


def test_binary_crossentropy_initial_prediction_is_logit_of_mean() -> None:
    y = np.array([0.0, 1.0, 1.0, 1.0])  # mean = 0.75
    expected = np.log(0.75 / 0.25)
    assert BinaryCrossEntropy().initial_prediction(y) == pytest.approx(expected)
