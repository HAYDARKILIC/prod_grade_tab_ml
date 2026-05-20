"""Unit tests for the from-scratch regression tree."""

from __future__ import annotations

import numpy as np

from gradient_forge.boosting.losses import SquaredError
from gradient_forge.boosting.tree import RegressionTree


def test_tree_overfits_small_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2]
    loss = SquaredError()
    g = loss.gradient(y, np.zeros_like(y))   # = -y
    h = loss.hessian(y, np.zeros_like(y))    # = 1
    tree = RegressionTree(max_depth=10, reg_lambda=0.0, min_child_weight=0.0).fit(X, g, h)
    preds = tree.predict(X)
    # Leaf weight = -ΣG / (ΣH + λ) = Σy / n.  A deep unregularized tree fits y closely.
    mse = float(np.mean((preds - y) ** 2))
    assert mse < 1e-3


def test_tree_leaf_weight_formula() -> None:
    # With a single leaf containing every point: w* = -ΣG / (ΣH + λ)
    X = np.array([[0.0], [1.0], [2.0]])
    g = np.array([1.0, 2.0, 3.0])
    h = np.array([1.0, 1.0, 1.0])
    tree = RegressionTree(max_depth=0, reg_lambda=2.0).fit(X, g, h)
    assert tree.root is not None and tree.root.is_leaf
    expected = -g.sum() / (h.sum() + 2.0)
    assert tree.root.value == expected
