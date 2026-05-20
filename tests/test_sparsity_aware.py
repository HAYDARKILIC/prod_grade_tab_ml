"""Unit tests for the sparsity-aware split finder."""

from __future__ import annotations

import numpy as np

from gradient_forge.xgboost_internals.sparsity_aware import SparsityAwareSplitFinder


def test_missing_routed_to_higher_gain_side() -> None:
    # 6 rows: 3 negative gradients on the left, 3 positive on the right.
    # Two missing rows whose gradients clearly fit on the positive side.
    X = np.array([
        [0.0], [1.0], [2.0], [4.0], [5.0], [6.0], [np.nan], [np.nan]
    ])
    g = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.9, 0.95])
    h = np.ones_like(g)
    finder = SparsityAwareSplitFinder(reg_lambda=1.0, min_child_weight=0.0)
    res = finder.find_best(X, g, h)
    assert res is not None
    # Missing rows have positive g — they should be routed right.
    assert res.default_left is False


def test_no_split_returns_none_on_constant_feature() -> None:
    X = np.full((10, 1), 1.0)
    g = np.random.default_rng(0).normal(size=10)
    h = np.ones(10)
    finder = SparsityAwareSplitFinder()
    assert finder.find_best(X, g, h) is None
