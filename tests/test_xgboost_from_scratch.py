"""Sanity tests for the from-scratch XGBoost reference implementation."""

from __future__ import annotations

import numpy as np

from gradient_forge.boosting import SquaredError, XGBoostFromScratch


def test_xgboost_from_scratch_fits_simple_signal() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(scale=0.05, size=300)

    model = XGBoostFromScratch(
        loss=SquaredError(), n_estimators=80, learning_rate=0.1,
        max_depth=4, reg_lambda=1.0, random_state=0,
    ).fit(X, y)
    preds = model.predict(X)
    r2 = 1.0 - float(np.var(y - preds) / np.var(y))
    assert r2 > 0.8
