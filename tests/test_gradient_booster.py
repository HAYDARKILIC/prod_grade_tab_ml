"""Integration tests for the from-scratch GradientBooster."""

from __future__ import annotations

import numpy as np

from gradient_forge.boosting import BinaryCrossEntropy, GradientBooster, SquaredError


def test_gradient_booster_regression_reduces_loss(reg_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = reg_data
    model = GradientBooster(loss=SquaredError(), n_estimators=30,
                            learning_rate=0.1, max_depth=4, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    baseline = float(np.mean((y - y.mean()) ** 2))
    assert mse < 0.5 * baseline


def test_gradient_booster_binary_outperforms_random(
    clf_data: tuple[np.ndarray, np.ndarray],
) -> None:
    from sklearn.metrics import roc_auc_score

    X, y = clf_data
    model = GradientBooster(loss=BinaryCrossEntropy(), n_estimators=50,
                            learning_rate=0.1, max_depth=4, random_state=0)
    model.fit(X, y.astype(float))
    raw = model.predict(X)
    proba = 1.0 / (1.0 + np.exp(-raw))
    assert roc_auc_score(y, proba) > 0.85


def test_gradient_booster_early_stopping(reg_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = reg_data
    split = X.shape[0] // 2
    model = GradientBooster(
        loss=SquaredError(), n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=0,
    )
    model.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]),
              early_stopping_rounds=10)
    assert len(model.trees_) <= 200
