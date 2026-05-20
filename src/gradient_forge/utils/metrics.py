"""Standard metric bundles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

Array = NDArray[np.float64]


def regression_metrics(y_true: Array, y_pred: Array) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: Array, proba: Array, threshold: float = 0.5) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "logloss": float(log_loss(y_true, np.clip(proba, 1e-15, 1.0 - 1e-15))),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
    }
