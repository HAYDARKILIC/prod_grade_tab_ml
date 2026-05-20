"""Smoke tests for the out-of-fold Stacker."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from gradient_forge.ensemble.stacker import Stacker


def test_stacker_produces_meta_features(clf_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = clf_data
    bases = [
        GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=0),
        LogisticRegression(max_iter=200),
    ]
    stk = Stacker(base_models=bases, task="binary", n_splits=3, random_state=0)
    stk.fit(X, y)
    assert stk.meta_features_ is not None
    assert stk.meta_features_.shape == (X.shape[0], 2)
    proba = stk.predict(X)
    assert roc_auc_score(y, proba) > 0.7
