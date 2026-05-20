"""Out-of-fold stacking.

Generates a leak-free meta-feature matrix:

    for fold k = 1..K:
        train each base model on folds ≠ k
        store predictions on fold k as the meta-feature for that block

The meta-learner (default: LogisticRegression / Ridge) is then trained
on the assembled (n × m) meta-features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

Array = NDArray[np.float64]


@dataclass
class Stacker:
    """Out-of-fold stacking with a configurable meta-learner."""

    base_models: list[Any]
    task: str = "binary"
    n_splits: int = 5
    random_state: int = 42
    meta_learner: Any | None = None

    meta_features_: Array | None = field(default=None, init=False)
    fitted_bases_: list[Any] = field(default_factory=list, init=False)
    meta_: Any | None = field(default=None, init=False)

    def _make_meta(self) -> Any:
        if self.meta_learner is not None:
            return self.meta_learner
        if self.task == "binary":
            return LogisticRegression(max_iter=1000)
        return Ridge(alpha=1.0)

    def fit(self, X: Array, y: Array) -> "Stacker":
        n = X.shape[0]
        m = len(self.base_models)
        oof = np.zeros((n, m), dtype=np.float64)

        if self.task == "binary":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                 random_state=self.random_state)
            splits = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True,
                       random_state=self.random_state)
            splits = cv.split(X)

        for tr, va in splits:
            for j, model in enumerate(self.base_models):
                m_copy = self._clone(model)
                m_copy.fit(X[tr], y[tr])
                if self.task == "binary":
                    oof[va, j] = m_copy.predict_proba(X[va])[:, 1]
                else:
                    oof[va, j] = m_copy.predict(X[va])

        # refit each base model on all data for inference
        self.fitted_bases_ = []
        for model in self.base_models:
            m_copy = self._clone(model)
            m_copy.fit(X, y)
            self.fitted_bases_.append(m_copy)

        self.meta_features_ = oof
        self.meta_ = self._make_meta()
        self.meta_.fit(oof, y)
        return self

    @staticmethod
    def _clone(model: Any) -> Any:
        """Shallow clone: re-instantiate with the same params."""
        from copy import deepcopy
        return deepcopy(model)

    def predict(self, X: Array) -> Array:
        if self.meta_ is None:
            raise RuntimeError("fit() must be called before predict()")
        m = len(self.fitted_bases_)
        P = np.zeros((X.shape[0], m), dtype=np.float64)
        for j, model in enumerate(self.fitted_bases_):
            if self.task == "binary":
                P[:, j] = model.predict_proba(X)[:, 1]
            else:
                P[:, j] = model.predict(X)
        if self.task == "binary":
            return self.meta_.predict_proba(P)[:, 1]
        return self.meta_.predict(P)
