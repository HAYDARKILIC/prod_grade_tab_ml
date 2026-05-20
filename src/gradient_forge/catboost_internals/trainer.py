"""Production CatBoost wrapper with GPU support and categorical handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class CatBoostTrainer:
    """Thin wrapper around catboost.CatBoostClassifier / CatBoostRegressor."""

    task: str = "regression"
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    border_count: int = 254
    cat_features: list[int] | None = None
    gpu: bool = False
    random_state: int = 42
    extra_params: dict[str, Any] = field(default_factory=dict)

    model_: Any = field(default=None, init=False)

    def _make_model(self) -> Any:
        from catboost import CatBoostClassifier, CatBoostRegressor

        params: dict[str, Any] = dict(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            task_type="GPU" if self.gpu else "CPU",
            random_state=self.random_state,
            verbose=False,
            **self.extra_params,
        )
        if self.task == "regression":
            return CatBoostRegressor(**params)
        if self.task in {"binary", "classification"}:
            return CatBoostClassifier(eval_metric="AUC", **params)
        raise ValueError(f"Unknown task: {self.task}")

    def fit(
        self,
        X: Array,
        y: Array,
        eval_set: tuple[Array, Array] | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "CatBoostTrainer":
        self.model_ = self._make_model()
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if self.cat_features is not None:
            fit_kwargs["cat_features"] = self.cat_features
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(self.model_.predict(X), dtype=np.float64).ravel()

    def predict_proba(self, X: Array) -> Array:
        return np.asarray(self.model_.predict_proba(X), dtype=np.float64)
