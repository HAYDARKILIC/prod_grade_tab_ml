"""Production LightGBM wrapper with GPU support and benchmarking hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class LightGBMTrainer:
    """Thin wrapper around lightgbm.LGBMClassifier / LGBMRegressor."""

    task: str = "regression"
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    boosting_type: str = "gbdt"
    gpu: bool = False
    random_state: int = 42
    extra_params: dict[str, Any] = field(default_factory=dict)

    model_: Any = field(default=None, init=False)

    def _make_model(self) -> Any:
        import lightgbm as lgb

        params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            boosting_type=self.boosting_type,
            device="gpu" if self.gpu else "cpu",
            random_state=self.random_state,
            verbose=-1,
            **self.extra_params,
        )
        if self.task == "regression":
            return lgb.LGBMRegressor(**params)
        if self.task in {"binary", "classification"}:
            return lgb.LGBMClassifier(**params)
        raise ValueError(f"Unknown task: {self.task}")

    def fit(
        self,
        X: Array,
        y: Array,
        eval_set: tuple[Array, Array] | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "LightGBMTrainer":
        import lightgbm as lgb

        self.model_ = self._make_model()
        callbacks = []
        if early_stopping_rounds is not None and eval_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = callbacks
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def predict_proba(self, X: Array) -> Array:
        return np.asarray(self.model_.predict_proba(X), dtype=np.float64)
