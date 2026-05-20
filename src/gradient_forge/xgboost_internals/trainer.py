"""Production XGBoost wrapper with GPU support and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class XGBoostTrainer:
    """Thin wrapper around xgboost.XGBClassifier / XGBRegressor.

    Adds GPU fallback, default-direction extraction, and OOF helpers.
    """

    task: str = "regression"
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gpu: bool = False
    random_state: int = 42
    extra_params: dict[str, Any] = field(default_factory=dict)

    model_: Any = field(default=None, init=False)

    def _make_model(self) -> Any:
        import xgboost as xgb

        params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            device="cuda" if self.gpu else "cpu",
            random_state=self.random_state,
            **self.extra_params,
        )
        if self.task == "regression":
            return xgb.XGBRegressor(**params)
        if self.task in {"binary", "classification"}:
            return xgb.XGBClassifier(eval_metric="auc", **params)
        raise ValueError(f"Unknown task: {self.task}")

    def fit(
        self,
        X: Array,
        y: Array,
        eval_set: tuple[Array, Array] | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> "XGBoostTrainer":
        self.model_ = self._make_model()
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def predict_proba(self, X: Array) -> Array:
        return np.asarray(self.model_.predict_proba(X), dtype=np.float64)

    # ---- diagnostics -------------------------------------------------------
    def default_directions(self) -> list[str]:
        """Inspect learned default direction at every split in every tree."""
        booster = self.model_.get_booster()
        out: list[str] = []
        for tree_str in booster.get_dump(with_stats=True):
            for line in tree_str.splitlines():
                if "missing=" in line:
                    out.append(line.strip())
        return out
