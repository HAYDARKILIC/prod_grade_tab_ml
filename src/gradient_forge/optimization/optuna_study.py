"""Unified Optuna study driver for XGBoost / LightGBM / CatBoost.

Uses the Tree-structured Parzen Estimator (TPE) sampler by default.
For task t, TPE models  p(θ | y < y*)  and  p(θ | y ≥ y*)  and proposes
new candidates by maximizing  ℓ(θ) / g(θ).  This is far more sample-
efficient than grid or random search whenever the response surface has
exploitable structure — which it almost always does in GBDT tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import optuna
from numpy.typing import NDArray
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

Array = NDArray[np.float64]

ObjectiveFn = Callable[[optuna.Trial, Array, Array], float]


@dataclass
class OptunaStudy:
    """High-level wrapper for tuning a single algorithm."""

    algorithm: str = "xgboost"          # "xgboost" | "lightgbm" | "catboost"
    task: str = "binary"                # "binary" | "regression"
    n_trials: int = 100
    timeout: int | None = None
    cv_splits: int = 5
    gpu: bool = False
    random_state: int = 42
    pruner: BasePruner = field(default_factory=lambda: MedianPruner(n_warmup_steps=10))
    storage: str | None = None
    study_name: str | None = None
    direction: str = "maximize"

    study_: optuna.Study | None = field(default=None, init=False)

    def _make_sampler(self) -> TPESampler:
        return TPESampler(seed=self.random_state, multivariate=True, group=True)

    def _objective(self, trial: optuna.Trial, X: Array, y: Array) -> float:
        from gradient_forge.optimization.search_spaces import build_search_space

        params = build_search_space(self.algorithm, trial, gpu=self.gpu)

        if self.task == "binary":
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True,
                                 random_state=self.random_state)
            scoring = roc_auc_score
        else:
            cv = KFold(n_splits=self.cv_splits, shuffle=True,
                       random_state=self.random_state)
            scoring = lambda yt, yp: -mean_squared_error(yt, yp)  # noqa: E731

        scores: list[float] = []
        for fold, (tr, va) in enumerate(cv.split(X, y if self.task == "binary" else None)):
            model = self._instantiate(params)
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            preds = model.predict_proba(X_va)[:, 1] if self.task == "binary" else model.predict(X_va)
            score = float(scoring(y_va, preds))
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            scores.append(score)
        return float(np.mean(scores))

    def _instantiate(self, params: dict[str, Any]) -> Any:
        if self.algorithm == "xgboost":
            from gradient_forge.xgboost_internals import XGBoostTrainer
            return XGBoostTrainer(task=self.task, gpu=self.gpu, **params)
        if self.algorithm == "lightgbm":
            from gradient_forge.lightgbm_internals import LightGBMTrainer
            return LightGBMTrainer(task=self.task, gpu=self.gpu, **params)
        if self.algorithm == "catboost":
            from gradient_forge.catboost_internals import CatBoostTrainer
            return CatBoostTrainer(task=self.task, gpu=self.gpu, **params)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def optimize(self, X: Array, y: Array) -> optuna.Study:
        self.study_ = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.storage is not None,
            direction=self.direction,
            sampler=self._make_sampler(),
            pruner=self.pruner,
        )
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        return self.study_
