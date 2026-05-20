"""Accuracy benchmark — 5-fold CV AUC across the three production GBDTs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from gradient_forge.catboost_internals import CatBoostTrainer
from gradient_forge.data.loaders import load_synthetic_classification
from gradient_forge.lightgbm_internals import LightGBMTrainer
from gradient_forge.utils import classification_metrics, seed_everything
from gradient_forge.xgboost_internals import XGBoostTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("reports/accuracy.json"))
    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--n-features", type=int, default=30)
    p.add_argument("--cv-splits", type=int, default=5)
    return p.parse_args()


def cv_auc(trainer_cls: type, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold,
           **kwargs: object) -> float:
    aucs: list[float] = []
    for tr, va in cv.split(X, y):
        model = trainer_cls(**kwargs)
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        proba = model.predict_proba(X[va])[:, 1]
        aucs.append(classification_metrics(y[va], proba)["auc"])
    return float(np.mean(aucs))


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    X, y = load_synthetic_classification(n_samples=args.n_samples, n_features=args.n_features)
    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=42)

    results = {
        "xgboost":  {"mean_auc": cv_auc(XGBoostTrainer,  X, y, cv, task="binary",
                                        n_estimators=300, learning_rate=0.05)},
        "lightgbm": {"mean_auc": cv_auc(LightGBMTrainer, X, y, cv, task="binary",
                                        n_estimators=300, learning_rate=0.05)},
        "catboost": {"mean_auc": cv_auc(CatBoostTrainer, X, y, cv, task="binary",
                                        iterations=300, learning_rate=0.05)},
    }
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
