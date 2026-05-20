"""End-to-end capstone pipeline (Week 6 deliverable).

Usage::

    python scripts/run_pipeline.py --config scripts/configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from gradient_forge.catboost_internals import CatBoostTrainer
from gradient_forge.data.loaders import load_synthetic_classification
from gradient_forge.ensemble import WeightedEnsemble
from gradient_forge.lightgbm_internals import LightGBMTrainer
from gradient_forge.utils import (
    classification_metrics,
    load_yaml,
    seed_everything,
)
from gradient_forge.xgboost_internals import XGBoostTrainer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("gradient_forge.pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GradientForge end-to-end pipeline")
    parser.add_argument("--config", type=str, default="scripts/configs/pipeline.yaml")
    return parser.parse_args()


def load_data(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = cfg["data"]["source"]
    if source == "synthetic":
        return load_synthetic_classification(random_state=cfg["pipeline"]["random_state"])
    if source == "csv":
        import pandas as pd
        df = pd.read_csv(cfg["data"]["path"])
        target = cfg["data"]["target"]
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols].to_numpy(dtype=np.float64)
        y = df[target].to_numpy()
        return X, y
    raise ValueError(f"Unknown data source: {source!r}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg["pipeline"]["random_state"])

    X, y = load_data(cfg)
    logger.info("Loaded data: X=%s  y=%s", X.shape, y.shape)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["pipeline"]["test_size"],
        stratify=y, random_state=cfg["pipeline"]["random_state"],
    )

    cv = StratifiedKFold(
        n_splits=cfg["pipeline"]["cv_splits"],
        shuffle=True,
        random_state=cfg["pipeline"]["random_state"],
    )

    base_models = {
        "xgb": XGBoostTrainer(**cfg["xgb"]),
        "lgb": LightGBMTrainer(**cfg["lgb"]),
        "cat": CatBoostTrainer(**cfg["cat"]),
    }

    oof = np.zeros((X_tr.shape[0], len(base_models)), dtype=np.float64)
    test_preds = np.zeros((X_te.shape[0], len(base_models)), dtype=np.float64)

    for j, (name, model) in enumerate(base_models.items()):
        logger.info("Training %s with %d-fold CV", name, cfg["pipeline"]["cv_splits"])
        for fold, (tr, va) in enumerate(cv.split(X_tr, y_tr)):
            m = base_models[name].__class__(**{
                k: v for k, v in base_models[name].__dict__.items() if not k.endswith("_")
            })
            m.fit(X_tr[tr], y_tr[tr], eval_set=(X_tr[va], y_tr[va]))
            oof[va, j] = m.predict_proba(X_tr[va])[:, 1]
            test_preds[:, j] += m.predict_proba(X_te)[:, 1] / cfg["pipeline"]["cv_splits"]
            logger.info("  fold %d done", fold + 1)

    blender = WeightedEnsemble(metric=cfg["ensemble"]["metric"]).fit(oof, y_tr)
    final_test = blender.predict(test_preds)
    metrics = classification_metrics(y_te, final_test)
    logger.info("Final test metrics: %s", metrics)
    logger.info("Blender weights: %s", blender.report())

    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "pipeline_results.json").open("w") as f:
        json.dump({"metrics": metrics, "weights": blender.report()}, f, indent=2)
    logger.info("Wrote report to %s", reports_dir / "pipeline_results.json")


if __name__ == "__main__":
    main()
