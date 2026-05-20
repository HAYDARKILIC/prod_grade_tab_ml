"""Standalone Optuna study runner.

Usage::

    python scripts/run_optuna_study.py \\
        --algorithm xgboost --n-trials 200 --gpu \\
        --storage sqlite:///optuna.db --study-name forge_xgb_v1
"""

from __future__ import annotations

import argparse
import logging

from gradient_forge.data.loaders import load_synthetic_classification
from gradient_forge.optimization import OptunaStudy, build_pruner
from gradient_forge.utils import seed_everything

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("gradient_forge.optuna")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["xgboost", "lightgbm", "catboost"], required=True)
    parser.add_argument("--task", choices=["binary", "regression"], default="binary")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--pruner", default="median", choices=["median", "hyperband", "none"])
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    X, y = load_synthetic_classification(random_state=args.seed)
    pruner = build_pruner(args.pruner)

    study = OptunaStudy(
        algorithm=args.algorithm,
        task=args.task,
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv_splits=args.cv_splits,
        gpu=args.gpu,
        random_state=args.seed,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
    )
    study_obj = study.optimize(X, y)
    logger.info("Best value : %.6f", study_obj.best_value)
    logger.info("Best params: %s", study_obj.best_params)


if __name__ == "__main__":
    main()
