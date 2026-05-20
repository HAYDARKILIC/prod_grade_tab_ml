"""Training-speed benchmark across XGBoost, LightGBM, and CatBoost.

Writes per-library timings (median over N runs) to a JSON file consumed
by the CI benchmarks workflow.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from gradient_forge.catboost_internals import CatBoostTrainer
from gradient_forge.data.loaders import load_synthetic_classification
from gradient_forge.lightgbm_internals import LightGBMTrainer
from gradient_forge.utils import Stopwatch, seed_everything
from gradient_forge.xgboost_internals import XGBoostTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("reports/speed.json"))
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--n-features", type=int, default=40)
    p.add_argument("--n-estimators", type=int, default=300)
    return p.parse_args()


def measure(trainer_cls: type, X: np.ndarray, y: np.ndarray, n_runs: int, **kwargs: object) -> dict:
    runs: list[float] = []
    for _ in range(n_runs):
        with Stopwatch() as sw:
            trainer_cls(**kwargs).fit(X, y)
        runs.append(sw.seconds)
    return {"median_s": statistics.median(runs), "runs_s": runs}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    X, y = load_synthetic_classification(n_samples=args.n_samples, n_features=args.n_features)

    results = {
        "xgboost":  measure(XGBoostTrainer,  X, y, args.n_runs,
                            task="binary", n_estimators=args.n_estimators),
        "lightgbm": measure(LightGBMTrainer, X, y, args.n_runs,
                            task="binary", n_estimators=args.n_estimators),
        "catboost": measure(CatBoostTrainer, X, y, args.n_runs,
                            task="binary", iterations=args.n_estimators),
    }
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
