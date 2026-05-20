"""Peak-memory benchmark across XGBoost, LightGBM, and CatBoost.

Uses ``tracemalloc`` to measure peak allocations during fit().  GPU
memory is not measured here — see ``benchmarks/benchmark_speed.py``
when running on CUDA hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gradient_forge.catboost_internals import CatBoostTrainer
from gradient_forge.data.loaders import load_synthetic_classification
from gradient_forge.lightgbm_internals import LightGBMTrainer
from gradient_forge.utils import MemoryProfiler, seed_everything
from gradient_forge.xgboost_internals import XGBoostTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("reports/memory.json"))
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--n-features", type=int, default=40)
    p.add_argument("--n-estimators", type=int, default=300)
    return p.parse_args()


def measure(trainer_cls: type, X: np.ndarray, y: np.ndarray, **kwargs: object) -> dict:
    with MemoryProfiler() as mp:
        trainer_cls(**kwargs).fit(X, y)
    return {"peak_mb": mp.peak_mb, "current_mb": mp.current_bytes / (1024**2)}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    X, y = load_synthetic_classification(n_samples=args.n_samples, n_features=args.n_features)
    results = {
        "xgboost":  measure(XGBoostTrainer,  X, y, task="binary", n_estimators=args.n_estimators),
        "lightgbm": measure(LightGBMTrainer, X, y, task="binary", n_estimators=args.n_estimators),
        "catboost": measure(CatBoostTrainer, X, y, task="binary", iterations=args.n_estimators),
    }
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
