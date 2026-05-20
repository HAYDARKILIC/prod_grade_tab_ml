"""XGBoost-specific algorithmic components (Week 2)."""

from gradient_forge.xgboost_internals.quantile_sketch import WeightedQuantileSketch
from gradient_forge.xgboost_internals.sparsity_aware import SparsityAwareSplitFinder
from gradient_forge.xgboost_internals.trainer import XGBoostTrainer

__all__ = ["XGBoostTrainer", "SparsityAwareSplitFinder", "WeightedQuantileSketch"]
