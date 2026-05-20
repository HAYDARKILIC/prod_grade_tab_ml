"""From-scratch gradient boosting implementations (Weeks 1-2)."""

from gradient_forge.boosting.gradient_booster import GradientBooster
from gradient_forge.boosting.losses import (
    BinaryCrossEntropy,
    LossFunction,
    SquaredError,
)
from gradient_forge.boosting.tree import RegressionTree
from gradient_forge.boosting.xgboost_from_scratch import XGBoostFromScratch

__all__ = [
    "GradientBooster",
    "RegressionTree",
    "XGBoostFromScratch",
    "LossFunction",
    "SquaredError",
    "BinaryCrossEntropy",
]
