"""Leak-aware preprocessing.

The cardinal sin of tabular ML is fitting any transformer (imputer,
scaler, encoder) on the union of train and test, leaking target-
correlated statistics across the split.  ``LeakAwarePipeline`` is a
thin wrapper around scikit-learn's ColumnTransformer that enforces
fit-on-train-only and exposes a single, serializable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class LeakAwarePipeline:
    """Numeric + categorical pipeline with explicit fit/transform separation."""

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    scale_numeric: bool = False
    numeric_strategy: str = "median"

    transformer_: ColumnTransformer | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: NDArray[np.float64] | None = None) -> "LeakAwarePipeline":
        steps: list[tuple[str, Any, list[str]]] = []
        if self.numeric_features:
            num_pipe_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy=self.numeric_strategy))
            ]
            if self.scale_numeric:
                num_pipe_steps.append(("scaler", StandardScaler()))
            steps.append(("num", Pipeline(num_pipe_steps), self.numeric_features))
        if self.categorical_features:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
            ])
            steps.append(("cat", cat_pipe, self.categorical_features))
        self.transformer_ = ColumnTransformer(steps, remainder="drop")
        self.transformer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.transformer_ is None:
            raise RuntimeError("fit() must be called before transform()")
        return np.asarray(self.transformer_.transform(X), dtype=np.float64)

    def fit_transform(
        self, X: pd.DataFrame, y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        return self.fit(X, y).transform(X)
