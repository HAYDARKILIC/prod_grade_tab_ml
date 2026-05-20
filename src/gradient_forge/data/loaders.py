"""Synthetic and real-data loaders for the notebooks and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression


def load_synthetic_regression(
    n_samples: int = 5_000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    return X.astype(np.float64), y.astype(np.float64)


def load_synthetic_classification(
    n_samples: int = 5_000,
    n_features: int = 20,
    n_informative: int = 10,
    class_sep: float = 1.0,
    weights: tuple[float, float] = (0.7, 0.3),
    random_state: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        class_sep=class_sep,
        weights=list(weights),
        random_state=random_state,
    )
    return X.astype(np.float64), y.astype(np.int64)


def inject_missingness(
    X: NDArray[np.float64], rate: float = 0.1, random_state: int = 42
) -> NDArray[np.float64]:
    """Return a copy of X with `rate` fraction of cells replaced by NaN."""
    rng = np.random.default_rng(random_state)
    X = X.copy()
    mask = rng.random(X.shape) < rate
    X[mask] = np.nan
    return X


def make_categorical_dataset(
    n_samples: int = 5_000,
    n_categories: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """High-cardinality categorical synthetic dataset (e.g. user_id / sku)."""
    rng = np.random.default_rng(random_state)
    cats = [f"cat_{i:03d}" for i in range(n_categories)]
    df = pd.DataFrame({
        "cat_a": rng.choice(cats, size=n_samples),
        "cat_b": rng.choice(cats, size=n_samples),
        "num_a": rng.normal(size=n_samples),
        "num_b": rng.normal(size=n_samples),
    })
    # response depends on category through a hidden lookup
    lookup = {c: rng.normal() for c in cats}
    logit = (
        df["cat_a"].map(lookup).to_numpy()
        + 0.5 * df["cat_b"].map(lookup).to_numpy()
        + df["num_a"].to_numpy()
        + rng.normal(scale=0.1, size=n_samples)
    )
    y = (logit > 0).astype(np.int64)
    return df, y
