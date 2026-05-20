"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from gradient_forge.data.loaders import (
    load_synthetic_classification,
    load_synthetic_regression,
)


@pytest.fixture(scope="session")
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    return load_synthetic_regression(n_samples=500, n_features=10, n_informative=5, random_state=0)


@pytest.fixture(scope="session")
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    return load_synthetic_classification(n_samples=500, n_features=10, n_informative=5, random_state=0)
