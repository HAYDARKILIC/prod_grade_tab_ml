"""Unit tests for WeightedEnsemble."""

from __future__ import annotations

import numpy as np

from gradient_forge.ensemble.weighted_ensemble import WeightedEnsemble


def test_weights_sum_to_one_and_are_nonneg() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float)
    # three "models": one good, two noisy
    P = np.column_stack([
        np.clip(y + rng.normal(scale=0.1, size=200), 0, 1),
        rng.uniform(size=200),
        rng.uniform(size=200),
    ])
    blend = WeightedEnsemble(metric="auc").fit(P, y)
    w = blend.weights_
    assert w is not None
    assert (w >= -1e-6).all()
    assert abs(float(w.sum()) - 1.0) < 1e-4


def test_better_model_gets_higher_weight() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=500).astype(float)
    # "good": probabilities monotonically aligned with y; "bad": fully random.
    good = 0.1 + 0.8 * y + rng.normal(scale=0.05, size=500)
    good = np.clip(good, 0, 1)
    bad = rng.uniform(size=500)
    P = np.column_stack([good, bad])
    blend = WeightedEnsemble(metric="auc").fit(P, y)
    assert blend.weights_ is not None
    assert blend.weights_[0] > blend.weights_[1]


def test_mse_metric() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    P = np.column_stack([y + rng.normal(scale=0.1, size=300),
                         y + rng.normal(scale=2.0, size=300)])
    blend = WeightedEnsemble(metric="mse").fit(P, y)
    assert blend.weights_ is not None
    assert blend.weights_[0] > blend.weights_[1]
