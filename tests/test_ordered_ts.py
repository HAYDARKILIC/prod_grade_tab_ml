"""Unit tests for ordered target statistics."""

from __future__ import annotations

import numpy as np

from gradient_forge.catboost_internals.ordered_ts import (
    OrderedTargetStatistics,
    naive_mean_target_encode,
)


def test_naive_encoding_is_perfectly_correlated_with_target() -> None:
    rng = np.random.default_rng(0)
    cats = rng.choice(["a", "b", "c"], size=300)
    y = (cats == "a").astype(float) + rng.normal(scale=0.01, size=300)
    enc = naive_mean_target_encode(cats, y)
    # Leaky encoding => corr ≈ 1.
    corr = float(np.corrcoef(enc, y)[0, 1])
    assert corr > 0.9


def test_ordered_encoding_has_lower_train_correlation() -> None:
    rng = np.random.default_rng(1)
    cats = rng.choice(["a", "b", "c"], size=500)
    y = (cats == "a").astype(float) + rng.normal(scale=0.05, size=500)

    leaky = naive_mean_target_encode(cats, y)
    ordered = OrderedTargetStatistics(smoothing=1.0, random_state=0).fit_transform(cats, y)

    c_leaky = abs(float(np.corrcoef(leaky, y)[0, 1]))
    c_ord = abs(float(np.corrcoef(ordered, y)[0, 1]))
    assert c_ord < c_leaky


def test_transform_uses_global_stats() -> None:
    rng = np.random.default_rng(2)
    cats = rng.choice(["a", "b"], size=200)
    y = rng.normal(size=200)
    enc = OrderedTargetStatistics(smoothing=1.0, random_state=0)
    _ = enc.fit_transform(cats, y)
    # Inference path uses cached stats — encoded values are finite scalars.
    out = enc.transform(np.array(["a", "b", "c"], dtype=object))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
