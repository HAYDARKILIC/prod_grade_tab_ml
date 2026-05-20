"""Unit tests for Exclusive Feature Bundling."""

from __future__ import annotations

import numpy as np

from gradient_forge.lightgbm_internals.efb import ExclusiveFeatureBundling


def test_truly_exclusive_features_bundle_together() -> None:
    # Three one-hot columns of a single categorical: exactly one is non-zero per row.
    n = 100
    X = np.zeros((n, 3))
    idx = np.arange(n)
    X[idx, idx % 3] = 1.0
    efb = ExclusiveFeatureBundling(max_conflict_rate=0.0).fit(X)
    assert len(efb.bundles_) == 1
    assert sorted(efb.bundles_[0].feature_ids) == [0, 1, 2]


def test_dense_features_do_not_bundle() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))  # all features always non-zero
    efb = ExclusiveFeatureBundling(max_conflict_rate=0.0).fit(X)
    assert len(efb.bundles_) == 4


def test_transform_preserves_row_count() -> None:
    rng = np.random.default_rng(1)
    X = rng.choice([0.0, 1.0], size=(50, 6), p=[0.9, 0.1])
    efb = ExclusiveFeatureBundling(max_conflict_rate=0.05)
    out = efb.fit_transform(X)
    assert out.shape[0] == 50
    assert out.shape[1] <= 6
