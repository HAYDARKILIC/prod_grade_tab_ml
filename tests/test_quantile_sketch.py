"""Unit tests for the weighted quantile sketch."""

from __future__ import annotations

import numpy as np

from gradient_forge.xgboost_internals.quantile_sketch import WeightedQuantileSketch


def test_uniform_weights_match_unweighted_quantiles() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    h = np.ones_like(x)
    sk = WeightedQuantileSketch(eps=0.05)
    cands = sk.candidates(x, h)
    # The 11 deciles ± boundary corrections fall within the candidate set.
    deciles = np.quantile(x, np.linspace(0.1, 0.9, 9))
    for q in deciles:
        assert np.min(np.abs(cands - q)) < 0.2


def test_weights_skew_candidates() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(loc=-3, size=500), rng.normal(loc=3, size=500)])
    h = np.concatenate([np.ones(500), 100.0 * np.ones(500)])  # right cluster dominates
    sk = WeightedQuantileSketch(eps=0.05)
    cands = sk.candidates(x, h)
    # Most candidates should lie in the heavily-weighted right cluster.
    assert (cands > 0).sum() > (cands < 0).sum()


def test_invalid_eps_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        WeightedQuantileSketch(eps=0.0)
    with pytest.raises(ValueError):
        WeightedQuantileSketch(eps=1.0)
