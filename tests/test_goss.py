"""Unit tests for Gradient-based One-Side Sampling."""

from __future__ import annotations

import numpy as np
import pytest

from gradient_forge.lightgbm_internals.goss import GradientBasedOneSideSampling


def test_top_indices_always_included() -> None:
    rng = np.random.default_rng(0)
    g = rng.normal(size=1000)
    h = np.ones_like(g)
    sampler = GradientBasedOneSideSampling(top_rate=0.1, other_rate=0.1, random_state=0)
    idx, g_s, h_s = sampler.sample(g, h)
    top10 = np.argsort(-np.abs(g))[:100]
    assert set(top10.tolist()).issubset(set(idx.tolist()))


def test_small_gradient_rows_are_amplified() -> None:
    g = np.linspace(-5, 5, 200)
    h = np.ones_like(g)
    sampler = GradientBasedOneSideSampling(top_rate=0.2, other_rate=0.1, random_state=0)
    idx, g_s, h_s = sampler.sample(g, h)
    # The "other" rows have their g/h scaled by (1-a)/b = 8.0
    amplify = (1.0 - 0.2) / 0.1
    assert np.any(h_s > amplify - 0.5)


def test_invalid_rates_raise() -> None:
    with pytest.raises(ValueError):
        GradientBasedOneSideSampling(top_rate=0.0)
    with pytest.raises(ValueError):
        GradientBasedOneSideSampling(top_rate=0.6, other_rate=0.5)
