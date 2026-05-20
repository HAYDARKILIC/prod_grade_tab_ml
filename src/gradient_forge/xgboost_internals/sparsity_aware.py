"""Sparsity-aware split finding (Chen & Guestrin, 2016, §3.4).

For each candidate feature, missing values are routed to *whichever* child
maximizes the gain.  The chosen direction becomes the learned default
direction for that node — the secret behind XGBoost's robustness to
high-missingness tabular data.

Mathematically:

    For each feature j with a partition of the *observed* rows (G_L, H_L),
    (G_R, H_R) and aggregate (G_miss, H_miss) of the missing rows,
    consider two scenarios:

        (a) missing → left  :   G_L' = G_L + G_miss,   H_L' = H_L + H_miss
        (b) missing → right :   G_R' = G_R + G_miss,   H_R' = H_R + H_miss

    Pick the direction with the larger split gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class SplitResult:
    feature: int
    threshold: float
    gain: float
    default_left: bool


class SparsityAwareSplitFinder:
    """Exact split finder with learned default direction for missing values."""

    def __init__(self, reg_lambda: float = 1.0, reg_gamma: float = 0.0,
                 min_child_weight: float = 1.0) -> None:
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight

    @staticmethod
    def _gain(G_L: float, H_L: float, G_R: float, H_R: float,
              reg_lambda: float, reg_gamma: float) -> float:
        G = G_L + G_R
        H = H_L + H_R
        return 0.5 * (
            G_L * G_L / (H_L + reg_lambda)
            + G_R * G_R / (H_R + reg_lambda)
            - G * G / (H + reg_lambda)
        ) - reg_gamma

    def find_best(self, X: Array, g: Array, h: Array) -> SplitResult | None:
        """Find the best (feature, threshold, default-direction) over X.

        ``np.nan`` entries in X are treated as missing and routed via the
        learned default direction.
        """
        n, d = X.shape
        best: SplitResult | None = None

        for j in range(d):
            xs = X[:, j]
            mask_miss = np.isnan(xs)
            xs_obs = xs[~mask_miss]
            g_obs = g[~mask_miss]
            h_obs = h[~mask_miss]
            g_miss = float(g[mask_miss].sum())
            h_miss = float(h[mask_miss].sum())

            if xs_obs.size < 2:
                continue

            order = np.argsort(xs_obs)
            xs_sorted = xs_obs[order]
            g_sorted = g_obs[order]
            h_sorted = h_obs[order]

            G_obs_total = float(g_sorted.sum())
            H_obs_total = float(h_sorted.sum())

            G_L_obs = 0.0
            H_L_obs = 0.0
            for i in range(xs_sorted.size - 1):
                G_L_obs += float(g_sorted[i])
                H_L_obs += float(h_sorted[i])
                if xs_sorted[i] == xs_sorted[i + 1]:
                    continue
                G_R_obs = G_obs_total - G_L_obs
                H_R_obs = H_obs_total - H_L_obs

                # ---- direction A: missing → left
                G_L_a, H_L_a = G_L_obs + g_miss, H_L_obs + h_miss
                G_R_a, H_R_a = G_R_obs, H_R_obs
                # ---- direction B: missing → right
                G_L_b, H_L_b = G_L_obs, H_L_obs
                G_R_b, H_R_b = G_R_obs + g_miss, H_R_obs + h_miss

                gain_a = -np.inf
                gain_b = -np.inf
                if H_L_a >= self.min_child_weight and H_R_a >= self.min_child_weight:
                    gain_a = self._gain(G_L_a, H_L_a, G_R_a, H_R_a,
                                        self.reg_lambda, self.reg_gamma)
                if H_L_b >= self.min_child_weight and H_R_b >= self.min_child_weight:
                    gain_b = self._gain(G_L_b, H_L_b, G_R_b, H_R_b,
                                        self.reg_lambda, self.reg_gamma)

                if gain_a >= gain_b:
                    gain = gain_a
                    default_left = True
                else:
                    gain = gain_b
                    default_left = False

                if gain > 0 and (best is None or gain > best.gain):
                    threshold = 0.5 * (xs_sorted[i] + xs_sorted[i + 1])
                    best = SplitResult(
                        feature=j,
                        threshold=float(threshold),
                        gain=float(gain),
                        default_left=default_left,
                    )
        return best
