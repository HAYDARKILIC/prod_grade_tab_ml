"""Second-order regression tree used as a weak learner.

The tree is grown greedily by maximizing the gain

        Gain = 1/2 · [ G_L² / (H_L + λ) + G_R² / (H_R + λ)
                       - (G_L + G_R)² / (H_L + H_R + λ) ] - γ

where G_J = Σ_{i ∈ I_J} g_i and H_J = Σ_{i ∈ I_J} h_i.  Leaf weights are
the closed-form optimum w* = -G / (H + λ).  This is the same objective
used by XGBoost, restricted to a single tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class TreeNode:
    """A node in a regression tree (internal or leaf)."""

    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    value: float = 0.0
    is_leaf: bool = True
    n_samples: int = 0
    gain: float = 0.0


@dataclass
class RegressionTree:
    """Second-order regression tree (XGBoost-style)."""

    max_depth: int = 6
    min_samples_split: int = 2
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_gamma: float = 0.0
    root: TreeNode | None = field(default=None, init=False)

    # ---------- public API -------------------------------------------------
    def fit(self, X: Array, g: Array, h: Array) -> "RegressionTree":
        """Grow the tree given gradients g and Hessians h."""
        self.root = self._grow(X, g, h, depth=0)
        return self

    def predict(self, X: Array) -> Array:
        if self.root is None:
            raise RuntimeError("Tree is not fitted; call fit() first.")
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self._traverse(X[i], self.root)
        return out

    # ---------- internals --------------------------------------------------
    def _leaf_weight(self, g: Array, h: Array) -> float:
        """Closed-form optimum  w* = -ΣG / (ΣH + λ)."""
        return float(-g.sum() / (h.sum() + self.reg_lambda))

    def _grow(self, X: Array, g: Array, h: Array, depth: int) -> TreeNode:
        n = X.shape[0]
        node = TreeNode(value=self._leaf_weight(g, h), n_samples=n, is_leaf=True)

        if depth >= self.max_depth or n < self.min_samples_split:
            return node

        best = self._best_split(X, g, h)
        if best is None:
            return node

        feature, threshold, gain, left_mask = best
        if gain <= 0.0:
            return node

        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        node.left = self._grow(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)
        return node

    def _best_split(
        self, X: Array, g: Array, h: Array
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        """Exhaustive split finder over (feature, threshold) pairs."""
        n, d = X.shape
        G_total, H_total = float(g.sum()), float(h.sum())
        best_gain = 0.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_mask: NDArray[np.bool_] | None = None

        for feat in range(d):
            xs = X[:, feat]
            order = np.argsort(xs)
            xs_sorted = xs[order]
            g_sorted = g[order]
            h_sorted = h[order]

            G_L = 0.0
            H_L = 0.0
            for i in range(n - 1):
                G_L += g_sorted[i]
                H_L += h_sorted[i]
                # avoid splitting on identical adjacent values
                if xs_sorted[i] == xs_sorted[i + 1]:
                    continue
                G_R = G_total - G_L
                H_R = H_total - H_L
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue
                gain = 0.5 * (
                    G_L * G_L / (H_L + self.reg_lambda)
                    + G_R * G_R / (H_R + self.reg_lambda)
                    - G_total * G_total / (H_total + self.reg_lambda)
                ) - self.reg_gamma
                if gain > best_gain:
                    threshold = 0.5 * (xs_sorted[i] + xs_sorted[i + 1])
                    best_gain = gain
                    best_feat = feat
                    best_thresh = threshold
                    best_mask = xs <= threshold

        if best_feat is None or best_thresh is None or best_mask is None:
            return None
        return best_feat, best_thresh, best_gain, best_mask

    def _traverse(self, x: Array, node: TreeNode) -> float:
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            if x[node.feature] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.value
