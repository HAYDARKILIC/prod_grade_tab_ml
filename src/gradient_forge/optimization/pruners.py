"""Pruner factory.

We expose Optuna's MedianPruner and HyperbandPruner under a unified
interface so the YAML configs can specify pruner strategies by name.
"""

from __future__ import annotations

from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner


def build_pruner(name: str = "median", **kwargs: object) -> BasePruner:
    name = name.lower()
    if name == "median":
        return MedianPruner(
            n_startup_trials=int(kwargs.get("n_startup_trials", 5)),
            n_warmup_steps=int(kwargs.get("n_warmup_steps", 10)),
        )
    if name == "hyperband":
        return HyperbandPruner(
            min_resource=int(kwargs.get("min_resource", 1)),
            max_resource=kwargs.get("max_resource", "auto"),
            reduction_factor=int(kwargs.get("reduction_factor", 3)),
        )
    if name == "none":
        return NopPruner()
    raise ValueError(f"Unknown pruner: {name}")
