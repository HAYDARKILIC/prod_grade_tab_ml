"""GradientForge — production-grade tabular machine learning from first principles.

Subpackages are loaded lazily so that, for example, importing
``gradient_forge.boosting`` does not require optuna to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Haydar Kılıç"

_LAZY_SUBPACKAGES = (
    "boosting",
    "xgboost_internals",
    "lightgbm_internals",
    "catboost_internals",
    "optimization",
    "ensemble",
    "data",
    "utils",
)

__all__ = ["__version__", *_LAZY_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"gradient_forge.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_SUBPACKAGES))
