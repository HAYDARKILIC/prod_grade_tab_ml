"""Export a trained model to ONNX for downstream deployment.

Usage::

    python scripts/export_onnx.py --model artifacts/xgb.pkl --output artifacts/xgb.onnx
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-features", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model.open("rb") as f:
        model = pickle.load(f)

    # Dispatch on model class name.
    cls = model.__class__.__name__
    if "XGB" in cls:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_xgboost(
            model, initial_types=[("input", FloatTensorType([None, args.n_features]))]
        )
        with args.output.open("wb") as out:
            out.write(onnx_model.SerializeToString())
        logger.info("Wrote XGBoost ONNX model to %s", args.output)
        return
    if "LGBM" in cls:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([None, args.n_features]))]
        )
        with args.output.open("wb") as out:
            out.write(onnx_model.SerializeToString())
        logger.info("Wrote LightGBM ONNX model to %s", args.output)
        return
    if "CatBoost" in cls:
        model.save_model(str(args.output), format="onnx")
        logger.info("Wrote CatBoost ONNX model to %s", args.output)
        return
    raise ValueError(f"Unsupported model type: {cls}")


if __name__ == "__main__":
    main()
