
import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Loader (singleton-style caching)
# ─────────────────────────────────────────────

_cache: dict = {}


def load_artifacts(model_dir: str = "models/") -> tuple:
    """
    Load and cache model, encoder, and metadata from disk.
    Returns (model, encoder, metadata).
    """
    global _cache
    model_dir = Path(model_dir)

    if "model" not in _cache:
        with open(model_dir / "model.pkl", "rb") as f:
            _cache["model"] = pickle.load(f)
        with open(model_dir / "encoder.pkl", "rb") as f:
            _cache["encoder"] = pickle.load(f)
        with open(model_dir / "metadata.json") as f:
            _cache["metadata"] = json.load(f)
        logger.info(
            "Loaded model: %s | Features: %d",
            _cache["metadata"]["best_model"],
            len(_cache["metadata"]["feature_cols"]),
        )

    return _cache["model"], _cache["encoder"], _cache["metadata"]


# ─────────────────────────────────────────────
# Single-row prediction (used by API)
# ─────────────────────────────────────────────

def predict_single(input_dict: dict, model_dir: str = "models/") -> dict:
    """
    Predict surge multiplier for a single ride request.

    Parameters
    ----------
    input_dict : dict
        Keys must match raw feature schema (see feature_engineering.ALLOWED_RAW).
    model_dir : str
        Path to saved artifacts directory.

    Returns
    -------
    dict with:
        surge_multiplier : float   (clamped 1.0 – 3.5)
        surge_category   : str     (Normal / Moderate / High / Very High)
        model_name       : str
    """
    model, encoder, metadata = load_artifacts(model_dir)

    df = pd.DataFrame([input_dict])
    # Add a dummy surge_multiplier so build_features doesn't raise (it's immediately dropped)
    df["surge_multiplier"] = 0.0
    X, _, _ = build_features(df, encoder=encoder, fit_encoder=False)

    # Align columns to training schema
    expected = metadata["feature_cols"]
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]

    raw_pred = float(model.predict(X)[0])
    surge    = round(max(1.0, min(raw_pred, 3.5)), 2)

    if surge < 1.3:
        category = "Normal"
    elif surge < 1.8:
        category = "Moderate"
    elif surge < 2.5:
        category = "High"
    else:
        category = "Very High"

    return {
        "surge_multiplier": surge,
        "surge_category":   category,
        "model_name":       metadata["best_model"],
    }


# ─────────────────────────────────────────────
# Batch scoring
# ─────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, model_dir: str = "models/") -> pd.DataFrame:
    """Score an entire DataFrame and return it with a prediction column appended."""
    model, encoder, metadata = load_artifacts(model_dir)

    if "surge_multiplier" not in df.columns:
        df = df.copy()
        df["surge_multiplier"] = 0.0

    X, _, _ = build_features(df, encoder=encoder, fit_encoder=False)

    expected = metadata["feature_cols"]
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]

    preds = model.predict(X)
    df = df.copy()
    df["predicted_surge"] = [round(max(1.0, min(p, 3.5)), 2) for p in preds]
    return df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Batch surge prediction")
    parser.add_argument("--input",     default="data/raw/uber_rides_india.csv")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--output",    default="data/processed/predictions.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raw = pd.read_csv(args.input)
    out = predict_batch(raw, model_dir=args.model_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)
