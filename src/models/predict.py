import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Loader (singleton-style caching)
# ─────────────────────────────────────────────

_cache: dict = {}


def load_artifacts(model_dir: str = "models/") -> tuple:
    """
    Load and cache model, encoder, metadata, and SHAP explainer from disk.
    Returns (model, encoder, metadata, explainer).
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
            
        # Initialize SHAP explainer for tree models
        try:
            _cache["explainer"] = shap.TreeExplainer(_cache["model"])
        except Exception as e:
            logger.warning(f"Could not load SHAP explainer: {e}")
            _cache["explainer"] = None
            
        logger.info(
            "Loaded model: %s | Features: %d",
            _cache["metadata"]["best_model"],
            len(_cache["metadata"]["feature_cols"]),
        )

    return _cache["model"], _cache["encoder"], _cache["metadata"], _cache.get("explainer")


# ─────────────────────────────────────────────
# Single-row prediction (used by API)
# ─────────────────────────────────────────────

def predict_single(input_dict: dict, model_dir: str = "models/") -> dict:
    """
    Predict surge multiplier for a single ride request, including SHAP explanations
    and confidence intervals.
    """
    model, encoder, metadata, explainer = load_artifacts(model_dir)

    df = pd.DataFrame([input_dict])
    df["surge_multiplier"] = 0.0
    X, _, _ = build_features(df, encoder=encoder, fit_encoder=False)

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
        
    # --- Confidence Intervals (If Random Forest) ---
    ci_lower, ci_upper = surge, surge
    if hasattr(model, "estimators_"):
        try:
            # Random Forest returns variance across trees
            preds = [est.predict(X.values)[0] for est in model.estimators_]
            ci_lower = round(max(1.0, np.percentile(preds, 10)), 2)
            ci_upper = round(min(3.5, np.percentile(preds, 90)), 2)
        except Exception:
            pass

    # --- SHAP Explanations ---
    shap_explanation = {}
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X)
            # Depending on SHAP version/model, it might be a list or array
            if isinstance(shap_values, list):
                sv = shap_values[0][0]
            else:
                sv = shap_values[0]
                
            # Create feature to SHAP value map
            feature_impacts = {feat: float(val) for feat, val in zip(expected, sv)}
            
            # Get top 3 positive and top 1 negative driver
            sorted_impacts = sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True)
            shap_explanation["pushing_surge_up"] = [f"{k} (+{v:.2f}x)" for k, v in sorted_impacts[:3] if v > 0.02]
            shap_explanation["pushing_surge_down"] = [f"{k} ({v:.2f}x)" for k, v in reversed(sorted_impacts) if v < -0.02][:2]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    return {
        "surge_multiplier": surge,
        "surge_category":   category,
        "confidence_interval": [ci_lower, ci_upper],
        "shap_explanation": shap_explanation,
        "model_name":       metadata["best_model"],
    }


# ─────────────────────────────────────────────
# Batch scoring
# ─────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, model_dir: str = "models/") -> pd.DataFrame:
    """Score an entire DataFrame and return it with a prediction column appended."""
    model, encoder, metadata, _ = load_artifacts(model_dir)

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
