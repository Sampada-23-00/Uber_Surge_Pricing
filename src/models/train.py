"""
train.py
--------
Train and compare multiple models, select the best, and persist it.

Usage:
    python -m src.models.train \
        --data data/processed/features.csv \
        --model-dir models/
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

MODELS = {
    "Ridge (baseline)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=10, subsample=0.8, random_state=42
    ),
}


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def cross_validate(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────

def run_training(data_path: str, model_dir: str) -> dict:
    """
    Full training run:
      1. Load data
      2. Build features (no leakage)
      3. Train + cross-validate all models
      4. Pick best model by CV RMSE
      5. Final evaluation on hold-out test set
      6. Save model + encoder + metadata

    Returns dict of results for all models.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    # ── 2. Feature engineering ────────────────
    X, y, encoder = build_features(df, fit_encoder=True)
    logger.info("Features built: %s", list(X.columns))

    # ── 3. Train / test split ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── 4. Train all models ───────────────────
    results = {}
    trained = {}

    for name, model in MODELS.items():
        logger.info("▶ Training: %s", name)
        model.fit(X_train, y_train)
        cv_rmse = cross_validate(model, X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["cv_rmse"] = round(cv_rmse, 4)
        results[name] = metrics
        trained[name] = model
        logger.info(
            "  RMSE=%.4f | MAE=%.4f | R²=%.4f | CV-RMSE=%.4f",
            metrics["rmse"], metrics["mae"], metrics["r2"], metrics["cv_rmse"],
        )

    # ── 5. Pick best ──────────────────────────
    best_name = min(results, key=lambda n: results[n]["cv_rmse"])
    best_model = trained[best_name]
    logger.info("✅ Best model: %s (CV-RMSE=%.4f)", best_name, results[best_name]["cv_rmse"])

    # ── 6. Feature importance (if available) ──
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        logger.info("Top 10 features:\n%s", importance.head(10).to_string())
        importance.to_csv(model_dir / "feature_importance.csv")

    # ── 7. Save artifacts ─────────────────────
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(model_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    metadata = {
        "best_model":    best_name,
        "feature_cols":  list(X.columns),
        "metrics":       results,
        "train_rows":    len(X_train),
        "test_rows":     len(X_test),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved to %s", model_dir)
    _print_summary(results, best_name)
    return results


def _print_summary(results: dict, best_name: str) -> None:
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV-RMSE':>10}")
    print("-" * 60)
    for name, m in results.items():
        marker = "  ← BEST" if name == best_name else ""
        print(f"{name:<25} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f} {m['cv_rmse']:>10.4f}{marker}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train surge prediction models")
    parser.add_argument("--data",      default="data/raw/uber_rides_india.csv")
    parser.add_argument("--model-dir", default="models/")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(args.data, args.model_dir)
