

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(n_rides: int, skip_generate: bool):
    raw_path       = Path("data/raw/uber_rides_india.csv")
    processed_path = Path("data/processed/features.csv")
    model_dir      = Path("models/")

    # ── Step 1: Generate data ─────────────────────────────────────────────
    if skip_generate and raw_path.exists():
        logger.info("Skipping data generation — using %s", raw_path)
    else:
        logger.info("=== STEP 1: Generating synthetic dataset (%d rides) ===", n_rides)
        from src.data.generator import generate_rides
        df_raw = generate_rides(n_rides=n_rides)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_path, index=False)
        logger.info("Raw data saved → %s", raw_path)

    # ── Step 2: Feature engineering ───────────────────────────────────────
    logger.info("=== STEP 2: Feature engineering ===")
    df_raw = pd.read_csv(raw_path)
    from src.features.feature_engineering import build_features
    X, y, encoder = build_features(df_raw, fit_encoder=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features_df = X.copy()
    features_df["surge_multiplier"] = y.values
    features_df.to_csv(processed_path, index=False)
    logger.info("Processed features saved → %s", processed_path)

    # ── Step 3: Train models ──────────────────────────────────────────────
    logger.info("=== STEP 3: Training models ===")
    from src.models.train import run_training
    run_training(str(raw_path), str(model_dir))

    # ── Step 4: Deep analysis (the WHY layer) ─────────────────────────────
    logger.info("=== STEP 4: Deep ML Analysis — comparison / features / scenarios / errors ===")
    from src.models.analysis import run_full_analysis
    run_full_analysis(str(raw_path), str(model_dir))

    # ── Step 5: Batch smoke test ──────────────────────────────────────────
    logger.info("=== STEP 5: Batch prediction smoke test ===")
    from src.models.predict import predict_batch
    sample = df_raw.sample(min(50, len(df_raw)), random_state=0)
    scored = predict_batch(sample, model_dir=str(model_dir))
    logger.info("Sample predictions:\n%s",
        scored[["ride_id", "surge_multiplier", "predicted_surge"]].head(8).to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE — ALL ARTIFACTS")
    print("=" * 62)
    print(f"  Raw data              : {raw_path}")
    print(f"  Processed features    : {processed_path}")
    print(f"  Model                 : {model_dir}/model.pkl")
    print(f"  Encoder               : {model_dir}/encoder.pkl")
    print(f"  Model comparison      : {model_dir}/model_comparison.csv")
    print(f"  Feature importance    : {model_dir}/feature_importance_detailed.csv")
    print(f"  Scenario test results : {model_dir}/scenario_results.csv")
    print(f"  Error analysis        : {model_dir}/error_analysis.csv")
    print("=" * 62)
    print("\n  Start the API:")
    print("  uvicorn src.api.main:app --reload --port 8000\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="End-to-end surge pricing pipeline")
    parser.add_argument("--n-rides",       type=int,  default=1500)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_rides=args.n_rides, skip_generate=args.skip_generate)
