"""
feature_engineering.py
-----------------------
All feature transformations live here.
Design rules:
  1. No leakage — target (surge_multiplier) and derived columns (final_fare)
     are NEVER used as input features.
  2. Every transform is a pure function → easy to test.
  3. fit_transform / transform pattern mirrors sklearn for pipeline compatibility.

Usage:
    from src.features.feature_engineering import build_features
    X, y = build_features(df)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Raw feature columns that the model is allowed to see
# (NEVER include surge_multiplier or final_fare here)
# ─────────────────────────────────────────────
ALLOWED_RAW = [
    "hour",
    "day_of_week",
    "distance_km",
    "base_fare",
    "driver_supply",
    "city",
    "weather",
    "special_event",
    "customer_segment",
    "wait_time_mins",
    "ride_completed",
]

TARGET = "surge_multiplier"

# ─────────────────────────────────────────────
# Time features
# ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interpretable time-based features from hour / day_of_week."""
    df = df.copy()
    df["is_morning_peak"]  = df["hour"].isin([8, 9, 10]).astype(int)
    df["is_evening_peak"]  = df["hour"].isin([18, 19, 20, 21]).astype(int)
    df["is_late_night"]    = df["hour"].isin([22, 23, 0, 1]).astype(int)
    df["is_weekend"]       = df["day_of_week"].isin([6, 7]).astype(int)
    df["is_weekend_evening"] = (
        df["is_weekend"] & df["hour"].isin([19, 20, 21, 22])
    ).astype(int)
    # Cyclical encoding so hour 23 and hour 0 are "close"
    df["hour_sin"]         = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]         = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]          = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]          = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ─────────────────────────────────────────────
# Weather features
# ─────────────────────────────────────────────

WEATHER_SEVERITY = {
    "Clear":      1,
    "Hot":        2,
    "Humid":      2,
    "Dusty":      3,
    "Light_Rain": 4,
    "Heavy_Rain": 5,
}

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weather_severity"] = df["weather"].map(WEATHER_SEVERITY).fillna(1).astype(int)
    df["is_rainy"]         = df["weather"].isin(["Light_Rain", "Heavy_Rain"]).astype(int)
    df["is_heavy_rain"]    = (df["weather"] == "Heavy_Rain").astype(int)
    # Interaction: rain during peak = very high demand
    df["rain_x_peak"]      = df["is_rainy"] * (df.get("is_evening_peak", 0) + df.get("is_morning_peak", 0))
    return df


# ─────────────────────────────────────────────
# Festival / event features
# ─────────────────────────────────────────────

FESTIVAL_SCORE = {"Diwali": 3, "Holi": 2, "Eid": 2, "Regular": 0}

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_festival"]    = (df["special_event"] != "Regular").astype(int)
    df["festival_score"] = df["special_event"].map(FESTIVAL_SCORE).fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────
# Customer & location features
# ─────────────────────────────────────────────

SEGMENT_PRICE_SENSITIVITY = {"Budget": 3, "Regular": 2, "Premium": 1}

def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_sensitivity"] = df["customer_segment"].map(SEGMENT_PRICE_SENSITIVITY).fillna(2).astype(int)
    return df


# ─────────────────────────────────────────────
# Encode categoricals
# ─────────────────────────────────────────────

CATEGORICAL_COLS = ["city", "weather", "special_event", "customer_segment"]

class CategoricalEncoder:
    """Fit label encoders on training data; transform train + test consistently."""

    def __init__(self):
        self._encoders: dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self._encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[f"{col}_enc"] = le.transform(df[col])
        return df


# ─────────────────────────────────────────────
# Final feature list (no target, no leakage)
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Raw numeric
    "hour", "day_of_week", "distance_km", "base_fare", "driver_supply", "wait_time_mins", "ride_completed",
    # Time
    "is_morning_peak", "is_evening_peak", "is_late_night", "is_weekend",
    "is_weekend_evening", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Weather
    "weather_severity", "is_rainy", "is_heavy_rain", "rain_x_peak",
    # Event
    "is_festival", "festival_score",
    # Customer
    "price_sensitivity",
    # Encoded categoricals
    "city_enc", "weather_enc", "special_event_enc", "customer_segment_enc",
]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    encoder: CategoricalEncoder | None = None,
    fit_encoder: bool = True,
) -> tuple[pd.DataFrame, pd.Series, CategoricalEncoder]:
    """
    Transform raw DataFrame into model-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ride data (must contain TARGET column for training).
    encoder : CategoricalEncoder | None
        Pass an existing encoder during inference / test set transforms.
    fit_encoder : bool
        Fit a new encoder if True (training mode).

    Returns
    -------
    X : pd.DataFrame        — feature matrix (no leakage)
    y : pd.Series           — surge_multiplier target
    encoder : CategoricalEncoder
    """
    # Safety: ensure target is present
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame.")

    # Verify no leakage columns are accidentally included
    leakage_cols = {"final_fare"}
    if leakage_cols.intersection(df.columns):
        logger.warning(
            "Leakage columns detected and will be excluded from features: %s",
            leakage_cols.intersection(df.columns),
        )

    df = df.copy()

    # Apply transformations
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_event_features(df)
    df = add_customer_features(df)

    # Encode categoricals
    if encoder is None:
        encoder = CategoricalEncoder()
    if fit_encoder:
        df = encoder.fit_transform(df)
    else:
        df = encoder.transform(df)

    # Extract target
    y = df[TARGET].copy()

    # Select only approved feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing feature columns (will be skipped): %s", missing)

    X = df[available].copy()

    logger.info(
        "Feature matrix: %d rows × %d features | Target: %s",
        len(X), X.shape[1], TARGET,
    )
    return X, y, encoder
