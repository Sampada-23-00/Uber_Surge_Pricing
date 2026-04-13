"""
tests/test_features.py
Unit tests for feature engineering and data generator.
Run: python tests/test_features.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import generate_rides
from src.features.feature_engineering import (
    add_event_features,
    add_time_features,
    add_weather_features,
    build_features,
)

SAMPLE_DF = generate_rides(n_rides=200, seed=0)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.df = SAMPLE_DF.copy()

    def test_row_count(self):
        self.assertEqual(len(self.df), 200)

    def test_required_columns(self):
        required = {"ride_id", "city", "hour", "surge_multiplier", "final_fare", "ride_completed"}
        self.assertTrue(required.issubset(set(self.df.columns)))

    def test_surge_bounds(self):
        self.assertTrue(self.df["surge_multiplier"].between(1.0, 3.5).all())

    def test_no_missing(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_hour_range(self):
        self.assertTrue(self.df["hour"].between(0, 23).all())


class TestTimeFeatures(unittest.TestCase):
    def setUp(self):
        self.df = SAMPLE_DF.copy()

    def test_peak_flags_are_binary(self):
        df = add_time_features(self.df)
        for col in ["is_morning_peak", "is_evening_peak", "is_late_night", "is_weekend"]:
            self.assertTrue(set(df[col].unique()).issubset({0, 1}), f"{col} is not binary")

    def test_cyclical_encoding_bounded(self):
        df = add_time_features(self.df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            self.assertTrue(df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]")


class TestWeatherFeatures(unittest.TestCase):
    def setUp(self):
        self.df = SAMPLE_DF.copy()

    def test_severity_range(self):
        df = add_time_features(self.df)
        df = add_weather_features(df)
        self.assertTrue(df["weather_severity"].between(1, 5).all())

    def test_rainy_flag(self):
        df = add_weather_features(self.df)
        rain_rows = df[df["weather"].isin(["Light_Rain", "Heavy_Rain"])]
        self.assertTrue((rain_rows["is_rainy"] == 1).all())


class TestBuildFeatures(unittest.TestCase):
    def setUp(self):
        self.df = SAMPLE_DF.copy()

    def test_no_leakage(self):
        X, y, _ = build_features(self.df.copy())
        self.assertNotIn("final_fare", X.columns, "LEAKAGE: final_fare in features!")
        self.assertNotIn("surge_multiplier", X.columns, "LEAKAGE: target in features!")

    def test_target_is_surge(self):
        _, y, _ = build_features(self.df.copy())
        self.assertEqual(y.name, "surge_multiplier")
        self.assertTrue(y.between(1.0, 3.5).all())

    def test_feature_count(self):
        X, _, _ = build_features(self.df.copy())
        self.assertGreater(X.shape[1], 10)

    def test_no_nan_in_features(self):
        X, _, _ = build_features(self.df.copy())
        self.assertEqual(X.isnull().sum().sum(), 0)

    def test_encoder_reuse(self):
        train = self.df.iloc[:150]
        test  = self.df.iloc[150:]
        _, _, encoder = build_features(train.copy(), fit_encoder=True)
        X_test, _, _ = build_features(test.copy(), encoder=encoder, fit_encoder=False)
        self.assertEqual(len(X_test), len(test))


if __name__ == "__main__":
    unittest.main(verbosity=2)
