
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]

CITY_BASE_FARES = {
    "Mumbai":    {"min": 15, "max": 45},
    "Delhi":     {"min": 12, "max": 35},
    "Bangalore": {"min": 10, "max": 30},
    "Hyderabad": {"min":  8, "max": 25},
    "Chennai":   {"min":  8, "max": 25},
    "Kolkata":   {"min":  7, "max": 22},
    "Pune":      {"min":  9, "max": 28},
    "Ahmedabad": {"min":  7, "max": 20},
}

WEATHER_CONDITIONS = ["Clear", "Light_Rain", "Heavy_Rain", "Hot", "Humid", "Dusty"]

SPECIAL_DAYS = ["Diwali", "Holi", "Eid", "Regular", "Regular", "Regular", "Regular"]

CUSTOMER_SEGMENTS = ["Budget", "Regular", "Premium"]
SEGMENT_WEIGHTS   = [0.50, 0.35, 0.15]

MAX_SURGE = 3.5


# ─────────────────────────────────────────────
# Core generation logic
# ─────────────────────────────────────────────

def _weather_surge(weather: str, hour: int) -> float:
    if weather == "Heavy_Rain":
        return 1.5
    if weather == "Light_Rain":
        return 1.2
    if weather == "Hot" and hour in range(12, 16):
        return 1.3
    return 1.0


def _festival_surge(event: str) -> float:
    return 1.8 if event in {"Diwali", "Holi", "Eid"} else 1.0


def _base_surge(hour: int, day_of_week: int) -> float:
    morning_peak   = hour in {8, 9, 10}
    evening_peak   = hour in {18, 19, 20, 21}
    late_night     = hour in {22, 23, 0, 1}
    weekend_evening = day_of_week in {6, 7} and hour in {19, 20, 21, 22}

    if late_night:
        return random.choice([1.5, 2.0, 2.5])
    if evening_peak:
        return random.choice([1.2, 1.5, 2.0])
    if morning_peak:
        return random.choice([1.0, 1.2, 1.5])
    if weekend_evening:
        return random.choice([1.3, 1.8, 2.2])
    return random.choice([1.0, 1.0, 1.0, 1.1])


def _cancel_probability(surge: float, weather: str, event: str, segment: str) -> float:
    """Return cancellation probability given context."""
    if surge >= 3.0:
        prob = 0.60
    elif surge >= 2.5:
        prob = 0.45
    elif surge >= 2.0:
        prob = 0.30
    elif surge >= 1.5:
        prob = 0.15
    elif surge >= 1.3:
        prob = 0.08
    else:
        prob = 0.03

    if weather == "Heavy_Rain":
        prob *= 0.7
    if event in {"Diwali", "Holi", "Eid"}:
        prob *= 0.6
    if segment == "Budget" and surge > 1.5:
        prob = min(prob + 0.10, 0.90)
    if segment == "Premium":
        prob = max(prob - 0.05, 0.01)

    return prob


def _wait_time(surge: float, hour: int, weather: str) -> int:
    base      = 8
    surge_w   = (surge - 1) * 4
    traffic_w = 2 if hour in {8, 9, 18, 19, 20} else 0
    weather_w = 5 if weather == "Heavy_Rain" else 0
    noise     = random.uniform(-3, 5)
    return max(2, round(base + surge_w + traffic_w + weather_w + noise))


def generate_rides(n_rides: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Indian ride-hailing dataset.

    Parameters
    ----------
    n_rides : int
        Number of ride records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Ride-level DataFrame with all raw features.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    for i in range(n_rides):
        city        = random.choice(CITIES)
        hour        = random.randint(0, 23)
        day_of_week = random.randint(1, 7)   # 1=Mon, 7=Sun
        weather     = random.choice(WEATHER_CONDITIONS)
        event       = random.choice(SPECIAL_DAYS)
        segment     = random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]

        fare_range = CITY_BASE_FARES[city]
        distance_km = round(random.uniform(2, 25), 1)
        base_fare   = round(random.uniform(fare_range["min"], fare_range["max"]), 2)

        total_surge = _base_surge(hour, day_of_week) * _weather_surge(weather, hour) * _festival_surge(event)
        surge       = round(min(total_surge, MAX_SURGE), 2)

        cancel_prob  = _cancel_probability(surge, weather, event, segment)
        completed    = random.random() > cancel_prob

        # Add realistic noise to final_fare (avoids trivial base_fare * surge leakage)
        noise      = np.random.normal(0, 2.0)
        final_fare = round(base_fare * surge + noise, 2)

        records.append({
            "ride_id":          f"IND_{i + 1:05d}",
            "city":             city,
            "hour":             hour,
            "day_of_week":      day_of_week,
            "weather":          weather,
            "special_event":    event,
            "distance_km":      distance_km,
            "base_fare":        base_fare,
            "customer_segment": segment,
            "surge_multiplier": surge,        # ← TARGET variable
            "final_fare":       final_fare,
            "ride_completed":   int(completed),
            "wait_time_mins":   _wait_time(surge, hour, weather),
        })

    df = pd.DataFrame(records)
    logger.info("Generated %d rides across %d cities.", len(df), df["city"].nunique())
    return df


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Uber India synthetic dataset")
    parser.add_argument("--n_rides", type=int, default=1500)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str, default="data/raw/uber_rides_india.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_rides(n_rides=args.n_rides, seed=args.seed)
    df.to_csv(out_path, index=False)
    logger.info("Saved dataset → %s", out_path)
