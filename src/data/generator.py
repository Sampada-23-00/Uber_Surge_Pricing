import argparse
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import os

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

def generate_rides(n_rides: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a semi-real Indian ride-hailing dataset by mapping actual 
    rideshare/taxi datasets and their anomalies to Indian contexts.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    
    try:
        logger.info(f"Downloading real dataset snapshot from NYC TLC (it may take a few seconds)...")
        # Reading TLC dataset to inject real demand/supply and pricing variance
        df_real = pd.read_parquet(TLC_URL)
        df_real = df_real.dropna(subset=['tpep_pickup_datetime', 'trip_distance', 'fare_amount'])
        # Filter anomalous negative fares
        df_real = df_real[df_real['fare_amount'] > 0]
        df_real = df_real.sample(min(n_rides * 5, len(df_real)), random_state=seed)
        logger.info("Successfully fetched real TLC data. Mapping to Indian context...")
    except Exception as e:
        logger.error(f"Failed to fetch TLC data: {e}.")
        raise e

    records = []
    
    # We will simulate driver_supply by looking at the density of pickups in the same hour in TLC!
    df_real['hour'] = df_real['tpep_pickup_datetime'].dt.hour
    hour_counts = df_real['hour'].value_counts().to_dict()
    max_count = max(hour_counts.values()) or 1
    
    df_real = df_real.head(n_rides)
    
    for i, row in enumerate(df_real.itertuples()):
        distance_km = round(row.trip_distance * 1.609, 1)
        if distance_km < 1.0: distance_km = 1.0
        
        hour = row.hour
        day_of_week = row.tpep_pickup_datetime.dayofweek + 1
        
        city = random.choice(CITIES)
        fare_range = CITY_BASE_FARES[city]
        base_fare = round(random.uniform(fare_range["min"], fare_range["max"]), 2)
        
        # Calculate real-world anomaly (proxy for traffic/demand surge)
        # Expected NYC fare = $3 + $2 * dist. Real fare is row.fare_amount.
        expected_fare = 3.0 + 2.0 * row.trip_distance
        real_anomaly = row.fare_amount / max(expected_fare, 1.0)
        
        weather = "Clear"
        event = "Regular"
        
        # Inject Indian weather & events dynamically based on the real time/anomaly!
        if real_anomaly > 1.5:
            weather = random.choice(["Heavy_Rain", "Dusty"])
            event = random.choice(["Diwali", "Holi", "Eid"])
        elif real_anomaly > 1.2:
            weather = random.choice(["Light_Rain", "Hot"])
            event = "Regular"
        else:
            weather = random.choice(["Clear", "Humid"])
            event = "Regular"
            
        segment = random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]
        
        # Real surge = the anomaly (clamped)
        surge = min(max(real_anomaly, 1.0), MAX_SURGE)
        surge = round(surge, 2)
        
        # Driver demand/supply ratio driven by real data density
        supply_ratio = 1.0 - (hour_counts.get(hour, 0) / max_count)
        # Low supply ratio = higher wait time
        wait_time_mins = max(2, int((1.0 - supply_ratio) * 15 + surge * 5))
        
        # Map driver_supply (0 to 100, 100=many drivers)
        # Inverse to demand density + random noise
        driver_supply = int(100 * supply_ratio) + random.randint(-10, 10)
        driver_supply = max(10, min(100, driver_supply))
        
        final_fare = round(base_fare * distance_km * surge, 2)
        
        cancel_prob = 0.05
        if segment == "Budget" and surge > 1.5: cancel_prob = 0.5
        completed = random.random() > cancel_prob
        
        records.append({
            "ride_id":          f"IND_{i + 1:05d}",
            "city":             city,
            "hour":             hour,
            "day_of_week":      day_of_week,
            "weather":          weather,
            "special_event":    event,
            "distance_km":      distance_km,
            "base_fare":        base_fare,
            "driver_supply":    driver_supply,
            "customer_segment": segment,
            "surge_multiplier": surge,        # ← TARGET variable
            "final_fare":       final_fare,
            "ride_completed":   int(completed),
            "wait_time_mins":   wait_time_mins,
        })

    df = pd.DataFrame(records)
    logger.info("Generated %d semi-real rides across %d cities.", len(df), df["city"].nunique())
    return df

# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Uber India real-mapped dataset")
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
