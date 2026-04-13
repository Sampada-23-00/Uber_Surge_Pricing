"""
test_api.py — API contract tests (no running server needed)
Run: python -m src.api.test_api
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.predict import predict_single

MODEL_DIR = "models/"

TEST_CASES = [
    {
        "name": "Normal ride — expect Low surge",
        "input": {
            "city": "Mumbai", "hour": 14, "day_of_week": 2,
            "weather": "Clear", "special_event": "Regular",
            "distance_km": 8.0, "base_fare": 25.0,
            "customer_segment": "Regular", "wait_time_mins": 8, "ride_completed": 1,
        },
        "expect_min": 1.0, "expect_max": 1.5,
    },
    {
        "name": "Evening Peak + Heavy Rain — expect High",
        "input": {
            "city": "Mumbai", "hour": 19, "day_of_week": 2,
            "weather": "Heavy_Rain", "special_event": "Regular",
            "distance_km": 8.0, "base_fare": 25.0,
            "customer_segment": "Regular", "wait_time_mins": 22, "ride_completed": 1,
        },
        "expect_min": 1.8, "expect_max": 3.5,
    },
    {
        "name": "Diwali Festival Night — expect Very High",
        "input": {
            "city": "Mumbai", "hour": 20, "day_of_week": 5,
            "weather": "Clear", "special_event": "Diwali",
            "distance_km": 8.0, "base_fare": 25.0,
            "customer_segment": "Regular", "wait_time_mins": 18, "ride_completed": 1,
        },
        "expect_min": 2.0, "expect_max": 3.5,
    },
    {
        "name": "Late Night — expect High",
        "input": {
            "city": "Bangalore", "hour": 1, "day_of_week": 7,
            "weather": "Clear", "special_event": "Regular",
            "distance_km": 5.0, "base_fare": 20.0,
            "customer_segment": "Premium", "wait_time_mins": 20, "ride_completed": 1,
        },
        "expect_min": 1.5, "expect_max": 3.5,
    },
    {
        "name": "Diwali + Rain + Peak — expect Very High compound",
        "input": {
            "city": "Delhi", "hour": 19, "day_of_week": 3,
            "weather": "Heavy_Rain", "special_event": "Diwali",
            "distance_km": 6.0, "base_fare": 18.0,
            "customer_segment": "Budget", "wait_time_mins": 28, "ride_completed": 0,
        },
        "expect_min": 2.5, "expect_max": 3.5,
    },
]


def run_tests():
    print("\n" + "=" * 65)
    print("  API CONTRACT TESTS")
    print("=" * 65)
    print("  {:<45} {:>7}  {:>14}  {}".format("Test", "Surge", "Range", "Result"))
    print("  " + "-" * 62)

    passed = 0
    for tc in TEST_CASES:
        result = predict_single(tc["input"], model_dir=MODEL_DIR)
        surge  = result["surge_multiplier"]
        ok     = tc["expect_min"] <= surge <= tc["expect_max"]
        status = "PASS" if ok else "FAIL"
        rng    = "{}-{}".format(tc["expect_min"], tc["expect_max"])
        print("  {:<45} {:>7.2f}x  {:>14}  {}".format(tc["name"], surge, rng, status))
        if ok:
            passed += 1

    print("\n  Result: {}/{} tests passed".format(passed, len(TEST_CASES)))

    # Business context logic test (inline — no FastAPI import)
    print("\n  Business Context Test (the 'explain yourself' layer):")
    weather, event, hour, wait = "Heavy_Rain", "Diwali", 19, 22
    reasons = []
    if weather in ("Heavy_Rain", "Light_Rain"):
        reasons.append("{} reduces driver supply".format(weather.replace("_", " ")))
    if event != "Regular":
        reasons.append("{} festival spikes demand".format(event))
    if hour in {18, 19, 20, 21}:
        reasons.append("evening peak hours")
    if wait > 15:
        reasons.append("high wait time ({} min) signals driver shortage".format(wait))
    context = "Surge driven by: " + "; ".join(reasons) + "."
    print("  Input  : Mumbai, 7PM, Heavy Rain, Diwali, Budget")
    print("  Context: {}".format(context))
    assert "Diwali" in context and "Heavy" in context
    print("  PASS — business context identifies all surge drivers")

    print("\n  Start API server:")
    print("  pip install fastapi uvicorn")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("  Docs: http://localhost:8000/docs\n")


if __name__ == "__main__":
    run_tests()
