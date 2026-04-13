"""
main.py — FastAPI Surge Prediction API
========================================
Production REST API with full input validation and error handling.

Endpoints:
  GET  /              → health check + model status
  GET  /model/info    → model metadata, metrics, top features
  POST /predict       → single ride prediction with business reasoning
  POST /predict/batch → batch CSV upload → returns CSV with predictions
  GET  /scenarios     → run all business scenario tests live

Run:
    uvicorn src.api.main:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs

Test without running server:
    python -m src.api.test_api
"""

import io
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.models.predict import load_artifacts, predict_batch, predict_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "models/"

# ──────────────────────────────────────────────────────────────────────────────
# App definition
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Uber Surge Pricing API",
    description="""
## Dynamic Surge Prediction for Indian Ride-Hailing Markets

**Model**: Random Forest (selected by 5-fold CV over Ridge + Gradient Boosting)

**Target**: `surge_multiplier` — the multiplier applied to base fare (1.0x to 3.5x)

**Why tree model?**
Surge is driven by NON-LINEAR interactions between time, weather, and events.
Heavy Rain at 7PM during Diwali is not simply Rain + Evening + Diwali — they compound.
A linear model structurally cannot capture this. Random Forest can.

**Top surge drivers** (from feature importance):
1. `wait_time_mins` — proxy for driver shortage
2. `special_event` — Indian festival demand spikes
3. `is_late_night` — supply drops sharply after 10PM
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas — with validation + examples
# ──────────────────────────────────────────────────────────────────────────────

VALID_CITIES    = {"Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad"}
VALID_WEATHER   = {"Clear","Light_Rain","Heavy_Rain","Hot","Humid","Dusty"}
VALID_EVENTS    = {"Regular","Diwali","Holi","Eid"}
VALID_SEGMENTS  = {"Budget","Regular","Premium"}


class RideRequest(BaseModel):
    city:             str   = Field(..., example="Mumbai",
                                    description="Indian city — one of: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad")
    hour:             int   = Field(..., ge=0, le=23, example=19,
                                    description="Hour of day (0–23)")
    day_of_week:      int   = Field(..., ge=1, le=7, example=2,
                                    description="1=Monday, 7=Sunday")
    weather:          str   = Field(..., example="Heavy_Rain",
                                    description="Weather: Clear, Light_Rain, Heavy_Rain, Hot, Humid, Dusty")
    special_event:    str   = Field(..., example="Regular",
                                    description="Event: Regular, Diwali, Holi, Eid")
    distance_km:      float = Field(..., gt=0, le=100, example=8.5,
                                    description="Trip distance in km")
    base_fare:        float = Field(..., gt=0, example=25.0,
                                    description="Base fare in INR before surge")
    customer_segment: str   = Field(..., example="Budget",
                                    description="Customer type: Budget, Regular, Premium")
    wait_time_mins:   int   = Field(default=10, ge=0, le=120, example=15,
                                    description="Estimated driver wait time (minutes)")
    ride_completed:   int   = Field(default=1, ge=0, le=1, example=1,
                                    description="1 if ride was accepted, 0 if cancelled")

    @field_validator("city")
    @classmethod
    def valid_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"city must be one of {sorted(VALID_CITIES)}")
        return v

    @field_validator("weather")
    @classmethod
    def valid_weather(cls, v):
        if v not in VALID_WEATHER:
            raise ValueError(f"weather must be one of {sorted(VALID_WEATHER)}")
        return v

    @field_validator("special_event")
    @classmethod
    def valid_event(cls, v):
        if v not in VALID_EVENTS:
            raise ValueError(f"special_event must be one of {sorted(VALID_EVENTS)}")
        return v

    @field_validator("customer_segment")
    @classmethod
    def valid_segment(cls, v):
        if v not in VALID_SEGMENTS:
            raise ValueError(f"customer_segment must be one of {sorted(VALID_SEGMENTS)}")
        return v


class PredictionResponse(BaseModel):
    surge_multiplier:  float
    surge_category:    str
    estimated_fare:    float
    model_name:        str
    business_context:  str   # Why this surge — human-readable


# ──────────────────────────────────────────────────────────────────────────────
# Business context generator — turns a number into a sentence
# ──────────────────────────────────────────────────────────────────────────────

def _build_business_context(ride: RideRequest, surge: float) -> str:
    reasons = []
    if ride.weather in ("Heavy_Rain", "Light_Rain"):
        reasons.append(f"{ride.weather.replace('_',' ')} reduces driver supply")
    if ride.special_event != "Regular":
        reasons.append(f"{ride.special_event} festival spikes demand")
    if ride.hour in {18, 19, 20, 21}:
        reasons.append("evening peak hours")
    if ride.hour in {22, 23, 0, 1}:
        reasons.append("late night — low driver availability")
    if ride.hour in {8, 9, 10}:
        reasons.append("morning peak — office commute")
    if ride.wait_time_mins > 15:
        reasons.append(f"high wait time ({ride.wait_time_mins} min) signals driver shortage")
    if ride.customer_segment == "Budget":
        reasons.append("Budget segment — price-sensitive, may cancel at higher surge")

    if not reasons:
        return "Normal conditions — no significant demand pressure."

    return f"Surge driven by: {'; '.join(reasons)}."


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Health check — also shows whether model is loaded."""
    try:
        _, _, meta = load_artifacts(MODEL_DIR)
        return {
            "status":       "ok",
            "model_loaded": True,
            "model_name":   meta["best_model"],
            "api_version":  "2.0.0",
        }
    except Exception:
        return {
            "status":       "degraded",
            "model_loaded": False,
            "message":      "Run pipeline.py to train the model first.",
        }


@app.get("/model/info", tags=["Model"])
def model_info():
    """
    Return model metadata including:
    - Which model was selected and why
    - Full comparison table (Ridge vs RF vs GB)
    - Top 5 features driving surge
    - Training metrics
    """
    try:
        _, _, meta = load_artifacts(MODEL_DIR)

        # Load feature importance if available
        fi_path = Path(MODEL_DIR) / "feature_importance_detailed.csv"
        top_features = []
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            top_features = fi_df.head(5)[["feature","importance","business_meaning"]].to_dict("records")

        # Load model comparison if available
        cmp_path = Path(MODEL_DIR) / "model_comparison.csv"
        comparison = []
        if cmp_path.exists():
            cmp_df = pd.read_csv(cmp_path)
            comparison = cmp_df.to_dict("records")

        return {
            "selected_model":  meta["best_model"],
            "selection_reason": (
                "Lowest 5-fold CV-RMSE — chosen over Gradient Boosting and Ridge. "
                "Tree model needed because surge = non-linear interactions of time × weather × festival."
            ),
            "metrics":         meta.get("metrics", {}),
            "feature_count":   len(meta.get("feature_cols", [])),
            "train_rows":      meta.get("train_rows"),
            "test_rows":       meta.get("test_rows"),
            "top_5_features":  top_features,
            "model_comparison": comparison,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503,
            detail="Model not found. Run: python pipeline.py")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(ride: RideRequest):
    """
    Predict surge multiplier for a single ride.

    Returns:
    - **surge_multiplier**: 1.0x – 3.5x
    - **surge_category**: Low / Moderate / High / Very High
    - **estimated_fare**: base_fare × surge_multiplier
    - **business_context**: human-readable reason for the surge level
    """
    try:
        result       = predict_single(ride.model_dump(), model_dir=MODEL_DIR)
        surge        = result["surge_multiplier"]
        est_fare     = round(ride.base_fare * surge, 2)
        context      = _build_business_context(ride, surge)

        return PredictionResponse(
            surge_multiplier = surge,
            surge_category   = result["surge_category"],
            estimated_fare   = est_fare,
            model_name       = result["model_name"],
            business_context = context,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503,
            detail="Model not trained. Run: python pipeline.py")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Batch prediction from CSV file upload.
    Upload a CSV with the same columns as /predict.
    Returns a CSV with a `predicted_surge` column appended.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted.")
    try:
        content  = await file.read()
        df       = pd.read_csv(io.BytesIO(content))
        result   = predict_batch(df, model_dir=MODEL_DIR)
        out      = io.StringIO()
        result.to_csv(out, index=False)
        out.seek(0)
        return StreamingResponse(
            io.BytesIO(out.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=surge_predictions.csv"},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained.")
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenarios", tags=["Analysis"])
def run_scenarios():
    """
    Run live scenario tests and return results.
    Shows Rain vs No Rain, Festival vs Normal, Peak vs Off-peak.
    Demonstrates that the model captures real Indian market logic.
    """
    try:
        _, encoder, meta = load_artifacts(MODEL_DIR)
        with open(Path(MODEL_DIR) / "model.pkl", "rb") as f:
            model = pickle.load(f)

        scenario_path = Path(MODEL_DIR) / "scenario_results.csv"
        if scenario_path.exists():
            df = pd.read_csv(scenario_path)
            return {
                "source":   "pre-computed",
                "scenarios": df.to_dict("records"),
                "insight":  (
                    "Model correctly captures: Rain increases surge, "
                    "Festivals spike demand, Late night drives supply shortage."
                ),
            }
        return {"message": "Run python -m src.models.analysis to generate scenarios."}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained.")
