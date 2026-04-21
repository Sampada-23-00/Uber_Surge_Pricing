"""
main.py — FastAPI Surge Prediction API
========================================
Production REST API with full input validation, API Key Auth, Rate Limiting, and Error Handling.

Endpoints:
  GET  /              → health check + model status
  GET  /model/info    → model metadata, metrics, top features
  POST /predict       → single ride prediction with SHAP explanations & Confidence Intervals
  POST /predict/batch → batch CSV upload → returns CSV with predictions
  POST /scenarios     → LIVE query of business scenario grid

Run:
    uvicorn src.api.main:app --reload --port 8000
"""

import io
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.models.predict import load_artifacts, predict_batch, predict_single
from src.features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "models/"

# ──────────────────────────────────────────────────────────────────────────────
# Security & Rate Limiting
# ──────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Simple API Key verification. In production, validate against DB."""
    if api_key != "SURGE_SECRET_KEY":
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# ──────────────────────────────────────────────────────────────────────────────
# App definition
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Uber Surge Pricing Enterprise API",
    description="""
## Dynamic Surge Prediction for Indian Ride-Hailing Markets
Secured with API Key Auth & Rate Limiting. Uses advanced Tree boosting (XGBoost/LightGBM/RF).
Includes real-time SHAP explainability.
    """,
    version="3.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    city:             str   = Field(..., example="Mumbai", description="Indian city")
    hour:             int   = Field(..., ge=0, le=23, example=19, description="Hour of day (0–23)")
    day_of_week:      int   = Field(..., ge=1, le=7, example=2, description="1=Monday, 7=Sunday")
    weather:          str   = Field(..., example="Heavy_Rain", description="Weather condition")
    special_event:    str   = Field(..., example="Regular", description="Festival or regular day")
    distance_km:      float = Field(..., gt=0, le=100, example=8.5, description="Trip distance in km")
    base_fare:        float = Field(..., gt=0, example=25.0, description="Base fare in INR before surge")
    driver_supply:    int   = Field(default=50, ge=1, le=100, example=30, description="Active driver availability gauge (1-100)")
    customer_segment: str   = Field(..., example="Budget", description="Customer type")
    wait_time_mins:   int   = Field(default=10, ge=0, le=120, example=15, description="Estimated driver wait time")
    ride_completed:   int   = Field(default=1, ge=0, le=1, example=1, description="1 if ride was accepted, 0 if cancelled")

    @field_validator("city", "weather", "special_event", "customer_segment")
    @classmethod
    def valid_categorical(cls, v, info):
        validators = {
            "city": VALID_CITIES, "weather": VALID_WEATHER, 
            "special_event": VALID_EVENTS, "customer_segment": VALID_SEGMENTS
        }
        domain = validators.get(info.field_name)
        if domain and v not in domain:
            raise ValueError(f"{info.field_name} must be within: {sorted(domain)}")
        return v


class PredictionResponse(BaseModel):
    surge_multiplier:    float
    surge_category:      str
    estimated_fare:      float
    confidence_interval: list[float]
    shap_explanation:    dict
    business_context:    str
    model_name:          str


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Health check — also shows whether model is loaded."""
    try:
        _, _, meta, _ = load_artifacts(MODEL_DIR)
        return {
            "status": "ok", "model_loaded": True, 
            "model_name": meta["best_model"], "api_version": "3.0.0"
        }
    except Exception:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return model metadata, full comparison table, and top 5 features driving surge."""
    try:
        _, _, meta, _ = load_artifacts(MODEL_DIR)
        fi_path = Path(MODEL_DIR) / "feature_importance_detailed.csv"
        top_features = pd.read_csv(fi_path).head(5).to_dict("records") if fi_path.exists() else []
        cmp_path = Path(MODEL_DIR) / "model_comparison.csv"
        comparison = pd.read_csv(cmp_path).to_dict("records") if cmp_path.exists() else []

        return {
            "selected_model": meta["best_model"],
            "metrics": meta.get("metrics", {}),
            "top_5_features": top_features,
            "model_comparison": comparison,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found.")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
def predict(request: Request, ride: RideRequest):
    """
    Predict surge multiplier for a single ride with SHAP analysis and CIs.
    Requires Header: `X-API-Key: SURGE_SECRET_KEY`
    """
    try:
        result    = predict_single(ride.model_dump(), model_dir=MODEL_DIR)
        surge     = result["surge_multiplier"]
        est_fare  = round(ride.base_fare * surge, 2)
        
        # Simple heuristic context combining with SHAP
        context = "Demand/Supply pressures identified via SHAP tree explainer."
        
        return PredictionResponse(
            surge_multiplier    = surge,
            surge_category      = result["surge_category"],
            estimated_fare      = est_fare,
            confidence_interval = result.get("confidence_interval", [surge, surge]),
            shap_explanation    = result.get("shap_explanation", {}),
            business_context    = context,
            model_name          = result["model_name"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained.")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenarios/live", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def run_live_scenarios(request: Request, base_ride: RideRequest):
    """
    Generate a live scenario grid by perturbing the provided RideRequest.
    Demonstrates model elasticity constraints in real-time.
    Requires Header: `X-API-Key: SURGE_SECRET_KEY`
    """
    try:
        model, encoder, meta, _ = load_artifacts(MODEL_DIR)
        scenarios = []
        
        # Perturbation Map
        tests = [
            ("Baseline", {}),
            ("Heavy Rain", {"weather": "Heavy_Rain", "driver_supply": max(10, base_ride.driver_supply - 30)}),
            ("Festival (Diwali)", {"special_event": "Diwali", "driver_supply": max(10, base_ride.driver_supply - 15)}),
            ("Late Night Drop", {"hour": 2, "driver_supply": 10}),
            ("Morning Peak", {"hour": 9})
        ]
        
        req_dict = base_ride.model_dump()
        base_surge = None
        
        for name, changes in tests:
            test_ride = req_dict.copy()
            test_ride.update(changes)
            
            df = pd.DataFrame([test_ride])
            df["surge_multiplier"] = 0.0
            X, _, _ = build_features(df, encoder=encoder, fit_encoder=False)
            
            for col in meta["feature_cols"]:
                if col not in X.columns:
                    X[col] = 0
            X = X[meta["feature_cols"]]
            
            pred = round(max(1.0, min(float(model.predict(X)[0]), 3.5)), 2)
            if base_surge is None:
                base_surge = pred
                
            scenarios.append({
                "scenario": name,
                "surge_prediction": pred,
                "delta": round(pred - base_surge, 2)
            })
            
        return {"live_scenario_grid": scenarios}
    except Exception as e:
        logger.exception("Live scenarios failed")
        raise HTTPException(status_code=500, detail=str(e))
