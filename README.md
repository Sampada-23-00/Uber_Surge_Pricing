# 🚗 Uber Surge Pricing — Production ML System

A **production-grade dynamic pricing prediction system** for Indian ride-hailing markets.  
Predicts `surge_multiplier` (1.0x – 3.5x) using multi-factor demand signals, served via a REST API.

---

## Architecture

```
uber-pricing/
│
├── src/
│   ├── data/
│   │   └── generator.py          # Synthetic Indian market data generator
│   ├── features/
│   │   └── feature_engineering.py # Zero-leakage feature pipeline
│   ├── models/
│   │   ├── train.py              # Model comparison + selection + saving
│   │   └── predict.py            # Inference (single + batch)
│   └── api/
│       └── main.py               # FastAPI REST endpoint
│
├── data/
│   ├── raw/                      # Generated CSV
│   └── processed/                # Feature-engineered outputs
│
├── models/                       # Saved model artifacts
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── metadata.json
│   └── feature_importance.csv
│
├── tests/
│   └── test_features.py          # 14 unit tests (zero-leakage verified)
│
├── notebooks/                    # EDA notebook (reference only)
├── pipeline.py                   # One-command end-to-end runner
├── dvc.yaml                      # DVC pipeline for versioning
├── Dockerfile                    # Production container
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (data → features → train → score)
```bash
python pipeline.py
```

Output:
```
MODEL COMPARISON SUMMARY
Model                  RMSE      MAE       R²    CV-RMSE
Ridge (baseline)     0.2852   0.2188   0.8643     0.2565
Random Forest        0.2660   0.1871   0.8820     0.2407  ← BEST
Gradient Boosting    0.2655   0.1873   0.8824     0.2428
```

### 3. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "city": "Mumbai",
        "hour": 18,
        "day_of_week": 2,
        "weather": "Heavy_Rain",
        "special_event": "Diwali",
        "distance_km": 8.5,
        "base_fare": 25.0,
        "customer_segment": "Budget",
        "wait_time_mins": 15,
        "ride_completed": 1
      }'
```

Response:
```json
{
  "surge_multiplier": 2.85,
  "surge_category": "High",
  "model_name": "Random Forest",
  "estimated_fare": 71.25
}
```

---

## API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/`             | Health check                       |
| GET    | `/model/info`   | Loaded model metadata + metrics    |
| POST   | `/predict`      | Single ride surge prediction       |
| POST   | `/predict/batch`| Batch CSV upload and scoring       |

Interactive docs → http://localhost:8000/docs

---

## ML Design Decisions

### Target variable: `surge_multiplier`
- NOT `final_fare` (which is a deterministic product of base_fare × surge — trivial to "predict")
- The model learns demand signals, not arithmetic

### Zero-leakage guarantee
- `final_fare` and `surge_multiplier` are **never** input features
- Verified by unit test `test_no_leakage`
- Encoder is fitted **only** on training data and reused for test/inference

### Feature engineering (26 features)
- **Time**: hour (cyclical sin/cos), morning peak, evening peak, late night, weekend flags
- **Weather**: severity score (1–5), rain flag, heavy rain flag, rain × peak interaction
- **Events**: festival flag, festival intensity score
- **Customer**: price sensitivity score (Budget=3, Regular=2, Premium=1)
- **Encoded**: city, weather, event, segment (LabelEncoder, leakage-safe)

### Model selection
Three models trained and compared via 5-fold cross-validation:
- Ridge Regression (baseline)
- Random Forest ✅ (best CV-RMSE)
- Gradient Boosting

---

## Docker Deployment

```bash
# Build
docker build -t uber-surge-api .

# Run
docker run -p 8000:8000 uber-surge-api

# Health check
curl http://localhost:8000/
```

---

## DVC (Data Version Control)

```bash
pip install dvc

# Reproduce full pipeline
dvc repro

# Track data file
dvc add data/raw/uber_rides_india.csv

# View metrics
dvc metrics show
```

---

## Tests

```bash
python tests/test_features.py
# Ran 14 tests in 0.085s — OK
```

Test coverage includes:
- Data generator (row count, column schema, bounds, nulls)
- Time feature flags (binary, cyclical bounds)
- Weather feature correctness
- Zero-leakage verification
- Encoder train/test consistency

---


