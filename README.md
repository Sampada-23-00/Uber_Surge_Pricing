# Uber_Surge_Pricing
🚕 Uber Surge Pricing — Streamlit App

Uber Surge Pricing is a data science project that analyzes ride data and builds models to predict Uber ride prices and surge multipliers. The repository contains data preprocessing and exploration notebooks, visualization and modeling work, and a Streamlit app to interactively estimate fares.

🔍 Repository structure
Uber_Surge_Pricing/
├── notebooks/
│   ├── 01_data_generator.ipynb        
│   ├── 02_data_exploration.ipynb      
│   ├── 03_visualization.ipynb         
│   ├── 04_advanced_models.ipynb       
│   └── Real Time Server/              
├── app.py                             
├── uber_rides_india.csv               
├── uber_rides_ml_enhanced.csv         
├── LICENSE                            
└── README.md

🧾 Project summary

This project aims to:

Explore an Uber ride dataset for India and analyze how price varies with factors such as distance, time, passenger count, and surge events.

Engineer features (Haversine distance, time-of-day, weekday/weekend flags, surge multipliers, etc.).

Train and evaluate regression models to predict ride price and (optionally) surge probability / multiplier.

Provide an interactive Streamlit app (app.py) so users can input trip details and get an estimated fare.

🚀 Quick start — run the Streamlit app

Clone the repo (if you haven’t already)

git clone https://github.com/Sampada-23-00/Uber_Surge_Pricing.git
cd Uber_Surge_Pricing


Create a virtual environment and install dependencies

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt


If you don't have a requirements.txt, create one with the likely packages:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost            # if used
geopy              # optional, for distances


Run the app

streamlit run app.py

📂 Notebooks — what’s inside

01_data_generator.ipynb
Scripts used to generate or sample the dataset, and to create synthetic features if required.

02_data_exploration.ipynb
Data cleaning and exploratory data analysis: missing value handling, outlier removal, basic statistics, and feature engineering (distance calculation, parsing datetime, etc.).

03_visualization.ipynb
Visual analyses that show price distributions, surge frequency by time/location, heatmaps of pickup/drop-offs, and other key charts.

04_advanced_models.ipynb
Model training and evaluation. Typical algorithms tried:

Linear Regression / Regularized variants

Random Forest Regressor

Gradient Boosting / XGBoost / LightGBM
Includes metrics: RMSE, MAE, R² and comparison visualizations.

🧠 Data & features

Files:

uber_rides_india.csv — raw / sample dataset (columns may include pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, passenger_count, fare_amount, surge_multiplier, etc.)

uber_rides_ml_enhanced.csv — cleaned and feature-engineered data ready for training.

Common engineered features in this project:

distance_km — Haversine distance between pickup and dropoff

hour_of_day, day_of_week, is_weekend

surge_flag or surge_multiplier

trip_duration_est (if available or estimated)

(Adjust exact column names in README to match your CSVs if they differ.)

📈 Model performance (example)

Replace these placeholders with your actual results from 04_advanced_models.ipynb:

Model	RMSE	MAE	R²
Random Forest	3.25	1.95	0.89
Gradient Boosting	2.98	1.72	0.91
Linear Regression	4.12	2.40	0.78
🧪 Reproducibility & tips

Use the uber_rides_ml_enhanced.csv as the training input for the notebooks in the order: EDA → visualization → modeling.

If models are large / trained offline, include a model/ folder with serialized model artifacts (.pkl / .joblib) so the app.py can load them quickly.

Add a requirements.txt and, if possible, a Procfile or Dockerfile for deployment.

♻️ How to contribute

Fork the repository

Create a feature branch: git checkout -b feat/your-feature

Commit your changes: git commit -m "Add …"

Push and create a PR

🧾 License

This project is licensed under the MPL-2.0 license.

✉️ Contact

Sampada Waghode — sampada.waghode@gmail.com
GitHub: https://github.com/Sampada-23-00
