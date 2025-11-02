import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import asyncio
import threading
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import websocket
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Uber Surge Pricing Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .live-data-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
    }
    .model-status {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .websocket-status {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for real integrations
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_scaler' not in st.session_state:
    st.session_state.model_scaler = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'realtime_data_stream' not in st.session_state:
    st.session_state.realtime_data_stream = []

@st.cache_data
def load_data():
    """Generate comprehensive surge pricing dataset for training"""
    np.random.seed(42)
    
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    areas = ['Airport', 'Business District', 'Mall', 'Railway Station', 'Residential', 'Tourist Area']
    weather_conditions = ['Sunny', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Thunderstorm']
    
    data = []
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for day in range(30):
        for hour in range(24):
            for city in cities:
                for area in areas[:3]:
                    timestamp = base_date + datetime.timedelta(days=day, hours=hour)
                    
                    is_rush_hour = hour in [7, 8, 9, 18, 19, 20]
                    is_weekend = timestamp.weekday() >= 5
                    base_demand = 30 + (20 if is_rush_hour else 0) + (15 if is_weekend else 0)
                    
                    is_night = hour < 6 or hour > 22
                    base_supply = 45 - (15 if is_night else 0)
                    
                    weather = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.15, 0.1, 0.05])
                    weather_demand_boost = {'Sunny': 0, 'Cloudy': 3, 'Light Rain': 15, 'Heavy Rain': 35, 'Thunderstorm': 60}
                    weather_supply_reduction = {'Sunny': 0, 'Cloudy': 0, 'Light Rain': 8, 'Heavy Rain': 20, 'Thunderstorm': 35}
                    
                    demand = max(5, base_demand + weather_demand_boost[weather] + np.random.normal(0, 8))
                    supply = max(3, base_supply - weather_supply_reduction[weather] + np.random.normal(0, 6))
                    
                    demand_supply_ratio = demand / supply
                    if demand_supply_ratio <= 1.0:
                        surge_multiplier = 1.0
                    elif demand_supply_ratio <= 1.3:
                        surge_multiplier = 1.0 + (demand_supply_ratio - 1.0) * 0.5
                    elif demand_supply_ratio <= 2.0:
                        surge_multiplier = 1.15 + (demand_supply_ratio - 1.3) * 1.2
                    else:
                        surge_multiplier = min(4.0, 1.99 + (demand_supply_ratio - 2.0) * 0.8)
                    
                    base_fare = np.random.uniform(45, 95)
                    completed_rides = min(demand, supply * 0.8) + np.random.normal(0, 2)
                    completed_rides = max(0, completed_rides)
                    revenue = base_fare * surge_multiplier * completed_rides
                    
                    # Additional features for ML
                    temperature = np.random.uniform(15, 45)  # Celsius
                    humidity = np.random.uniform(30, 90)     # Percentage
                    
                    data.append({
                        'timestamp': timestamp,
                        'city': city,
                        'area': area,
                        'hour': hour,
                        'day_of_week': timestamp.weekday(),
                        'is_weekend': int(is_weekend),
                        'is_rush_hour': int(is_rush_hour),
                        'weather': weather,
                        'demand': round(demand, 1),
                        'supply': round(supply, 1),
                        'temperature': round(temperature, 1),
                        'humidity': round(humidity, 1),
                        'surge_multiplier': round(surge_multiplier, 2),
                        'base_fare': round(base_fare, 2),
                        'completed_rides': round(completed_rides, 1),
                        'revenue': round(revenue, 2),
                        'driver_utilization': min(100, max(0, supply * 1.5 + np.random.uniform(-8, 8)))
                    })
    
    return pd.DataFrame(data)

def prepare_features_for_ml(df):
    """Prepare features for machine learning"""
    # Create feature DataFrame
    features_df = df.copy()
    
    # Encode categorical variables
    city_dummies = pd.get_dummies(features_df['city'], prefix='city')
    area_dummies = pd.get_dummies(features_df['area'], prefix='area')
    weather_dummies = pd.get_dummies(features_df['weather'], prefix='weather')
    
    # Combine all features
    ml_features = pd.concat([
        features_df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                     'demand', 'supply', 'temperature', 'humidity']],
        city_dummies,
        area_dummies,
        weather_dummies
    ], axis=1)
    
    return ml_features

def train_surge_prediction_model(df):
    """Train actual Random Forest model for surge prediction"""
    
    # Prepare features
    X = prepare_features_for_ml(df)
    y = df['surge_multiplier']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Store the feature column names in session state for later use
    st.session_state.feature_columns = X_train.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'feature_importance': dict(zip(X.columns, model.feature_importances_)),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, scaler, metrics

def predict_surge_with_model(demand, supply, hour, day_of_week, is_weekend,
                             city, area, weather, temperature, humidity):
    """Make prediction using trained model"""
    
    if not st.session_state.model_trained or st.session_state.trained_model is None:
        return None
    
    # Retrieve the feature columns from the session state
    feature_columns = st.session_state.get('feature_columns')
    if feature_columns is None:
        st.error("Model features not found. Please train the model first.")
        return None
    
    # Create a dictionary for the new data point, initializing all features to 0
    input_data_dict = {col: 0 for col in feature_columns}

    # Populate the dictionary with the input values
    input_data_dict['hour'] = hour
    input_data_dict['day_of_week'] = day_of_week
    input_data_dict['is_weekend'] = int(is_weekend)
    input_data_dict['is_rush_hour'] = int(hour in [7, 8, 9, 18, 19, 20])
    input_data_dict['demand'] = demand
    input_data_dict['supply'] = supply
    input_data_dict['temperature'] = temperature
    input_data_dict['humidity'] = humidity
    
    # Set the one-hot encoded columns to 1 based on user selection
    if f'city_{city}' in input_data_dict:
        input_data_dict[f'city_{city}'] = 1
    if f'area_{area}' in input_data_dict:
        input_data_dict[f'area_{area}'] = 1
    if f'weather_{weather}' in input_data_dict:
        input_data_dict[f'weather_{weather}'] = 1

    # Create the final DataFrame from the dictionary
    input_data = pd.DataFrame([input_data_dict])

    # Scale and predict
    try:
        input_scaled = st.session_state.model_scaler.transform(input_data)
        prediction = st.session_state.trained_model.predict(input_scaled)[0]
        return max(1.0, min(4.0, prediction))  # Clamp between 1.0 and 4.0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# WebSocket Integration (Simulated)
class WebSocketManager:
    def __init__(self):
        self.connected = False
        self.data_stream = []
    
    def simulate_websocket_connection(self):
        """Simulate WebSocket connection for demo"""
        try:
            st.session_state.websocket_connected = True
            return True
        except:
            return False
    
    def simulate_realtime_data(self):
        """Simulate receiving real-time data"""
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai']
        areas = ['Airport', 'Business District', 'Mall']
        
        current_time = datetime.datetime.now()
        city = np.random.choice(cities)
        area = np.random.choice(areas)
        
        current_hour = current_time.hour
        is_peak = current_hour in [7, 8, 9, 18, 19, 20]
        
        base_demand = 40 + (25 if is_peak else 0) + np.random.uniform(-10, 15)
        base_supply = 35 + np.random.uniform(-8, 12)
        
        demand_supply_ratio = base_demand / max(base_supply, 1)
        
        if demand_supply_ratio <= 1.0:
            surge = 1.0
        elif demand_supply_ratio <= 1.5:
            surge = 1.0 + (demand_supply_ratio - 1.0) * 0.8
        else:
            surge = min(4.0, 1.4 + (demand_supply_ratio - 1.5) * 1.2)
        
        data_point = {
            'timestamp': current_time,
            'city': city,
            'area': area,
            'demand': round(base_demand, 1),
            'supply': round(base_supply, 1),
            'surge_multiplier': round(surge, 2),
            'status': 'HIGH' if surge > 2.0 else 'MEDIUM' if surge > 1.5 else 'NORMAL',
            'temperature': np.random.uniform(20, 40),
            'humidity': np.random.uniform(40, 80)
        }
        
        return data_point

# Initialize WebSocket manager
ws_manager = WebSocketManager()

def render_dashboard_tab(df, selected_city, selected_area):
    """Render the main dashboard tab"""
    
    # Filter data
    filtered_df = df.copy()
    if selected_city != 'All Cities':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    if selected_area != 'All Areas':
        filtered_df = filtered_df[filtered_df['area'] == selected_area]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_surge = filtered_df['surge_multiplier'].mean()
        st.metric("Average Surge", f"{avg_surge:.2f}x")
    
    with col2:
        total_rides = filtered_df['completed_rides'].sum()
        st.metric("Total Rides", f"{total_rides:,.0f}")
    
    with col3:
        total_revenue = filtered_df['revenue'].sum()
        if total_revenue >= 10000000:
            revenue_display = f"₹{total_revenue/10000000:.1f}Cr"
        elif total_revenue >= 100000:
            revenue_display = f"₹{total_revenue/100000:.1f}L"
        else:
            revenue_display = f"₹{total_revenue/1000:.1f}K"
        st.metric("Revenue", revenue_display)
    
    with col4:
        peak_surge = filtered_df['surge_multiplier'].max()
        st.metric("Peak Surge", f"{peak_surge:.1f}x")
    
    with col5:
        avg_utilization = filtered_df['driver_utilization'].mean()
        st.metric("Driver Utilization", f"{avg_utilization:.0f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("24-Hour Demand vs Supply Pattern")
    
        hourly_data = filtered_df.groupby('hour').agg({
            'demand': 'mean',
            'supply': 'mean',
            'surge_multiplier': 'mean'
        }).reset_index()
    
        if not hourly_data.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
    
            fig.add_trace(
                go.Scatter(x=hourly_data['hour'], y=hourly_data['demand'],
                           mode='lines+markers', name='Demand',
                           line=dict(color='#EF4444', width=3)),
                secondary_y=False
            )
    
            fig.add_trace(
                go.Scatter(x=hourly_data['hour'], y=hourly_data['supply'],
                           mode='lines+markers', name='Supply',
                           line=dict(color='#10B981', width=3)),
                secondary_y=False
            )
    
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Demand/Supply Level", secondary_y=False)
            fig.update_layout(height=400)
    
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("City Performance Comparison")
    
        city_performance = filtered_df.groupby('city').agg({
            'surge_multiplier': 'mean',
            'revenue': 'sum'
        }).reset_index()
    
        if not city_performance.empty and len(city_performance) > 1:
            fig = px.bar(city_performance, x='surge_multiplier', y='city',
                         orientation='h', color='revenue',
                         title="Average Surge by City")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select 'All Cities' to see city comparison")
def render_predictions_tab(df, selected_city, selected_area):
    """Render the predictions tab with REAL ML integration"""
    
    st.subheader("🔮 Advanced Predictive Analytics")
    
    # Model training section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 Machine Learning Model")
    
        if st.button("🚀 Train Random Forest Model"):
            with st.spinner("Training Random Forest model on 30 days of data..."):
                try:
                    model, scaler, metrics = train_surge_prediction_model(df)
    
                    # Save to session state
                    st.session_state.trained_model = model
                    st.session_state.model_scaler = scaler
                    st.session_state.model_metrics = metrics
                    st.session_state.model_trained = True
    
                    st.success("✅ Random Forest model trained successfully!")
    
                    # Show metrics
                    st.markdown(f"""
                    <div class="model-status">
                        <h4>📊 Model Performance</h4>
                        <p><strong>R² Score:</strong> {metrics['r2']:.3f}</p>
                        <p><strong>RMSE:</strong> {metrics['rmse']:.3f}</p>
                        <p><strong>Training Samples:</strong> {metrics['training_samples']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")
    
        if st.session_state.model_trained:
            st.markdown("""
            <div class="model-status">
                <h4>✅ Random Forest Model Ready</h4>
                <p>100 trees, max depth 10</p>
                <p>Features: demand, supply, time, weather, location</p>
            </div>
            """, unsafe_allow_html=True)
    
            # Feature importance
            if st.session_state.model_metrics:
                st.markdown("#### 🎯 Feature Importance")
                importance_df = pd.DataFrame(
                    list(st.session_state.model_metrics['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
    
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎲 ML-Powered Prediction")
    
        if st.session_state.model_trained:
            # Input parameters
            pred_demand = st.number_input("Expected Demand", 10, 200, 60, key='pred_demand')
            pred_supply = st.number_input("Expected Supply", 10, 150, 45, key='pred_supply')
            pred_city = st.selectbox("City", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], key='pred_city')
            pred_area = st.selectbox("Area", ['Airport', 'Business District', 'Mall'], key='pred_area')
            pred_weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Light Rain', 'Heavy Rain'], key='pred_weather')
            pred_temp = st.slider("Temperature (°C)", 15, 45, 30, key='pred_temp')
            pred_humidity = st.slider("Humidity (%)", 30, 90, 60, key='pred_humidity')
    
            current_time = datetime.datetime.now()
            pred_hour = st.slider("Hour", 0, 23, current_time.hour, key='pred_hour')
            pred_weekend = st.checkbox("Weekend", current_time.weekday() >= 5, key='pred_weekend')
    
            if st.button("🔮 Predict with ML Model"):
                prediction = predict_surge_with_model(
                    pred_demand, pred_supply, pred_hour, current_time.weekday(),
                    pred_weekend, pred_city, pred_area, pred_weather, pred_temp, pred_humidity
                )
    
                if prediction:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 ML Prediction: {prediction:.2f}x</h2>
                        <h4>{"🔴 HIGH SURGE" if prediction > 2.0 else "🟡 MEDIUM SURGE" if prediction > 1.5 else "🟢 NORMAL"}</h4>
                        <p>Confidence: Random Forest Model</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction failed. Please check model training.")
        else:
            st.info("Train the model first to enable ML predictions")
    
    # Forecast section with real ML
    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("📈 6-Hour ML Forecast")
    
        col1, col2 = st.columns([1, 1])
    
        with col1:
            forecast_city = st.selectbox("Forecast City", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], key='forecast_city_key')
        
        with col2:
            forecast_area = st.selectbox("Forecast Area", ['Airport', 'Business District', 'Mall'], key='forecast_area_key')
            forecast_weather = st.selectbox("Expected Weather", ['Sunny', 'Cloudy', 'Light Rain'], key='forecast_weather_key')
    
        if st.button("📊 Generate ML Forecast"):
            forecast_hours = []
            forecast_surge = []
    
            base_demand = 50
            base_supply = 40
    
            for h in range(1, 7):
                future_time = datetime.datetime.now() + datetime.timedelta(hours=h)
                hour = future_time.hour
                is_weekend = future_time.weekday() >= 5
    
                # Adjust demand/supply based on hour
                hour_demand = base_demand + (20 if hour in [7, 8, 9, 18, 19, 20] else 0)
                hour_supply = base_supply - (10 if hour < 6 or hour > 22 else 0)
    
                prediction = predict_surge_with_model(
                    hour_demand, hour_supply, hour, future_time.weekday(),
                    is_weekend, forecast_city, forecast_area, forecast_weather, 25, 50
                )
    
                if prediction:
                    forecast_hours.append(future_time.strftime('%H:00'))
                    forecast_surge.append(prediction)
    
            if forecast_surge:
                # Create forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_hours,
                    y=forecast_surge,
                    mode='lines+markers',
                    name='ML Predicted Surge',
                    line=dict(color='purple', width=3),
                    marker=dict(size=8)
                ))
    
                fig.update_layout(
                    title=f'ML Forecast - {forecast_city}, {forecast_area}',
                    xaxis_title='Time',
                    yaxis_title='Predicted Surge Multiplier',
                    height=400
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
                # Show alerts
                high_surge_hours = [h for h, s in zip(forecast_hours, forecast_surge) if s > 2.0]
                if high_surge_hours:
                    st.error(f"🚨 ML PREDICTS HIGH SURGE at: {', '.join(high_surge_hours)}")
                else:
                    st.success("🟢 ML predicts normal surge levels")
                # ... (rest of the render_predictions_tab function code) ...

    # Forecast section with real ML
    if st.session_state.model_trained:
        # ... (code for the 6-hour forecast) ...
    
        # Add the new API integration section here
        st.markdown("---")
        st.subheader("🌐 API Integration Example")
        st.markdown("This section demonstrates how an external system could use an API to get a surge prediction.")
    
        st.markdown("#### ➡️ API Request (Example)")
        st.code("""
    import requests
    import json
    
    url = "https://your-api-endpoint.com/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "city": "Mumbai",
        "area": "Airport",
        "demand": 120,
        "supply": 75,
        "hour": 18,
        "weather": "Heavy Rain",
        "temperature": 28,
        "humidity": 85
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
    """)
    
        st.markdown("#### ⬅️ API Response (Example)")
        st.code("""
    {
        "surge_multiplier": 2.25,
        "predicted_status": "HIGH_SURGE",
        "timestamp": "2025-09-09T13:00:00Z"
    }
    """)

def render_realtime_tab():
    """Render the real-time monitoring tab with WebSocket simulation"""
    
    st.subheader("⚡ Real-time Monitoring with WebSocket")
    
    # WebSocket connection status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔌 WebSocket Connection")
    
        if st.button("🔗 Connect WebSocket"):
            if ws_manager.simulate_websocket_connection():
                st.session_state.websocket_connected = True
                st.success("✅ WebSocket connected!")
            else:
                st.error("❌ WebSocket connection failed")
    
        if st.session_state.websocket_connected:
            st.markdown("""
            <div class="websocket-status">
                🟢 <strong>WebSocket CONNECTED</strong><br>
                📡 Receiving live data stream<br>
                ⚡ Auto-refresh: ON
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="websocket-status">
                🔴 <strong>WebSocket DISCONNECTED</strong><br>
                📡 Click 'Connect WebSocket' to start
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 System Metrics")
    
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Connected Users", "1,247")
            st.metric("Active Drivers", "3,891")
    
        with col2b:
            st.metric("Messages/sec", "45" if st.session_state.websocket_connected else "0")
            st.metric("Latency", "23ms" if st.session_state.websocket_connected else "∞")
    
    # Live data streaming
    if st.session_state.websocket_connected:
        st.markdown("---")
        st.subheader("📺 Live Data Stream")
    
        # Auto-refresh placeholder
        live_data_placeholder = st.empty()
    
        if st.button("📡 Fetch Live Data") or len(st.session_state.realtime_data_stream) == 0:
            # Simulate receiving data via WebSocket
            new_data = ws_manager.simulate_realtime_data()
            st.session_state.realtime_data_stream.append(new_data)
    
            # Keep only last 10 data points
            if len(st.session_state.realtime_data_stream) > 10:
                st.session_state.realtime_data_stream = st.session_state.realtime_data_stream[-10:]
    
        # Display latest data
        if st.session_state.realtime_data_stream:
            latest_data = st.session_state.realtime_data_stream[-1]
    
            status_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'NORMAL': '🟢'
            }.get(latest_data['status'], '🟢')
    
            st.markdown(f"""
            <div class="live-data-box">
                <h4>📍 {latest_data['city']} - {latest_data['area']} (LIVE)</h4>
                <p><strong>Surge:</strong> {latest_data['surge_multiplier']}x {status_color}</p>
                <p><strong>Demand:</strong> {latest_data['demand']} | <strong>Supply:</strong> {latest_data['supply']}</p>
                <p><strong>Temp:</strong> {latest_data['temperature']:.1f}°C | <strong>Humidity:</strong> {latest_data['humidity']:.0f}%</p>
                <p><strong>WebSocket Time:</strong> {latest_data['timestamp'].strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
    
            # Real-time chart of last 10 data points
            if len(st.session_state.realtime_data_stream) > 1:
                st.subheader("📈 Live Surge Trend")
    
                stream_df = pd.DataFrame(st.session_state.realtime_data_stream)
    
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(stream_df))),
                    y=stream_df['surge_multiplier'],
                    mode='lines+markers',
                    name='Live Surge',
                    line=dict(color='red', width=3),
                    marker=dict(size=8, color=stream_df['surge_multiplier'],
                                 colorscale='RdYlGn_r', showscale=True)
                ))
    
                fig.update_layout(
                    title='Real-time Surge Data (WebSocket Stream)',
                    xaxis_title='Data Points (Latest 10)',
                    yaxis_title='Surge Multiplier',
                    height=300
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
    # Real-time alerts
    st.markdown("---")
    st.subheader("🚨 Live Alerts")
    
    if st.session_state.websocket_connected and st.session_state.realtime_data_stream:
        latest = st.session_state.realtime_data_stream[-1]
        if latest['surge_multiplier'] > 2.0:
            st.error(f"🚨 HIGH SURGE ALERT: {latest['surge_multiplier']}x in {latest['city']} - {latest['area']}")
        elif latest['surge_multiplier'] > 1.5:
            st.warning(f"⚠️ MEDIUM SURGE: {latest['surge_multiplier']}x in {latest['city']} - {latest['area']}")
        else:
            st.success("✅ All monitored areas operating normally")
    else:
        st.info("Connect WebSocket to receive live alerts")

def render_testing_tab():
    """Render the A/B testing tab"""
    
    st.subheader("🧪 A/B Testing Framework")
    
    # A/B Testing Controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Test Configuration")
    
        test_name = st.text_input("Test Name", "Surge Algorithm V2")
        test_cities = st.multiselect("Test Cities",
                                     ['Mumbai', 'Delhi', 'Bangalore'],
                                     default=['Mumbai'])
        test_duration = st.slider("Test Duration (days)", 1, 30, 7)
        traffic_split = st.slider("Traffic Split (%)", 10, 50, 20)
    
        if st.button("🚀 Launch A/B Test"):
            st.success(f"✅ A/B Test '{test_name}' launched!")
            st.info(f"📊 {traffic_split}% of users in {', '.join(test_cities)} will see the new algorithm")
    
    with col2:
        st.markdown("#### 📊 Active Tests")
    
        # Mock active tests
        active_tests = [
            {"name": "Surge Algorithm V2", "status": "Running", "days_left": 3, "conversion": "+5.2%"},
            {"name": "Peak Hour Pricing", "status": "Analysis", "days_left": 0, "conversion": "+2.8%"},
        ]
    
        for test in active_tests:
            status_color = "🟢" if test["status"] == "Running" else "🟡"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;">
                <strong>{status_color} {test['name']}</strong><br>
                Status: {test['status']}<br>
                Days Left: {test['days_left']}<br>
                Conversion: {test['conversion']}
            </div>
            """, unsafe_allow_html=True)
    
    # Test Results
    st.markdown("---")
    st.subheader("📈 Test Results & Statistical Analysis")
    
    # Mock A/B test results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Control Group Revenue", "₹2.4L", "-1.2%")
        st.metric("Control Conversion Rate", "78.5%", "-0.8%")
    
    with col2:
        st.metric("Test Group Revenue", "₹2.5L", "+5.2%")
        st.metric("Test Conversion Rate", "82.3%", "+3.8%")
    
    with col3:
        st.metric("Statistical Significance", "95.2%", "+2.1%")
        st.metric("Revenue Uplift", "+4.2%", "🟢")
    
    # Statistical significance chart
    st.subheader("📊 Statistical Significance Over Time")
    
    days = list(range(1, 8))
    significance = [45, 62, 78, 85, 91, 94, 95.2]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days,
        y=significance,
        mode='lines+markers',
        name='Statistical Significance',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_hline(y=95, line_dash="dash", line_color="green",
                  annotation_text="95% Confidence Threshold")
    
    fig.update_layout(
        title='A/B Test Statistical Significance',
        xaxis_title='Days',
        yaxis_title='Significance (%)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Uber Surge Pricing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Dynamic Pricing Analytics with ML & WebSocket Integration</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading surge data for ML training...'):
        df = load_data()
    
        if df.empty:
            st.error("Failed to generate data. Please refresh the page.")
            return
    
    # Sidebar filters
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Integration status in sidebar
    st.sidebar.markdown("### 🔧 System Status")
    
    ml_status = "🟢 Ready" if st.session_state.model_trained else "🔴 Not Trained"
    ws_status = "🟢 Connected" if st.session_state.websocket_connected else "🔴 Disconnected"
    
    st.sidebar.markdown(f"""
    **ML Model:** {ml_status}  
    **WebSocket:** {ws_status}  
    **Data Points:** {len(df):,}
    """)
    
    st.sidebar.markdown("---")
    
    available_cities = sorted(df['city'].unique().tolist())
    selected_city = st.sidebar.selectbox(
        "🏙️ Select City",
        options=['All Cities'] + available_cities,
        index=0
    )
    
    available_areas = sorted(df['area'].unique().tolist())
    selected_area = st.sidebar.selectbox(
        "📍 Select Area Type",
        options=['All Areas'] + available_areas
    )
    
    # Quick actions in sidebar
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🧠 Quick Train ML Model"):
        if not st.session_state.model_trained:
            with st.spinner("Training model..."):
                try:
                    model, scaler, metrics = train_surge_prediction_model(df)
                    st.session_state.trained_model = model
                    st.session_state.model_scaler = scaler
                    st.session_state.model_metrics = metrics
                    st.session_state.model_trained = True
                    st.sidebar.success("✅ Model trained!")
                except Exception as e:
                    st.sidebar.error(f"❌ Training failed: {str(e)}")
        else:
            st.sidebar.info("Model already trained!")
    
    if st.sidebar.button("🔗 Quick Connect WebSocket"):
        if ws_manager.simulate_websocket_connection():
            st.session_state.websocket_connected = True
            st.sidebar.success("✅ WebSocket connected!")
        else:
            st.sidebar.error("❌ Connection failed")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔮 ML Predictions",
        "⚡ WebSocket Live",
        "🧪 A/B Testing"
    ])
    
    with tab1:
        render_dashboard_tab(df, selected_city, selected_area)
    
    with tab2:
        render_predictions_tab(df, selected_city, selected_area)
    
    with tab3:
        render_realtime_tab()
    
    with tab4:
        render_testing_tab()

if __name__ == "__main__":
    main()