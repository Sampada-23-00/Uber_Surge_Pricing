"""
SURGE_COMMAND — Uber Surge Pricing Operations Dashboard
========================================================
An operations-center-inspired dark-mode dashboard for the Indian
Ride-Hailing Surge Pricing Prediction system.

Run:
    streamlit run streamlit_app.py
"""

import json
import pickle
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.models.predict import load_artifacts, predict_single

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SURGE_COMMAND // India Ops",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — SURGE_COMMAND Design Language
# ──────────────────────────────────────────────────────────────────────────────

SURGE_CSS = """
<style>
/* ——— Google Fonts ——— */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ——— Global ——— */
html, body, .stApp {
    background-color: #0a0a0f !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ——— Hide Streamlit branding ——— */
#MainMenu, footer, header {visibility: hidden;}

/* ——— Sidebar ——— */
section[data-testid="stSidebar"] {
    background-color: #0d0d14 !important;
    border-right: 1px solid #1a1a2e !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label {
    color: #5a5a7a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
}

/* ——— Metric cards ——— */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #12121e 0%, #0d0d17 100%) !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 4px !important;
    padding: 18px 14px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #276EF1;
    border-radius: 4px 0 0 4px;
}
div[data-testid="stMetric"]:hover {
    border-color: #276EF1 !important;
    box-shadow: 0 0 20px rgba(39, 110, 241, 0.1), inset 0 0 20px rgba(39, 110, 241, 0.03) !important;
}
div[data-testid="stMetric"] label {
    color: #4a4a6a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #e0e0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ——— Buttons ——— */
div.stButton > button {
    width: 100% !important;
    border-radius: 3px !important;
    height: 3.2em !important;
    background: linear-gradient(135deg, #276EF1 0%, #1a4fd4 100%) !important;
    color: white !important;
    border: 1px solid #3a7ff5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-size: 0.75rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 15px rgba(39, 110, 241, 0.2) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #3a7ff5 0%, #276EF1 100%) !important;
    box-shadow: 0 0 30px rgba(39, 110, 241, 0.4), 0 0 60px rgba(39, 110, 241, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* ——— Input boxes ——— */
div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background-color: #0f0f1a !important;
    border-radius: 3px !important;
    border: 1px solid #1e1e35 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #276EF1 !important;
    box-shadow: 0 0 10px rgba(39, 110, 241, 0.15) !important;
}

/* ——— Tabs ——— */
div.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: #0d0d17;
    border: 1px solid #1e1e35;
    border-radius: 3px;
    padding: 3px;
}
div.stTabs [data-baseweb="tab"] {
    border-radius: 2px;
    color: #5a5a7a;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 20px;
}
div.stTabs [aria-selected="true"] {
    background-color: #276EF1 !important;
    color: white !important;
    box-shadow: 0 0 15px rgba(39, 110, 241, 0.3);
}

/* ——— Dataframes ——— */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e1e35 !important;
    border-radius: 3px !important;
}

/* ——— Divider ——— */
hr {
    border-color: #1a1a2e !important;
}

/* ——— Expander ——— */
details {
    background-color: #0d0d17 !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 3px !important;
}

/* ——— Scrollbar ——— */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1e1e35; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #276EF1; }

/* ——— Animations ——— */
@keyframes pulse-glow {
    0%   { box-shadow: 0 0 5px rgba(39, 110, 241, 0.3); }
    50%  { box-shadow: 0 0 20px rgba(39, 110, 241, 0.6); }
    100% { box-shadow: 0 0 5px rgba(39, 110, 241, 0.3); }
}
@keyframes scan-line {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}
@keyframes blink-dot {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
}
.live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #06C167;
    border-radius: 50%;
    animation: blink-dot 1.5s infinite;
    margin-right: 6px;
}
.critical-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #E11900;
    border-radius: 50%;
    animation: blink-dot 0.8s infinite;
    margin-right: 6px;
}
</style>
"""

st.markdown(SURGE_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = "models/"
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
WEATHER_OPTIONS = ["Clear", "Light_Rain", "Heavy_Rain", "Hot", "Humid", "Dusty"]
EVENTS = ["Regular", "Diwali", "Holi", "Eid"]
SEGMENTS = ["Budget", "Regular", "Premium"]
DAYS = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}

CITY_COORDS = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
}

# City zone codes for the ops-center feel
CITY_ZONES = {
    "Mumbai": "ZONE_MUM",
    "Delhi": "ZONE_DEL",
    "Bangalore": "ZONE_BLR",
    "Hyderabad": "ZONE_HYD",
    "Chennai": "ZONE_CHE",
    "Kolkata": "ZONE_KOL",
    "Pune": "ZONE_PUN",
    "Ahmedabad": "ZONE_AHM",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def get_surge_color(surge: float) -> str:
    """Return color based on surge level."""
    if surge < 1.3:
        return "#06C167"
    elif surge < 1.8:
        return "#276EF1"
    elif surge < 2.5:
        return "#F6A623"
    else:
        return "#E11900"


def get_surge_status(surge: float) -> str:
    """Return ops-center status string."""
    if surge < 1.3:
        return "NOMINAL"
    elif surge < 1.8:
        return "ELEVATED"
    elif surge < 2.5:
        return "HIGH_ALERT"
    else:
        return "CRITICAL"


def get_status_dot(surge: float) -> str:
    """Return HTML for status indicator dot."""
    if surge < 1.3:
        return '<span style="display:inline-block;width:8px;height:8px;background:#06C167;border-radius:50%;box-shadow:0 0 6px #06C167;"></span>'
    elif surge < 1.8:
        return '<span style="display:inline-block;width:8px;height:8px;background:#276EF1;border-radius:50%;box-shadow:0 0 6px #276EF1;"></span>'
    elif surge < 2.5:
        return '<span style="display:inline-block;width:8px;height:8px;background:#F6A623;border-radius:50%;box-shadow:0 0 6px #F6A623;"></span>'
    else:
        return '<span class="critical-dot" style="width:8px;height:8px;box-shadow:0 0 6px #E11900;"></span>'


def build_business_context(city, hour, weather, event, wait_time, segment) -> list:
    """Generate human-readable surge reasons."""
    reasons = []
    if weather in ("Heavy_Rain", "Light_Rain"):
        reasons.append(("WEATHER_ALERT", f"{weather.replace('_', ' ').upper()} — reduced driver supply", "#F6A623"))
    if event != "Regular":
        reasons.append(("EVENT_SPIKE", f"{event.upper()} festival — demand surge detected", "#E11900"))
    if hour in {18, 19, 20, 21}:
        reasons.append(("PEAK_HOURS", "Evening peak window active (1800–2100)", "#276EF1"))
    if hour in {22, 23, 0, 1}:
        reasons.append(("LOW_SUPPLY", "Late night — driver availability critical", "#E11900"))
    if hour in {8, 9, 10}:
        reasons.append(("COMMUTE_PEAK", "Morning commute window (0800–1000)", "#276EF1"))
    if wait_time > 15:
        reasons.append(("DRIVER_SHORTAGE", f"Wait time {wait_time}min — supply deficit", "#F6A623"))
    if segment == "Budget":
        reasons.append(("SEGMENT_FLAG", "Budget segment — high price sensitivity", "#5a5a7a"))
    return reasons


def generate_surge_map_data(city: str, surge: float):
    """Generate synthetic surge heatmap data around the selected city."""
    center = CITY_COORDS[city]
    np.random.seed(42)
    n_points = 200

    lats = center["lat"] + np.random.normal(0, 0.03, n_points)
    lons = center["lon"] + np.random.normal(0, 0.03, n_points)
    distances = np.sqrt((lats - center["lat"])**2 + (lons - center["lon"])**2)
    intensities = surge * np.exp(-distances / 0.02) + np.random.uniform(0.5, 1.0, n_points)

    return pd.DataFrame({"lat": lats, "lon": lons, "surge": intensities})


def utc_timestamp():
    """Return formatted UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("UTC %H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Mission Control Input
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # SURGE_COMMAND branding
    st.markdown("""
        <div style="text-align: center; padding: 16px 0 24px 0;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem;
                        font-weight: 800; letter-spacing: 3px; color: #e0e0ff;">
                SURGE_COMMAND
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        font-weight: 500; color: #3a3a5a; letter-spacing: 2px;
                        margin-top: 4px;">
                V3.0_ACTIVE
            </div>
            <div style="width: 100%; height: 1px;
                        background: linear-gradient(90deg, transparent, #276EF1, transparent);
                        margin: 14px auto 0;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation-style section
    st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                    color: #3a3a5a; letter-spacing: 2px; text-transform: uppercase;
                    margin-bottom: 12px; padding-left: 4px;">
            ▸ MARKET_PARAMETERS
        </div>
    """, unsafe_allow_html=True)

    city = st.selectbox("MARKET", CITIES, index=0)
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("HOUR", 0, 23, 19)
    with col2:
        day_of_week = st.selectbox("DAY", options=list(DAYS.keys()),
                                    format_func=lambda x: DAYS[x][:3].upper(), index=1)

    st.markdown("""
        <div style="width: 100%; height: 1px;
                    background: linear-gradient(90deg, transparent, #1e1e35, transparent);
                    margin: 10px 0;"></div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                    color: #3a3a5a; letter-spacing: 2px; text-transform: uppercase;
                    margin-bottom: 12px; padding-left: 4px;">
            ▸ ENVIRONMENT_CONDITIONS
        </div>
    """, unsafe_allow_html=True)

    weather = st.selectbox("WEATHER_STATE", WEATHER_OPTIONS, index=2)
    special_event = st.selectbox("EVENT_FLAG", EVENTS, index=0)

    st.markdown("""
        <div style="width: 100%; height: 1px;
                    background: linear-gradient(90deg, transparent, #1e1e35, transparent);
                    margin: 10px 0;"></div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                    color: #3a3a5a; letter-spacing: 2px; text-transform: uppercase;
                    margin-bottom: 12px; padding-left: 4px;">
            ▸ TRIP_TELEMETRY
        </div>
    """, unsafe_allow_html=True)

    distance_km = st.slider("DISTANCE_KM", 1.0, 50.0, 8.5, 0.5)
    base_fare = st.number_input("BASE_FARE_INR", min_value=10.0, max_value=200.0,
                                 value=25.0, step=5.0)
    customer_segment = st.selectbox("CUSTOMER_SEGMENT", SEGMENTS, index=0)
    wait_time_mins = st.slider("WAIT_TIME_MIN", 0, 60, 15)

    st.markdown("""
        <div style="width: 100%; height: 1px;
                    background: linear-gradient(90deg, transparent, #1e1e35, transparent);
                    margin: 14px 0;"></div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button("⚡  EXECUTE_PREDICTION", use_container_width=True)

    # Sidebar footer
    st.markdown(f"""
        <div style="position: fixed; bottom: 0; padding: 12px 0; width: 100%;">
            <div style="width: 90%; height: 1px;
                        background: linear-gradient(90deg, transparent, #1e1e35, transparent);
                        margin-bottom: 10px;"></div>
            <div style="display: flex; align-items: center; gap: 6px; padding-left: 8px;">
                <span class="live-dot"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                             color: #06C167; letter-spacing: 1px;">SYSTEM_ONLINE</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                        color: #2a2a45; letter-spacing: 1px; margin-top: 4px; padding-left: 8px;">
                RF_MODEL // 26_FEATURES
            </div>
        </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Top Bar
# ──────────────────────────────────────────────────────────────────────────────

now_ts = utc_timestamp()

st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
                padding: 8px 0 4px 0; border-bottom: 1px solid #1e1e35; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
                         font-weight: 700; color: #5a5a7a; letter-spacing: 2px;">
                PRECISION DATA & MOTION
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                         color: #3a3a5a; letter-spacing: 1px;">
                {now_ts}
            </span>
            <span class="live-dot"></span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────

try:
    _, _, metadata = load_artifacts(MODEL_DIR)
    model_loaded = True
except Exception:
    model_loaded = False
    st.markdown("""
        <div style="background: #1a0a0a; border: 1px solid #E11900; border-radius: 3px;
                    padding: 16px; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #E11900; font-weight: 700; font-size: 0.75rem; letter-spacing: 1px;">
                ⚠ SYSTEM_ERROR
            </span>
            <span style="color: #888; font-size: 0.75rem; margin-left: 12px;">
                Model artifacts not found. Execute: python pipeline.py
            </span>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_predict, tab_scenarios, tab_model, tab_data = st.tabs([
    "⚡ MARKETS",
    "🧪 SCENARIOS",
    "📊 MODELS",
    "📁 DATA_EXPLORER",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MARKETS — SURGE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    if predict_clicked:
        ride_input = {
            "city": city,
            "hour": hour,
            "day_of_week": day_of_week,
            "weather": weather,
            "special_event": special_event,
            "distance_km": distance_km,
            "base_fare": base_fare,
            "customer_segment": customer_segment,
            "wait_time_mins": wait_time_mins,
            "ride_completed": 1,
        }

        with st.spinner("COMPUTING_SURGE..."):
            result = predict_single(ride_input, model_dir=MODEL_DIR)

        surge = result["surge_multiplier"]
        category = result["surge_category"]
        estimated_fare = round(base_fare * surge, 2)
        surge_color = get_surge_color(surge)
        surge_status = get_surge_status(surge)
        status_dot = get_status_dot(surge)
        reasons = build_business_context(city, hour, weather, special_event,
                                          wait_time_mins, customer_segment)
        zone_code = CITY_ZONES[city]

        # ── Two-column layout like the reference ──
        main_col, pulse_col = st.columns([2.2, 1])

        with main_col:
            # Active Market card overlay on map area
            st.markdown(f"""
                <div style="position: relative;">
                    <div style="background: #0d0d17; border: 1px solid #1e1e35;
                                border-radius: 3px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                            color: #5a5a7a; letter-spacing: 2px; text-transform: uppercase;">
                                    ACTIVE MARKET
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                                            font-weight: 800; color: #e0e0ff; letter-spacing: 1px; margin-top: 4px;">
                                    {city.upper()}_DISTRICT
                                </div>
                                <div style="display: flex; gap: 28px; margin-top: 12px;">
                                    <div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                                    color: #5a5a7a; letter-spacing: 1px;">AVG_SURGE</div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                                                    font-weight: 800; color: {surge_color};">{surge}×</div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                                    color: #5a5a7a; letter-spacing: 1px;">ZONE_ID</div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                                                    font-weight: 800; color: #e0e0ff;">{zone_code}</div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                                    color: #5a5a7a; letter-spacing: 1px;">STATUS</div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
                                                    font-weight: 700; color: {surge_color}; margin-top: 4px;
                                                    display: flex; align-items: center; gap: 6px;">
                                            {status_dot} {surge_status}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                        color: #3a3a5a; text-align: right;">
                                {now_ts}
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Surge Heatmap
            map_data = generate_surge_map_data(city, surge)
            coords = CITY_COORDS[city]

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(
                    latitude=coords["lat"],
                    longitude=coords["lon"],
                    zoom=12,
                    pitch=45,
                ),
                layers=[
                    pdk.Layer(
                        "HexagonLayer",
                        data=map_data,
                        get_position=["lon", "lat"],
                        radius=150,
                        elevation_scale=40,
                        elevation_range=[0, 800],
                        pickable=True,
                        extruded=True,
                        color_range=[
                            [6, 193, 103],
                            [255, 255, 178],
                            [254, 204, 92],
                            [253, 141, 60],
                            [240, 59, 32],
                            [189, 0, 38],
                        ],
                    ),
                ],
            ), use_container_width=True)

            # Legend bar
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;
                            background: #0d0d17; border: 1px solid #1e1e35; border-radius: 3px;
                            padding: 10px 16px; margin-top: 8px;">
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="width: 10px; height: 10px; background: #06C167;
                                         border-radius: 2px; display: inline-block;"></span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                         color: #5a5a7a; letter-spacing: 1px;">BASELINE (1.0×)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="width: 10px; height: 10px; background: #E11900;
                                         border-radius: 2px; display: inline-block;"></span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                         color: #5a5a7a; letter-spacing: 1px;">HIGH_SURGE (>2.5×)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span style="width: 10px; height: 10px; background: #2a2a45;
                                         border-radius: 2px; display: inline-block; border: 1px solid #3a3a5a;"></span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                         color: #5a5a7a; letter-spacing: 1px;">UNMAPPED_ZONE</span>
                        </div>
                    </div>
                    <div style="background: linear-gradient(135deg, #276EF1 0%, #1a4fd4 100%);
                                border: 1px solid #3a7ff5; border-radius: 3px; padding: 6px 16px;
                                font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                                font-weight: 700; color: white; letter-spacing: 1.5px;
                                box-shadow: 0 0 10px rgba(39, 110, 241, 0.2); cursor: pointer;">
                        LIVE_TELEMETRY_STREAM
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with pulse_col:
            # Market Pulse header
            st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                color: #5a5a7a; letter-spacing: 2px;">MARKET PULSE</div>
                    <div style="display: flex; align-items: baseline; gap: 10px;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                                     font-weight: 800; color: #e0e0ff; letter-spacing: 1px;">
                            SYSTEM_LOAD
                        </span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                     color: #3a3a5a;">{now_ts}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Simulated live metrics
            np.random.seed(int(surge * 100))
            active_riders = int(np.random.normal(12000, 2000))
            avail_partners = int(np.random.normal(3000, 500))
            unmet_demand = int(max(0, active_riders - avail_partners * 4) * surge)
            rider_delta = round(np.random.uniform(-3, 6), 1)
            partner_delta = round(np.random.uniform(-5, 2), 1)

            # Active Riders card
            delta_color_r = "#06C167" if rider_delta >= 0 else "#E11900"
            st.markdown(f"""
                <div style="background: #0d0d17; border: 1px solid #1e1e35; border-radius: 3px;
                            padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1rem;">🚶</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                         color: #5a5a7a; letter-spacing: 1px;">ACTIVE_RIDERS</span>
                        </div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                                     font-weight: 700; color: {delta_color_r}; background: {delta_color_r}18;
                                     padding: 2px 8px; border-radius: 2px; letter-spacing: 1px;">
                            {'+' if rider_delta >= 0 else ''}{rider_delta}%
                        </span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem;
                                font-weight: 800; color: #e0e0ff; margin-top: 8px;">
                        {active_riders:,}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Available Partners card
            delta_color_p = "#06C167" if partner_delta >= 0 else "#E11900"
            st.markdown(f"""
                <div style="background: #0d0d17; border: 1px solid #1e1e35; border-radius: 3px;
                            padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1rem;">🚗</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                         color: #5a5a7a; letter-spacing: 1px;">AVAILABLE_PARTNERS</span>
                        </div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                                     font-weight: 700; color: {delta_color_p}; background: {delta_color_p}18;
                                     padding: 2px 8px; border-radius: 2px; letter-spacing: 1px;">
                            {'+' if partner_delta >= 0 else ''}{partner_delta}%
                        </span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem;
                                font-weight: 800; color: #e0e0ff; margin-top: 8px;">
                        {avail_partners:,}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Unmet Demand / Critical threshold card
            is_critical = surge >= 2.5
            card_bg = "#1a0505" if is_critical else "#0d0d17"
            card_border = "#E11900" if is_critical else "#1e1e35"
            threshold_label = "CRITICAL_THRESHOLD" if is_critical else "ELEVATED" if surge >= 1.8 else "STABLE"
            threshold_color = "#E11900" if is_critical else "#F6A623" if surge >= 1.8 else "#06C167"

            st.markdown(f"""
                <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 3px;
                            padding: 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1rem;">{'⚠' if is_critical else '📊'}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                     font-weight: 700; color: {threshold_color}; background: {threshold_color}18;
                                     padding: 2px 8px; border-radius: 2px; letter-spacing: 1px;">
                            {threshold_label}
                        </span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.5rem;
                                font-weight: 800; color: {threshold_color}; margin-top: 8px;">
                        {unmet_demand:,}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                color: #5a5a7a; letter-spacing: 1.5px; margin-top: 4px;">
                        UNMET_DEMAND<br/>
                        <span style="font-size: 0.5rem; letter-spacing: 1px;">POTENTIAL LOST GMV</span>
                    </div>
                    <div style="background: #1e1e35; border-radius: 2px; height: 4px; margin-top: 10px;">
                        <div style="background: {threshold_color}; height: 4px; border-radius: 2px;
                                    width: {min(surge / 3.5 * 100, 100)}%;
                                    box-shadow: 0 0 8px {threshold_color}40;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Estimated fare card
            st.markdown(f"""
                <div style="background: #0d0d17; border: 1px solid #1e1e35; border-radius: 3px;
                            padding: 16px; margin-bottom: 10px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                color: #5a5a7a; letter-spacing: 1.5px;">ESTIMATED_FARE</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem;
                                font-weight: 800; color: #e0e0ff; margin-top: 6px;">
                        ₹{estimated_fare}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                                color: #5a5a7a; margin-top: 4px;">
                        BASE: ₹{base_fare} × {surge}× SURGE
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Surge Log Events
            st.markdown(f"""
                <div style="margin-top: 8px;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                color: #5a5a7a; letter-spacing: 2px; margin-bottom: 10px;">
                        SURGE_LOG_EVENTS
                    </div>
            """, unsafe_allow_html=True)

            if reasons:
                for tag, msg, color in reasons:
                    st.markdown(f"""
                        <div style="display: flex; align-items: flex-start; gap: 8px;
                                    margin-bottom: 10px; padding-left: 4px;">
                            <span style="display: inline-block; width: 6px; height: 6px;
                                         background: {color}; border-radius: 50%;
                                         box-shadow: 0 0 4px {color}; margin-top: 5px; flex-shrink: 0;"></span>
                            <div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                                            font-weight: 700; color: #e0e0ff; letter-spacing: 0.5px;">
                                    {tag} [{zone_code}]
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                            color: #5a5a7a; font-style: italic;">
                                    {msg}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding-left: 4px;">
                        <span style="display: inline-block; width: 6px; height: 6px;
                                     background: #06C167; border-radius: 50%;
                                     box-shadow: 0 0 4px #06C167;"></span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                                    font-weight: 700; color: #06C167;">
                            ALL_CLEAR — nominal conditions
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Default state — Command Center idle
        st.markdown(f"""
            <div style="text-align: center; padding: 60px 40px;">
                <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.6;">⚡</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem;
                            font-weight: 800; color: #e0e0ff; letter-spacing: 2px;
                            margin-bottom: 8px;">
                    AWAITING_INPUT
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                            color: #3a3a5a; max-width: 500px; margin: 0 auto;
                            line-height: 1.8; letter-spacing: 0.5px;">
                    Configure market parameters in the sidebar panel.<br/>
                    Execute <span style="color: #276EF1; font-weight: 700;">PREDICT_SURGE</span>
                    to initiate real-time pricing computation.
                </div>
                <div style="width: 100%; max-width: 300px; height: 1px;
                            background: linear-gradient(90deg, transparent, #276EF1, transparent);
                            margin: 28px auto;"></div>
                <div style="display: flex; justify-content: center; gap: 50px; margin-top: 24px;">
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                                    font-weight: 800; color: #276EF1;">8</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                    color: #3a3a5a; letter-spacing: 1.5px; margin-top: 2px;">MARKETS</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                                    font-weight: 800; color: #276EF1;">26</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                    color: #3a3a5a; letter-spacing: 1.5px; margin-top: 2px;">FEATURES</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                                    font-weight: 800; color: #276EF1;">0.88</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                    color: #3a3a5a; letter-spacing: 1.5px; margin-top: 2px;">R²_SCORE</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                                    font-weight: 800; color: #06C167;">
                            <span class="live-dot"></span>LIVE
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                    color: #3a3a5a; letter-spacing: 1.5px; margin-top: 2px;">STATUS</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_scenarios:

    st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                        font-weight: 800; color: #e0e0ff; letter-spacing: 1px; margin-bottom: 4px;">
                SCENARIO_TESTING_LAB
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        color: #3a3a5a; letter-spacing: 1px;">
                Pre-computed scenarios validating model response to Indian market dynamics
            </div>
        </div>
    """, unsafe_allow_html=True)

    scenario_path = Path(MODEL_DIR) / "scenario_results.csv"
    if scenario_path.exists():
        scenarios_df = pd.read_csv(scenario_path)
        groups = scenarios_df["group"].unique()

        for group in groups:
            group_df = scenarios_df[scenarios_df["group"] == group]

            group_icon = {
                "Rain Effect": "☔",
                "Festival Effect": "🎉",
                "Time of Day": "🕐",
                "Customer Segment": "👤",
                "Compound Scenarios": "⚡",
            }.get(group, "📊")

            group_code = group.upper().replace(" ", "_")

            st.markdown(f"""
                <div style="background: #0d0d17; border: 1px solid #1e1e35;
                            border-radius: 3px; padding: 20px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                        <span style="font-size: 1rem;">{group_icon}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
                                     font-weight: 700; color: #e0e0ff; letter-spacing: 1px;">
                            {group_code}
                        </span>
                    </div>
            """, unsafe_allow_html=True)

            cols = st.columns(len(group_df))
            for i, (_, row) in enumerate(group_df.iterrows()):
                surge_val = row["predicted_surge"]
                color = get_surge_color(surge_val)
                status = get_surge_status(surge_val)
                with cols[i]:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 14px 8px;
                                    background: #0a0a14; border-radius: 3px;
                                    border: 1px solid #1e1e35;
                                    transition: all 0.3s ease;">
                            <div style="font-family: 'JetBrains Mono', monospace;
                                        font-size: 1.8rem; font-weight: 800;
                                        color: {color}; text-shadow: 0 0 20px {color}40;">
                                {surge_val}×
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace;
                                        font-size: 0.55rem; color: #5a5a7a;
                                        margin-top: 6px; letter-spacing: 0.5px;">
                                {row['scenario']}
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace;
                                        font-size: 0.5rem; color: {color};
                                        margin-top: 6px; font-weight: 700;
                                        letter-spacing: 1.5px;">
                                {status}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Key insights
        st.markdown("""
            <div style="background: linear-gradient(135deg, #0a0f1e 0%, #0d0d17 100%);
                        border: 1px solid #276EF1; border-radius: 3px;
                        padding: 20px; margin-top: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                            font-weight: 700; color: #276EF1; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 14px;">
                    💡 KEY_INSIGHTS
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; color: #8888aa;
                            font-size: 0.7rem; line-height: 2;">
                    <span style="color: #276EF1;">▸</span> <strong style="color: #e0e0ff;">RAIN_EFFECT</strong>: baseline → 1.63× (Heavy Rain alone)<br>
                    <span style="color: #E11900;">▸</span> <strong style="color: #e0e0ff;">DIWALI_EVENING</strong>: highest single-factor at 2.93×<br>
                    <span style="color: #F6A623;">▸</span> <strong style="color: #e0e0ff;">LATE_NIGHT</strong> (0100h): 2.54× due to supply shortage<br>
                    <span style="color: #E11900;">▸</span> <strong style="color: #e0e0ff;">COMPOUND</strong>: Diwali + Rain + Evening → 3.38× (near max)<br>
                    <span style="color: #5a5a7a;">▸</span> <strong style="color: #e0e0ff;">SEGMENT</strong>: alone doesn't change surge — affects cancellation rate
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Scenario results not found. Run `python -m src.models.analysis` first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:

    st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                        font-weight: 800; color: #e0e0ff; letter-spacing: 1px; margin-bottom: 4px;">
                MODEL_PERFORMANCE & ARCHITECTURE
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        color: #3a3a5a; letter-spacing: 1px;">
                Random Forest selected by 5-fold cross-validation // Ridge + Gradient Boosting benchmarks
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison ──
    st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                    font-weight: 600; color: #5a5a7a; text-transform: uppercase;
                    letter-spacing: 2px; margin-bottom: 12px;">
            MODEL_COMPARISON
        </div>
    """, unsafe_allow_html=True)

    cmp_path = Path(MODEL_DIR) / "model_comparison.csv"
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path)

        model_cols = st.columns(3)
        for i, (_, row) in enumerate(cmp_df.iterrows()):
            is_selected = "Random Forest" in row["Model"]
            border_color = "#276EF1" if is_selected else "#1e1e35"
            glow = "box-shadow: 0 0 20px rgba(39, 110, 241, 0.15), inset 0 0 20px rgba(39, 110, 241, 0.03);" if is_selected else ""
            badge = '<span style="font-family: \'JetBrains Mono\', monospace; background: #276EF1; color: white; font-size: 0.5rem; padding: 2px 8px; border-radius: 2px; font-weight: 700; letter-spacing: 1.5px;">SELECTED</span>' if is_selected else ""

            with model_cols[i]:
                st.markdown(f"""
                    <div style="background: #0d0d17; border: {'2px' if is_selected else '1px'} solid {border_color};
                                border-radius: 3px; padding: 20px; height: 290px; {glow}">
                        <div style="display: flex; justify-content: space-between; align-items: center;
                                    margin-bottom: 4px;">
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                                         font-weight: 700; color: #e0e0ff; letter-spacing: 0.5px;">
                                {row['Model']}
                            </span>
                            {badge}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                    color: #3a3a5a; margin-bottom: 18px; letter-spacing: 0.5px;">
                            {row['Type']}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                            <div style="background: #0a0a14; border-radius: 3px; padding: 10px;
                                        text-align: center; border: 1px solid #1a1a2e;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                                            font-weight: 800; color: #e0e0ff;">{row['RMSE']:.4f}</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                                            color: #5a5a7a; letter-spacing: 1.5px;">RMSE</div>
                            </div>
                            <div style="background: #0a0a14; border-radius: 3px; padding: 10px;
                                        text-align: center; border: 1px solid #1a1a2e;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                                            font-weight: 800; color: #e0e0ff;">{row['R2']:.4f}</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                                            color: #5a5a7a; letter-spacing: 1.5px;">R²</div>
                            </div>
                            <div style="background: #0a0a14; border-radius: 3px; padding: 10px;
                                        text-align: center; border: 1px solid #1a1a2e;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                                            font-weight: 800; color: #e0e0ff;">{row['CV_RMSE']:.4f}</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                                            color: #5a5a7a; letter-spacing: 1.5px;">CV-RMSE</div>
                            </div>
                            <div style="background: #0a0a14; border-radius: 3px; padding: 10px;
                                        text-align: center; border: 1px solid #1a1a2e;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                                            font-weight: 800; color: #e0e0ff;">{row['MAE']:.4f}</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                                            color: #5a5a7a; letter-spacing: 1.5px;">MAE</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance ──
    fi_path = Path(MODEL_DIR) / "feature_importance_detailed.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)

        st.markdown("""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        font-weight: 600; color: #5a5a7a; text-transform: uppercase;
                        letter-spacing: 2px; margin-bottom: 12px; margin-top: 8px;">
                TOP_10_FEATURE_IMPORTANCE
            </div>
        """, unsafe_allow_html=True)

        top10 = fi_df.head(10)

        for _, row in top10.iterrows():
            importance_pct = row["importance"] * 100
            bar_width = min(importance_pct / 0.6 * 100, 100)

            if importance_pct > 20:
                bar_color = "#276EF1"
            elif importance_pct > 5:
                bar_color = "#06C167"
            else:
                bar_color = "#3a3a5a"

            st.markdown(f"""
                <div style="background: #0d0d17; border-radius: 3px; padding: 12px 16px;
                            margin-bottom: 4px; border: 1px solid #1a1a2e;">
                    <div style="display: flex; justify-content: space-between;
                                align-items: center; margin-bottom: 6px;">
                        <div>
                            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700;
                                         color: #e0e0ff; font-size: 0.75rem;">
                                {row['feature']}
                            </span>
                            <span style="font-family: 'JetBrains Mono', monospace; color: #2a2a45;
                                         font-size: 0.6rem; margin-left: 8px;">
                                #{'%02d' % int(row['rank'])}
                            </span>
                        </div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800;
                                     color: {bar_color}; font-size: 0.75rem;">
                            {importance_pct:.1f}%
                        </span>
                    </div>
                    <div style="background: #0a0a14; border-radius: 2px; height: 4px; width: 100%;">
                        <div style="background: {bar_color}; border-radius: 2px;
                                    height: 4px; width: {bar_width}%;
                                    box-shadow: 0 0 8px {bar_color}40;
                                    transition: width 0.5s ease;"></div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                                color: #3a3a5a; margin-top: 6px; letter-spacing: 0.3px;">
                        {row['business_meaning']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ── Error Analysis ──
    report_path = Path(MODEL_DIR) / "analysis_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        err = report.get("error_analysis", {})
        if err:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                            font-weight: 600; color: #5a5a7a; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 12px;">
                    ERROR_ANALYSIS
                </div>
            """, unsafe_allow_html=True)

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("PRED_BIAS", f"{err.get('bias', 0):.4f}")
            with e2:
                st.metric("WITHIN_±0.2", f"{err.get('within_0_2', 0)}%")
            with e3:
                st.metric("WITHIN_±0.5", f"{err.get('within_0_5', 0)}%")
            with e4:
                st.metric("LARGE_ERRORS", f"{err.get('large_errors_pct', 0)}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:

    st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                        font-weight: 800; color: #e0e0ff; letter-spacing: 1px; margin-bottom: 4px;">
                DATA_EXPLORER
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        color: #3a3a5a; letter-spacing: 1px;">
                Browse the raw ride dataset used to train the surge prediction model
            </div>
        </div>
    """, unsafe_allow_html=True)

    raw_path = Path("data/raw/uber_rides_india.csv")
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("TOTAL_RIDES", f"{len(raw_df):,}")
        with s2:
            st.metric("MARKETS", f"{raw_df['city'].nunique()}")
        with s3:
            st.metric("AVG_SURGE", f"{raw_df['surge_multiplier'].mean():.2f}×")
        with s4:
            st.metric("MAX_SURGE", f"{raw_df['surge_multiplier'].max():.2f}×")

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_cities = st.multiselect("FILTER_MARKET", CITIES, default=CITIES)
        with filter_col2:
            selected_weather = st.multiselect("FILTER_WEATHER", WEATHER_OPTIONS,
                                               default=WEATHER_OPTIONS)
        with filter_col3:
            selected_events = st.multiselect("FILTER_EVENT", EVENTS, default=EVENTS)

        filtered = raw_df[
            (raw_df["city"].isin(selected_cities)) &
            (raw_df["weather"].isin(selected_weather)) &
            (raw_df["special_event"].isin(selected_events))
        ]

        st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        color: #3a3a5a; margin-bottom: 8px; letter-spacing: 0.5px;">
                SHOWING {len(filtered):,} OF {len(raw_df):,} RECORDS
            </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            filtered.head(200),
            width='stretch',
            height=400,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Surge distribution by city ──
        st.markdown("""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        font-weight: 600; color: #5a5a7a; text-transform: uppercase;
                        letter-spacing: 2px; margin-bottom: 12px;">
                AVG_SURGE_BY_MARKET
            </div>
        """, unsafe_allow_html=True)

        city_surge = (filtered.groupby("city")["surge_multiplier"]
                      .agg(["mean", "max", "min", "count"])
                      .round(2)
                      .sort_values("mean", ascending=False)
                      .reset_index())
        city_surge.columns = ["City", "Avg Surge", "Max Surge", "Min Surge", "Rides"]

        for _, row in city_surge.iterrows():
            color = get_surge_color(row["Avg Surge"])
            bar_w = min((row["Avg Surge"] / 3.5) * 100, 100)
            zone = CITY_ZONES.get(row["City"], "ZONE_UNK")
            st.markdown(f"""
                <div style="background: #0d0d17; border-radius: 3px; padding: 12px 16px;
                            margin-bottom: 4px; border: 1px solid #1a1a2e;
                            display: flex; align-items: center; justify-content: space-between;">
                    <div style="width: 140px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700;
                                     color: #e0e0ff; font-size: 0.75rem;">{row['City']}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                                     color: #2a2a45;">{zone}</span>
                    </div>
                    <div style="flex: 1; margin: 0 16px;">
                        <div style="background: #0a0a14; border-radius: 2px; height: 6px;">
                            <div style="background: {color}; border-radius: 2px;
                                        height: 6px; width: {bar_w}%;
                                        box-shadow: 0 0 8px {color}40;"></div>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 170px;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800;
                                     color: {color}; font-size: 0.8rem;">
                            {row['Avg Surge']}×
                        </span>
                        <span style="font-family: 'JetBrains Mono', monospace; color: #2a2a45;
                                     font-size: 0.6rem; margin-left: 10px;">
                            ({int(row['Rides'])} rides)
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ── Multi-city surge map ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
                        font-weight: 600; color: #5a5a7a; text-transform: uppercase;
                        letter-spacing: 2px; margin-bottom: 12px;">
                SURGE_INTENSITY_MAP // ALL_MARKETS
            </div>
        """, unsafe_allow_html=True)

        map_points = []
        for _, row in city_surge.iterrows():
            c = CITY_COORDS.get(row["City"])
            if c:
                map_points.append({
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "surge": row["Avg Surge"],
                    "city": row["City"],
                    "rides": int(row["Rides"]),
                })

        map_df = pd.DataFrame(map_points)

        if len(map_df) > 0:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(
                    latitude=22.0,
                    longitude=78.0,
                    zoom=4.2,
                    pitch=30,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lon", "lat"],
                        get_radius="surge * 30000",
                        get_fill_color=[
                            "surge > 2.0 ? 240 : surge > 1.5 ? 246 : 6",
                            "surge > 2.0 ? 59 : surge > 1.5 ? 166 : 193",
                            "surge > 2.0 ? 32 : surge > 1.5 ? 35 : 103",
                            180,
                        ],
                        pickable=True,
                    ),
                ],
                tooltip={
                    "html": "<div style='font-family: JetBrains Mono, monospace;'>"
                            "<b>{city}</b><br>Surge: {surge}×<br>Rides: {rides}</div>",
                    "style": {
                        "backgroundColor": "#0d0d17",
                        "color": "#e0e0ff",
                        "border": "1px solid #1e1e35",
                        "fontSize": "12px",
                    },
                },
            ), use_container_width=True)

    else:
        st.warning("Raw data file not found at `data/raw/uber_rides_india.csv`.")


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div style="text-align: center; padding: 30px 0 16px; border-top: 1px solid #1a1a2e;
                margin-top: 40px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
                    color: #2a2a45; letter-spacing: 2px;">
            SURGE_COMMAND V3.0 &nbsp;·&nbsp; ML-POWERED DYNAMIC PRICING &nbsp;·&nbsp; INDIA_OPS
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
                    color: #1e1e35; margin-top: 4px; letter-spacing: 1px;">
            RANDOM_FOREST (5-FOLD CV) &nbsp;·&nbsp; 26 FEATURES &nbsp;·&nbsp; 1,500 RIDES &nbsp;·&nbsp; R² = 0.882
        </div>
    </div>
""", unsafe_allow_html=True)
