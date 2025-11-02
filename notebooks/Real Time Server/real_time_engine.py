import asyncio
import websockets
import json
import datetime
import numpy as np
import random

# Synthetic data generator for real-time streaming
def generate_live_surge_data():
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
    areas = ['Airport', 'Business District', 'Mall']
    weather_conditions = ['Sunny', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Thunderstorm']

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    city = random.choice(cities)
    area = random.choice(areas)
    hour = datetime.datetime.now().hour

    is_rush_hour = hour in [7,8,9,18,19,20]
    is_weekend = datetime.datetime.now().weekday() >= 5
    base_demand = 30 + (20 if is_rush_hour else 0) + (15 if is_weekend else 0)
    is_night = hour < 6 or hour > 22
    base_supply = 45 - (15 if is_night else 0)

    weather = random.choices(weather_conditions, weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
    weather_demand_boost = {'Sunny': 0, 'Cloudy': 3, 'Light Rain':15, 'Heavy Rain':35, 'Thunderstorm':60}
    weather_supply_reduction = {'Sunny': 0, 'Cloudy': 0, 'Light Rain':8, 'Heavy Rain':20, 'Thunderstorm':35}

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

    driver_utilization = min(100, max(0, supply * 1.5 + np.random.uniform(-8,8)))

    data = {
        'timestamp': timestamp,
        'city': city,
        'area': area,
        'hour': hour,
        'is_weekend': is_weekend,
        'weather': weather,
        'demand': round(demand, 1),
        'supply': round(supply, 1),
        'surge_multiplier': round(surge_multiplier, 2),
        'base_fare': round(base_fare, 2),
        'completed_rides': round(completed_rides, 1),
        'revenue': round(revenue, 2),
        'driver_utilization': round(driver_utilization, 1)
    }
    return data

async def send_live_data(websocket, path):
    print(f"Client connected: {path}")
    try:
        while True:
            data = generate_live_surge_data()
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1)  # stream every second
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(send_live_data, "localhost", 6789):
        print("Real-time WebSocket server started at ws://localhost:6789")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
