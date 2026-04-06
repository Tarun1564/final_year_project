import requests
import joblib
import numpy as np
import os
from django.utils.translation import gettext as _

# -------------------------------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models ONCE
crop_model = joblib.load(os.path.join(BASE_DIR, "models\\crop_index_regressor.pkl"))
irrigation_model = joblib.load(os.path.join(BASE_DIR, "models\\irrigation_index_regressor.pkl"))
acl_model = joblib.load(os.path.join(BASE_DIR, "models\\acli_regressor.pkl"))
yield_model = joblib.load(os.path.join(BASE_DIR, "models\\gradient_boosting_classifier.pkl"))

print("Models loaded successfully.")

# -------------------------------------------------
# Fetch weather data from API
def fetch_weather_data(lat, lon):
    API_KEY = "5d14f273be1a4f3f9b3145407260302"
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={lat},{lon}&days=30"
    response = requests.get(url)
    data = response.json()

    weather = []
    for day in data["forecast"]["forecastday"]:
        weather.append([
            day["day"]["avgtemp_c"],      # 0: Average temperature
            day["day"]["avghumidity"],    # 1: Average humidity
            day["day"]["totalprecip_mm"], # 2: Total precipitation
            day["day"]["maxwind_kph"]     # 3: Max wind speed
        ])
    return weather

# -------------------------------------------------
# Generate daily indices using regression models
def generate_indices(weather_data):
    daily_indices = []

    for row in weather_data:
        X = np.array(row).reshape(1, -1)  # shape (1, 4) to match training

        # Predict each index
        water = crop_model.predict(X)[0]
        irrigation = irrigation_model.predict(X)[0]
        acl = acl_model.predict(X)[0]

        daily_indices.append({
            "water": round(water, 3),
            "irrigation": round(irrigation, 3),
            "acl": round(acl, 3)
        })

    return daily_indices

# -------------------------------------------------
# Aggregate daily indices and include extra feature for classifier
def aggregate_features(daily_indices, weather_data):
    water = np.mean([d["water"] for d in daily_indices])
    irrigation = np.mean([d["irrigation"] for d in daily_indices])
    acl = np.mean([d["acl"] for d in daily_indices])

    # Include the missing 4th feature for yield_model
    # Here, we use the average max wind speed across the days
    avg_max_wind = np.mean([day[3] for day in weather_data])

    # Return all 4 features for classifier
    return [water, irrigation, acl, avg_max_wind]

# -------------------------------------------------
# Predict yield using the classifier model
def predict_yield(features):
    features = np.array(features).reshape(1, -1)  # ensure shape (1,4)
    prediction = yield_model.predict(features)[0]
    return prediction

# -------------------------------------------------
# Farmer-friendly messages
def map_water_index(val):
    if val < 0.4: return _("💧 Irrigate today")
    elif val < 0.7: return _("⚠️ Monitor water")
    else: return _("✅ Water sufficient")

def map_irrigation_index(val):
    if val < 0.4: return _("💧 Irrigation needed")
    elif val < 0.7: return _("⚠️ Moderate irrigation")
    else: return _("✅ Irrigation optimal")

def map_acl_index(val):
    if val < 0.4: return _("❌ Crop stress high")
    elif val < 0.7: return _("⚠️ Moderate crop stress")
    else: return _("✅ Crop healthy")

# -------------------------------------------------
# Example usage
if __name__ == "__main__":
    lat, lon = 17.3850, 78.4867  # Hyderabad example
    weather_data = fetch_weather_data(lat, lon)
    daily_indices = generate_indices(weather_data)
    features = aggregate_features(daily_indices, weather_data)
    predicted_yield = predict_yield(features)

    # Add farmer-friendly messages
    for d in daily_indices:
        d['water_msg'] = map_water_index(d['water'])
        d['irrigation_msg'] = map_irrigation_index(d['irrigation'])
        d['acl_msg'] = map_acl_index(d['acl'])

    print("Daily Indices:", daily_indices[:3], "...")  # first 3 days
    print("Aggregated Features:", features)
    print("Predicted Yield:", predicted_yield)
