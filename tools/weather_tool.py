"""
tools/weather_tool.py

Uses Open-Meteo API — completely free, no API key needed, no rate limits.
https://open-meteo.com/
"""
import requests
import json
from datetime import datetime
from langchain.tools import tool


def _geocode(place_name: str) -> tuple:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "TripMindAI/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", place_name)
        return None, None, place_name
    except Exception:
        return None, None, place_name


def _interpret_weather_code(code: int) -> str:
    """Convert WMO weather code to human readable string."""
    codes = {
        0: "Clear sky ☀️",
        1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
        45: "Foggy 🌫️", 48: "Icy fog 🌫️",
        51: "Light drizzle 🌦️", 53: "Moderate drizzle 🌦️", 55: "Dense drizzle 🌧️",
        61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
        71: "Slight snow 🌨️", 73: "Moderate snow 🌨️", 75: "Heavy snow ❄️",
        77: "Snow grains ❄️",
        80: "Slight showers 🌦️", 81: "Moderate showers 🌧️", 82: "Violent showers ⛈️",
        85: "Slight snow showers 🌨️", 86: "Heavy snow showers ❄️",
        95: "Thunderstorm ⛈️", 96: "Thunderstorm with hail ⛈️", 99: "Thunderstorm with heavy hail ⛈️",
    }
    return codes.get(code, f"Weather code {code}")


@tool
def get_weather_forecast(query: str) -> str:
    """
    Get detailed weather forecast for a destination and date range.
    Input format: "DESTINATION|START_DATE|END_DATE"
    Example: "Manali, Himachal Pradesh|2024-12-20|2024-12-25"
    Returns temperature, rain, wind and packing advice.
    """
    try:
        parts = query.split("|")
        destination = parts[0].strip()
        start_date  = parts[1].strip() if len(parts) > 1 else datetime.now().strftime("%Y-%m-%d")
        end_date    = parts[2].strip() if len(parts) > 2 else start_date

        lat, lon, display_name = _geocode(destination)
        if not lat:
            return json.dumps({"error": f"Could not geocode: {destination}"})

        # Open-Meteo API — completely free
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max",
                "weathercode",
                "uv_index_max",
            ],
            "timezone":   "Asia/Kolkata",
            "start_date": start_date,
            "end_date":   end_date,
            "wind_speed_unit": "kmh",
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        # Build day by day forecast
        forecast_days = []
        dates   = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip    = daily.get("precipitation_sum", [])
        windspeed = daily.get("windspeed_10m_max", [])
        codes     = daily.get("weathercode", [])
        uv        = daily.get("uv_index_max", [])

        for i, date in enumerate(dates):
            forecast_days.append({
                "date":        date,
                "condition":   _interpret_weather_code(codes[i] if i < len(codes) else 0),
                "max_temp_c":  max_temps[i] if i < len(max_temps) else "N/A",
                "min_temp_c":  min_temps[i] if i < len(min_temps) else "N/A",
                "rainfall_mm": precip[i] if i < len(precip) else 0,
                "max_wind_kmh": windspeed[i] if i < len(windspeed) else "N/A",
                "uv_index":   uv[i] if i < len(uv) else "N/A",
            })

        # Smart packing advice based on forecast
        avg_max = sum(d["max_temp_c"] for d in forecast_days if isinstance(d["max_temp_c"], (int, float))) / max(len(forecast_days), 1)
        total_rain = sum(d["rainfall_mm"] for d in forecast_days if isinstance(d["rainfall_mm"], (int, float)))

        packing = []
        if avg_max < 5:
            packing = ["Heavy winter jacket ❄️", "Thermal innerwear", "Woolen gloves & cap", "Snow boots", "Hand warmers"]
        elif avg_max < 15:
            packing = ["Warm jacket 🧥", "Sweaters & fleece", "Light gloves", "Layered clothing"]
        elif avg_max < 25:
            packing = ["Light jacket for evenings", "T-shirts & jeans", "Comfortable walking shoes"]
        else:
            packing = ["Light cotton clothes 👕", "Sunscreen SPF50+ 🧴", "Sunglasses 🕶️", "Hat/cap"]

        if total_rain > 50:
            packing.extend(["Rain jacket / poncho 🌧️", "Waterproof bag covers", "Quick-dry clothes"])

        return json.dumps({
            "destination":   destination,
            "location":      display_name,
            "coordinates":   {"lat": lat, "lon": lon},
            "forecast":      forecast_days,
            "summary": {
                "avg_max_temp": round(avg_max, 1),
                "total_rainfall_mm": round(total_rain, 1),
                "overall_condition": "Rainy" if total_rain > 30 else ("Cold" if avg_max < 10 else ("Pleasant" if avg_max < 28 else "Hot")),
            },
            "packing_advice": packing,
            "source": "Open-Meteo (free, open data)",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
