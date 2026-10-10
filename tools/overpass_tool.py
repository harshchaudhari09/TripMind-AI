"""
tools/overpass_tool.py

Uses OpenStreetMap Overpass API (completely free, no key needed)
to find petrol pumps, ATMs, hospitals, bus stations and train stations
near any destination.
"""
import requests
import json
import threading
from langchain.tools import tool

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Geocoding cache. The graph runs research nodes in parallel, so several tools
# ask for the same city's coordinates at the same moment. Nominatim allows
# ~1 request/second, so we look each place up once and share the answer.
_geocode_cache: dict = {}
_geocode_lock = threading.Lock()


def _query_overpass(lat: float, lon: float, amenity: str, radius_m: int = 5000, limit: int = 5) -> list:
    """
    Core function to query Overpass API for any amenity near lat/lon.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["{amenity}"](around:{radius_m},{lat},{lon});
      way["{amenity}"](around:{radius_m},{lat},{lon});
    );
    out center {limit};
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            # get coordinates
            if el["type"] == "node":
                elat, elon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                elat, elon = center.get("lat"), center.get("lon")

            # calculate rough distance in km
            if elat and elon:
                dist_km = round(((elat - lat)**2 + (elon - lon)**2)**0.5 * 111, 2)
            else:
                dist_km = None

            results.append({
                "name":    tags.get("name") or tags.get("operator") or tags.get("brand") or "Unnamed",
                "address": tags.get("addr:full") or tags.get("addr:street") or tags.get("addr:city") or "",
                "phone":   tags.get("phone") or tags.get("contact:phone") or "",
                "opening_hours": tags.get("opening_hours") or "Not available",
                "distance_km": dist_km,
            })
        # sort by distance
        results.sort(key=lambda x: x["distance_km"] or 999)
        return results[:limit]
    except Exception as e:
        return [{"error": str(e)}]


def _geocode(place_name: str) -> tuple:
    """
    Convert place name to lat/lon using Nominatim (free OpenStreetMap geocoder).
    Successful lookups are cached; failures are not, so a later call can retry.
    """
    key = place_name.strip().lower()
    # The lock makes parallel callers wait for the first lookup instead of
    # all hitting Nominatim at once.
    with _geocode_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": place_name, "format": "json", "limit": 1}
            headers = {"User-Agent": "TripMindAI/1.0 travel-planner-student-project"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                coords = float(data[0]["lat"]), float(data[0]["lon"])
                _geocode_cache[key] = coords
                return coords
            return None, None
        except Exception:
            return None, None


@tool
def get_petrol_pumps(destination: str) -> str:
    """
    Find petrol pumps / fuel stations near a travel destination.
    Input: destination name e.g. "Manali, Himachal Pradesh"
    Returns JSON with list of nearby petrol pumps with names, addresses, distances.
    """
    lat, lon = _geocode(destination)
    if not lat:
        return json.dumps({"error": f"Could not find location: {destination}"})

    results = _query_overpass(lat, lon, amenity="fuel", radius_m=10000, limit=5)
    return json.dumps({
        "destination": destination,
        "coordinates": {"lat": lat, "lon": lon},
        "petrol_pumps": results,
        "note": "Data from OpenStreetMap. Distances are approximate.",
    }, indent=2)


@tool
def get_atms(destination: str) -> str:
    """
    Find ATMs near a travel destination.
    Input: destination name e.g. "Jaisalmer, Rajasthan"
    Returns JSON with list of nearby ATMs.
    """
    lat, lon = _geocode(destination)
    if not lat:
        return json.dumps({"error": f"Could not find location: {destination}"})

    results = _query_overpass(lat, lon, amenity="atm", radius_m=5000, limit=8)
    return json.dumps({
        "destination": destination,
        "atms": results,
        "note": "Data from OpenStreetMap.",
    }, indent=2)


@tool
def get_hospitals(destination: str) -> str:
    """
    Find hospitals and medical facilities near a travel destination.
    Input: destination name e.g. "Srinagar, Kashmir"
    Returns JSON with list of nearby hospitals.
    """
    lat, lon = _geocode(destination)
    if not lat:
        return json.dumps({"error": f"Could not find location: {destination}"})

    results = _query_overpass(lat, lon, amenity="hospital", radius_m=15000, limit=5)
    # also check for clinics if hospitals list is small
    if len(results) < 3:
        clinics = _query_overpass(lat, lon, amenity="clinic", radius_m=10000, limit=3)
        results.extend(clinics)

    return json.dumps({
        "destination": destination,
        "hospitals_and_clinics": results,
        "emergency_number": "112 (India universal emergency)",
        "note": "Data from OpenStreetMap.",
    }, indent=2)


@tool
def get_bus_stations(destination: str) -> str:
    """
    Find bus stations and bus stops near a travel destination.
    Input: destination name e.g. "Shimla, Himachal Pradesh"
    Returns JSON with list of nearby bus stations.
    """
    lat, lon = _geocode(destination)
    if not lat:
        return json.dumps({"error": f"Could not find location: {destination}"})

    # query both bus_station and bus_stop
    stations = _query_overpass(lat, lon, amenity="bus_station", radius_m=5000, limit=5)
    if len(stations) < 3:
        stops = _query_overpass(lat, lon, amenity="bus_stop", radius_m=3000, limit=5)
        stations.extend(stops)

    return json.dumps({
        "destination": destination,
        "bus_stations": stations[:6],
        "booking_links": {
            "HRTC (Himachal)": "https://www.hrtchp.com",
            "JKSRTC (J&K)":    "https://jksrtc.co.in",
            "RSRTC (Rajasthan)": "https://rsrtc.rajasthan.gov.in",
            "Redbus":           "https://www.redbus.in",
        },
        "note": "Data from OpenStreetMap.",
    }, indent=2)


@tool
def get_train_stations(destination: str) -> str:
    """
    Find railway stations near a travel destination.
    Input: destination name e.g. "Jaipur, Rajasthan"
    Returns JSON with list of nearby railway stations.
    """
    lat, lon = _geocode(destination)
    if not lat:
        return json.dumps({"error": f"Could not find location: {destination}"})

    # use railway=station tag in Overpass
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"](around:20000,{lat},{lon});
      node["railway"="halt"](around:20000,{lat},{lon});
    );
    out 6;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        data = resp.json()
        stations = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            dist_km = round(((el.get("lat", lat) - lat)**2 + (el.get("lon", lon) - lon)**2)**0.5 * 111, 2)
            stations.append({
                "name":         tags.get("name") or "Unknown Station",
                "station_code": tags.get("ref") or tags.get("uic_ref") or "N/A",
                "distance_km":  dist_km,
                "address":      tags.get("addr:city") or "",
            })
        stations.sort(key=lambda x: x["distance_km"])

        return json.dumps({
            "destination": destination,
            "railway_stations": stations[:5],
            "booking_links": {
                "IRCTC": "https://www.irctc.co.in",
                "RailYatri": "https://www.railyatri.in",
                "ConfirmTkt": "https://www.confirmtkt.com",
            },
            "note": "Data from OpenStreetMap.",
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Export all Overpass tools
OVERPASS_TOOLS = [
    get_petrol_pumps,
    get_atms,
    get_hospitals,
    get_bus_stations,
    get_train_stations,
]
