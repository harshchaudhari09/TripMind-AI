"""
tools/overpass_tool.py — nearby facilities (petrol pumps, ATMs, hospitals, bus and
railway stations) from OpenStreetMap.

History: this used the public Overpass API. LangSmith traces showed every lookup in
every trip failing — "406 Not Acceptable" or a 30-second read timeout — so each trip
waited ~32 s for nothing and the "Useful nearby" section had no real data. The public
Overpass servers also block cloud hosts (where the live demo runs).

Now it uses Nominatim (OpenStreetMap's search service) instead:
  • one search per category, bounded to a ~13 km box around the destination
  • Nominatim's usage policy is respected: an identifying User-Agent, at most one
    request per second (shared across the whole app), and results are cached
  • results are cached per destination for 30 days (in memory + on disk), so popular
    destinations are instant and repeat trips cost no requests at all
  • short timeouts: if the service is unavailable the trip continues within seconds,
    with an explicit "unavailable" note so the writer doesn't invent names
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import requests
from langchain.tools import tool

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "TripMindAI/1.0 (+https://github.com/harshchaudhari09/TripMind-AI)"
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "nearby_places.json")
CACHE_DAYS = 30
SEARCH_BOX_DEG = 0.12          # ≈ 13 km each way around the destination

# What to search for, and which OpenStreetMap tags count as a match.
CATEGORIES = {
    "petrol_pumps":          {"q": "fuel",            "keep": {("amenity", "fuel")},                                   "limit": 5},
    "atms":                  {"q": "atm",             "keep": {("amenity", "atm"), ("amenity", "bank")},              "limit": 6},
    "hospitals_and_clinics": {"q": "hospital",        "keep": {("amenity", "hospital"), ("amenity", "clinic"),
                                                               ("healthcare", "hospital"), ("building", "hospital")},  "limit": 5},
    "bus_stations":          {"q": "bus station",     "keep": {("amenity", "bus_station"), ("highway", "bus_stop")},   "limit": 5},
    "railway_stations":      {"q": "railway station", "keep": {("railway", "station"), ("railway", "halt"),
                                                               ("building", "train_station")},                         "limit": 5},
}

_request_lock = threading.Lock()   # one Nominatim request at a time, ≥1 s apart
_last_request = [0.0]
_geocode_cache: dict = {}
_places_cache: dict = {}
_places_locks: dict = {}
_places_locks_guard = threading.Lock()


def _nominatim(params: dict, timeout: float = 8) -> list:
    """GET Nominatim with the policy-required User-Agent and ≥1.1 s between requests."""
    with _request_lock:
        wait = 1.1 - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(NOMINATIM_URL, params={**params, "format": "jsonv2"},
                                headers={"User-Agent": USER_AGENT}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        finally:
            _last_request[0] = time.time()


def _geocode(place_name: str) -> tuple:
    """Place name → (lat, lon). Successful lookups are cached; failures are retried later."""
    key = place_name.strip().lower()
    if key in _geocode_cache:
        return _geocode_cache[key]
    try:
        data = _nominatim({"q": place_name, "limit": 1})
    except Exception:
        return None, None
    if not data:
        return None, None
    coords = float(data[0]["lat"]), float(data[0]["lon"])
    _geocode_cache[key] = coords
    return coords


def _km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km (good enough at city scale)."""
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110.57
    return round(math.hypot(dx, dy), 1)


def _describe(item: dict, lat: float, lon: float) -> dict:
    addr = item.get("address") or {}
    extra = item.get("extratags") or {}
    name = (item.get("name") or "").strip()
    area = addr.get("road") or addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district") or ""
    generic = not name or name.lower() in {"atm", "hospital", "railway station", "bus stand", "bus station", "petrol pump"}
    # The place's own name first ("Jaipur Junction"); brand/operator only for generic ones
    # ("ATM" → "SBI"), otherwise every station would be called "NWR" (the railway operator).
    label = (name if not generic else None) or extra.get("brand") or extra.get("operator") or name \
        or item.get("type", "").replace("_", " ").title()
    if generic and area:
        label = f"{label} — {area}"          # "ATM — MI Road" instead of just "ATM"
    return {
        "name":          label,
        "address":       ", ".join(p for p in [area, addr.get("city") or addr.get("town") or addr.get("village")] if p),
        "phone":         extra.get("phone") or extra.get("contact:phone") or "",
        "opening_hours": extra.get("opening_hours") or "",
        "distance_km":   _km(lat, lon, float(item["lat"]), float(item["lon"])),
    }


def _is_excluded(item: dict) -> bool:
    """Metro stations and railway offices are tagged like railway stations — skip them."""
    extra = item.get("extratags") or {}
    name = (item.get("name") or "").lower()
    return (extra.get("station") in ("subway", "light_rail", "monorail") or "metro" in name
            or extra.get("office") is not None or name in {"nwr", "ncr", "wr", "cr", "nr"})


def _search_category(spec: dict, lat: float, lon: float) -> list:
    box = SEARCH_BOX_DEG
    items = _nominatim({"q": spec["q"], "limit": 25, "bounded": 1, "addressdetails": 1, "extratags": 1,
                        "viewbox": f"{lon - box},{lat + box},{lon + box},{lat - box}"})
    matches = [i for i in items if (i.get("category"), i.get("type")) in spec["keep"]
               and not _is_excluded(i)]
    places = sorted((_describe(i, lat, lon) for i in matches), key=lambda p: p["distance_km"])
    seen, unique = set(), []
    for p in places:                         # the same place can appear as node + building
        if p["name"].lower() not in seen:
            seen.add(p["name"].lower())
            unique.append(p)
    return unique[: spec["limit"]]


def _load_disk_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_disk_cache(key: str, value: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        data = _load_disk_cache()
        data[key] = value
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass                                  # caching is a speed-up, never a requirement


def nearby_places(destination: str) -> dict:
    """
    All five facility lists for a destination, fetched once and shared: the graph's two
    facility nodes run in parallel, so the second waits for the first instead of
    repeating the same searches.
    """
    key = destination.strip().lower()
    with _places_locks_guard:
        lock = _places_locks.setdefault(key, threading.Lock())
    with lock:
        fresh = lambda v: v and time.time() - v.get("fetched_at", 0) < CACHE_DAYS * 86400
        if fresh(_places_cache.get(key)):
            return _places_cache[key]
        disk = _load_disk_cache().get(key)
        if fresh(disk):
            _places_cache[key] = disk
            return disk

        lat, lon = _geocode(destination)
        if not lat:
            return {"error": f"Could not find location: {destination}"}
        result, failed = {"coordinates": {"lat": lat, "lon": lon}}, []
        for category, spec in CATEGORIES.items():
            try:
                result[category] = _search_category(spec, lat, lon)
            except Exception:
                result[category] = []
                failed.append(category)
        result["fetched_at"] = time.time()
        if failed:
            result["unavailable"] = failed     # don't cache a partial failure
        else:
            _places_cache[key] = result
            _save_disk_cache(key, result)
        return result


def _tool_output(destination: str, category: str, extra: dict | None = None) -> str:
    data = nearby_places(destination)
    if "error" in data:
        return json.dumps({"error": data["error"]})
    places = data.get(category, [])
    note = "Data from OpenStreetMap. Distances are straight-line and approximate."
    if category in data.get("unavailable", []):
        note = "Live map data was unavailable — do not name specific places for this category."
    elif not places:
        note = "No places of this type are mapped nearby — do not invent names."
    return json.dumps({"destination": destination, category: places, **(extra or {}), "note": note}, indent=2)


@tool
def get_petrol_pumps(destination: str) -> str:
    """
    Find petrol pumps / fuel stations near a travel destination.
    Input: destination name e.g. "Manali, Himachal Pradesh"
    Returns JSON with nearby petrol pumps (names, areas, distances).
    """
    return _tool_output(destination, "petrol_pumps")


@tool
def get_atms(destination: str) -> str:
    """
    Find ATMs near a travel destination.
    Input: destination name e.g. "Jaisalmer, Rajasthan"
    Returns JSON with nearby ATMs and banks.
    """
    return _tool_output(destination, "atms")


@tool
def get_hospitals(destination: str) -> str:
    """
    Find hospitals and clinics near a travel destination.
    Input: destination name e.g. "Srinagar, Kashmir"
    Returns JSON with nearby hospitals and clinics.
    """
    return _tool_output(destination, "hospitals_and_clinics",
                        {"emergency_number": "112 (India universal emergency)"})


@tool
def get_bus_stations(destination: str) -> str:
    """
    Find bus stations near a travel destination.
    Input: destination name e.g. "Shimla, Himachal Pradesh"
    Returns JSON with nearby bus stations.
    """
    return _tool_output(destination, "bus_stations", {"booking_links": {
        "HRTC (Himachal)": "https://www.hrtchp.com", "JKSRTC (J&K)": "https://jksrtc.co.in",
        "RSRTC (Rajasthan)": "https://rsrtc.rajasthan.gov.in", "Redbus": "https://www.redbus.in"}})


@tool
def get_train_stations(destination: str) -> str:
    """
    Find railway stations near a travel destination.
    Input: destination name e.g. "Jaipur, Rajasthan"
    Returns JSON with nearby railway stations.
    """
    return _tool_output(destination, "railway_stations", {"booking_links": {
        "IRCTC": "https://www.irctc.co.in", "RailYatri": "https://www.railyatri.in",
        "ConfirmTkt": "https://www.confirmtkt.com"}})


# Export all map tools
OVERPASS_TOOLS = [
    get_petrol_pumps,
    get_atms,
    get_hospitals,
    get_bus_stations,
    get_train_stations,
]
