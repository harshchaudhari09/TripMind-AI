"""Nearby facilities (Nominatim): filtering, naming, one shared fetch, and failing fast."""
import json

import pytest

import tools.overpass_tool as O


def place(name, category, type_, lat=26.92, lon=75.79, **extratags):
    return {"name": name, "category": category, "type": type_, "lat": str(lat), "lon": str(lon),
            "address": {"road": "MI Road", "city": "Jaipur"}, "extratags": extratags}


FAKE = {
    "geocode":         [{"lat": "26.9124", "lon": "75.7873"}],
    "fuel":            [place("HP", "amenity", "fuel"), place("Some Cafe", "amenity", "cafe")],
    "atm":             [place("ATM", "amenity", "atm", brand="SBI")],
    "hospital":        [place("SMS Hospital", "amenity", "hospital", phone="0141 2560291")],
    "bus station":     [place("Sindhi Camp Bus Stand", "amenity", "bus_station")],
    "railway station": [place("Jaipur Junction", "railway", "station", operator="NWR"),
                        place("Civil Lines Metro", "railway", "station", station="subway")],
}


@pytest.fixture
def fake_nominatim(monkeypatch, tmp_path):
    calls = []

    def fake(params, timeout=8):
        calls.append(params["q"])
        return FAKE["geocode"] if "viewbox" not in params else FAKE[params["q"]]
    monkeypatch.setattr(O, "_nominatim", fake)
    monkeypatch.setattr(O, "CACHE_FILE", str(tmp_path / "places.json"))
    O._places_cache.clear(); O._geocode_cache.clear()
    return calls


def test_only_matching_places_are_kept_and_named_well(fake_nominatim):
    data = O.nearby_places("Jaipur")
    assert [p["name"] for p in data["petrol_pumps"]] == ["HP"]               # the cafe is dropped
    assert data["atms"][0]["name"] == "SBI — MI Road"                        # generic "ATM" → brand + road
    assert data["hospitals_and_clinics"][0]["phone"] == "0141 2560291"
    assert [p["name"] for p in data["railway_stations"]] == ["Jaipur Junction"]  # metro dropped, not "NWR"


def test_all_five_tools_share_one_fetch(fake_nominatim):
    for tool in O.OVERPASS_TOOLS:
        tool.invoke("Jaipur")
    assert len(fake_nominatim) == 6                                          # 1 geocode + 5 categories, once


def test_results_are_reused_from_the_disk_cache(fake_nominatim):
    O.nearby_places("Jaipur")
    O._places_cache.clear()                                                  # e.g. after a server restart
    fake_nominatim.clear()
    O.nearby_places("Jaipur")
    assert fake_nominatim == []


def test_failure_is_reported_quickly_and_not_cached(monkeypatch, tmp_path):
    def broken(params, timeout=8):
        if "viewbox" not in params:
            return FAKE["geocode"]
        raise TimeoutError("map service down")
    monkeypatch.setattr(O, "_nominatim", broken)
    monkeypatch.setattr(O, "CACHE_FILE", str(tmp_path / "places.json"))
    O._places_cache.clear(); O._geocode_cache.clear()
    out = json.loads(O.get_atms.invoke("Jaipur"))
    assert out["atms"] == [] and "unavailable" in out["note"]
    assert "jaipur" not in O._places_cache


def test_writer_gets_a_compact_list_or_none_found():
    import agents.graph as G
    line = G._places_block("Hospitals", [{"name": "SMS Hospital", "distance_km": 1.1, "phone": "0141"}])
    assert line == "- Hospitals: SMS Hospital (1.1 km, ph 0141)"
    assert G._places_block("Railway stations", []) == "- Railway stations: none found"
