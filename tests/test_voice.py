"""Voice assistant: what the small model extracts is checked by code before it's used."""
from datetime import date

from agents.voice_intent import clean_intent, parse_voice_command

TODAY = date(2026, 10, 14)


def test_new_trip_fields_pass_through_when_valid():
    out = clean_intent({"action": "new_trip", "destination": "Goa", "origin": "Mumbai",
                        "start_date": "2026-10-17", "end_date": "2026-10-19", "num_days": 3,
                        "budget_inr": 30000, "num_people": 2, "preferences": ["beaches"],
                        "message": "Planning a 3-day trip to Goa"}, TODAY, has_itinerary=False)
    assert out["destination"] == "Goa" and out["origin"] == "Mumbai"
    assert (out["start_date"], out["end_date"]) == ("2026-10-17", "2026-10-19")
    assert out["budget_inr"] == 30000 and out["num_people"] == 2


def test_past_dates_and_impossible_values_are_dropped():
    out = clean_intent({"action": "new_trip", "destination": "Goa", "start_date": "2025-01-01",
                        "budget_inr": 50, "num_people": 40, "num_days": 90}, TODAY, False)
    assert out["start_date"] is None and out["budget_inr"] is None
    assert out["num_people"] is None and out["num_days"] is None


def test_end_date_is_filled_from_trip_length_and_capped_at_10_days():
    out = clean_intent({"action": "new_trip", "destination": "Goa", "start_date": "2026-11-01",
                        "num_days": 4}, TODAY, False)
    assert out["end_date"] == "2026-11-04"
    out = clean_intent({"action": "new_trip", "destination": "Goa", "start_date": "2026-11-01",
                        "end_date": "2026-11-30"}, TODAY, False)
    assert out["end_date"] == "2026-11-10"


def test_edit_without_an_open_itinerary_becomes_a_new_trip():
    assert clean_intent({"action": "edit", "message": "make day 2 cheaper"}, TODAY, False)["action"] == "new_trip"
    assert clean_intent({"action": "edit", "message": "make day 2 cheaper"}, TODAY, True)["action"] == "edit"


def test_parse_uses_the_model_reply(monkeypatch, fake_llm):
    import agents.voice_intent as V
    reply = '{"action": "edit", "message": "Make day 2 cheaper"}'
    monkeypatch.setattr(V, "get_llm", fake_llm([reply]))
    assert parse_voice_command("make day two cheaper", True, TODAY) == {"action": "edit", "message": "Make day 2 cheaper"}


def test_voice_endpoint_and_limit(monkeypatch, fake_llm):
    from fastapi.testclient import TestClient
    import agents.voice_intent as V
    import api
    monkeypatch.setattr(V, "get_llm", fake_llm(['{"action": "new_trip", "destination": "Goa"}'] * 3))
    monkeypatch.setattr(api, "GROQ_API_KEY", "test")
    monkeypatch.setattr(api, "VOICE_COMMANDS_PER_IP_PER_DAY", 2)
    api._usage.clear()
    c = TestClient(api.app)
    body = {"text": "plan a trip to goa", "today": "2026-10-14"}
    codes = [c.post("/voice/intent", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.post("/voice/intent", json={"text": ""}).status_code == 422
