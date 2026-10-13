"""Public-demo protection: request validation, usage limits and example trips."""
import json
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

TRIP = {"destination": "Goa", "origin": "Mumbai", "start_date": "2026-12-05",
        "end_date": "2026-12-06", "budget_inr": 20000}


@pytest.fixture
def client(monkeypatch):
    # Swap the planner for a stub so /plan returns instantly without API calls.
    stub = types.ModuleType("agents.graph")

    class Graph:
        def invoke(self, state, config=None):
            return {"itinerary": "# ok", "budget_tier": "mid", "num_days": 2,
                    "critic_passed": True, "revision_count": 0}
    stub.travel_graph = Graph()
    monkeypatch.setitem(sys.modules, "agents.graph", stub)
    import api
    monkeypatch.setattr(api, "GROQ_API_KEY", "test")
    monkeypatch.setattr(api, "TAVILY_API_KEY", "test")
    api._usage.clear()
    return api, TestClient(api.app)


def test_valid_trip_is_planned(client):
    _, c = client
    assert c.post("/plan", json=TRIP).status_code == 200


@pytest.mark.parametrize("change, message", [
    ({"end_date": "2027-01-30"}, "at most 10 days"),
    ({"end_date": "2026-12-01"}, "on or after"),
    ({"budget_inr": 10}, "budget_inr"),
    ({"num_people": 50}, "num_people"),
])
def test_bad_requests_are_rejected_before_using_tokens(client, change, message):
    _, c = client
    r = c.post("/plan", json={**TRIP, **change})
    assert r.status_code == 422 and message in r.text


def test_per_visitor_limit(client, monkeypatch):
    api, c = client
    monkeypatch.setattr(api, "PLANS_PER_IP_PER_DAY", 2)
    codes = [c.post("/plan", json=TRIP, headers={"x-forwarded-for": "1.1.1.1"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.post("/plan", json=TRIP, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200


def test_site_wide_limit(client, monkeypatch):
    api, c = client
    monkeypatch.setattr(api, "DAILY_PLAN_LIMIT", 2)
    codes = [c.post("/plan", json=TRIP, headers={"x-forwarded-for": f"9.9.9.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_published_example_trips_pass_every_rule_check():
    from agents import critic
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "examples.json")
    examples = json.load(open(path))
    assert examples, "no example trips published"
    for ex in examples:
        trip = {**ex["trip"], "num_days": ex["num_days"]}
        assert critic.rule_checks(ex["itinerary"], ex["num_days"], trip["budget_inr"], trip) == [], ex["id"]
