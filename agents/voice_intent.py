"""
agents/voice_intent.py

Turns a spoken request (already converted to text by the browser) into an action:

  "Plan a 3-day trip to Goa from Mumbai next weekend for 2, budget 30k"
      → {"action": "new_trip", "destination": "Goa", "origin": "Mumbai", ...}
  "Make day 2 cheaper"
      → {"action": "edit", "message": "Make day 2 cheaper"}

The small model only extracts what was said. Code then checks every value (dates in the
future, trip length, budget, group size) so a misheard number can't produce a bad request.
"""
from __future__ import annotations
import json
import re
from datetime import date, timedelta

from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import CRITIC_MODEL
from utils.llm_factory import get_llm

MAX_TRIP_DAYS = 10

INTENT_PROMPT = """You turn a traveller's spoken request into JSON for TripMind, an Indian trip planner.
Today is {today} ({weekday}). {context}

Choose the action:
- "new_trip": they want a NEW trip planned (e.g. "plan a trip to Goa", "I want to go to Manali").
- "edit": they want to change the itinerary that is currently open.
- "question": they are asking something about the current trip.

Reply with JSON only:
{{"action": "new_trip" | "edit" | "question",
  "destination": string or null, "origin": string or null,
  "start_date": "YYYY-MM-DD" or null, "end_date": "YYYY-MM-DD" or null,
  "num_days": integer or null, "budget_inr": number or null, "num_people": integer or null,
  "preferences": [strings], "message": string}}

Rules:
- Resolve relative dates against today: "next weekend" = the coming Saturday to Sunday,
  "in December" = early December, "from the 5th to the 8th" = the next such dates. Dates must
  be in the future.
- If only a length is said ("3 days", "a week"), set num_days and leave the dates null.
- Budget in rupees: "30k" = 30000, "1.5 lakh" = 150000. "20k each" = 20000 × number of people.
- "me and my wife" = 2 people, "family of four" = 4, "solo" = 1.
- Use null for anything not said. Never invent an origin or destination.
- num_people ONLY if the number of travellers was actually said ("for 2 days" is a trip
  length, not 2 people). Don't mention a number of people in "message" unless it was said.
- "message": for edit/question, the request rewritten as one clear instruction or question
  (e.g. "Make day 2 cheaper"); for new_trip, a short confirmation such as
  "Planning a 3-day trip to Goa from Mumbai for 2 people".
"""


def _iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def clean_intent(data: dict, today: date, has_itinerary: bool) -> dict:
    """Validate what the model extracted. Anything doubtful becomes None (the form keeps its value)."""
    action = data.get("action")
    if action not in ("new_trip", "edit", "question"):
        action = "edit" if has_itinerary else "new_trip"
    if action in ("edit", "question") and not has_itinerary:
        action = "new_trip"                          # nothing to edit yet

    out = {"action": action, "message": str(data.get("message") or "").strip()[:300]}
    if action != "new_trip":
        return out

    start, end = _iso(data.get("start_date")), _iso(data.get("end_date"))
    if start and start < today:
        start = None                                 # misheard / past date: keep the form's date
    if end and (not start or end < start):
        end = None
    days = data.get("num_days")
    days = int(days) if isinstance(days, (int, float)) and 1 <= days <= MAX_TRIP_DAYS else None
    if start and not end and days:
        end = start + timedelta(days=days - 1)
    if start and end and (end - start).days + 1 > MAX_TRIP_DAYS:
        end = start + timedelta(days=MAX_TRIP_DAYS - 1)

    budget = data.get("budget_inr")
    people = data.get("num_people")
    prefs = [str(p).strip()[:40] for p in (data.get("preferences") or []) if str(p).strip()][:6]
    out.update({
        "destination": (str(data["destination"]).strip()[:80] or None) if data.get("destination") else None,
        "origin":      (str(data["origin"]).strip()[:80] or None) if data.get("origin") else None,
        "start_date":  start.isoformat() if start else None,
        "end_date":    end.isoformat() if end else None,
        "num_days":    days,
        "budget_inr":  float(budget) if isinstance(budget, (int, float)) and 1000 <= budget <= 10_000_000 else None,
        "num_people":  int(people) if isinstance(people, (int, float)) and 1 <= people <= 10 else None,
        "preferences": prefs,
    })
    return out


def parse_voice_command(text: str, has_itinerary: bool, today: date, current_trip: dict | None = None) -> dict:
    context = "No itinerary is open yet."
    if has_itinerary and current_trip:
        context = (f"An itinerary is open: {current_trip.get('origin')} → {current_trip.get('destination')}, "
                   f"{current_trip.get('start_date')} to {current_trip.get('end_date')}.")
    prompt = INTENT_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"), context=context)
    llm = get_llm(temperature=0, max_tokens=900, reasoning_effort="low", model=CRITIC_MODEL)
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=text)],
                          config={"run_name": "voice: understand request"})
    raw = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    return clean_intent(data, today, has_itinerary)
