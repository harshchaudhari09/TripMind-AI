"""
agents/graph.py

Upgraded LangGraph pipeline with:
- Overpass API (petrol pumps, ATMs, hospitals, bus/train stations)
- Open-Meteo weather
- Wikipedia destination info
- Tavily for specific hotel names, flight names, train names with prices
- Very specific LLM prompt for detailed itinerary

Flow:  init ──┬── wiki ──────────────┬── generate_itinerary ──► critic ──► finalize ── END
              ├── weather            │            ▲                 │
              ├── flights            │            └── needs fixes ──┘
              ├── ... (10 research   │               (max MAX_REVISIONS times)
              │   nodes in parallel) │
              └── transport_stations ┘

Research nodes run at the same time, so each one returns ONLY the keys it
fills in. If two parallel nodes both returned the whole state, LangGraph
would not know which copy to keep and would raise InvalidUpdateError.
"""
from __future__ import annotations
import json
import re
from typing import TypedDict, List
from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from tools.search_tools import (
    search_flights, search_trains, search_buses,
    search_hotels, search_attractions, search_local_food,
)
from tools.overpass_tool import (
    get_petrol_pumps, get_atms, get_hospitals,
    get_bus_stations, get_train_stations,
)
from tools.weather_tool import get_weather_forecast
from tools.wikipedia_tool import get_destination_info
from utils.llm_factory import get_llm
from config.settings import budget_profile, MAX_REVISIONS, GROQ_TOKENS_PER_MINUTE
from agents import critic
from agents.budget import recompute_totals
from config.seasonal_facts import facts_for, describe


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class TravelState(TypedDict):
    # User inputs
    destination:  str
    origin:       str
    start_date:   str
    end_date:     str
    budget_inr:   float
    num_people:   int
    preferences:  List[str]

    # Derived
    budget_tier:  str
    num_days:     int      # calendar days, counting both start and end date
    num_nights:   int

    # Data collected by each node
    flight_data:      str
    train_data:       str
    bus_data:         str
    hotel_data:       str
    weather_data:     str
    attraction_data:  str
    food_data:        str
    wiki_data:        str
    petrol_data:      str
    atm_data:         str
    hospital_data:    str
    bus_station_data: str
    train_station_data: str

    # Critic loop
    critique:        List[str]   # problems found in the latest draft
    issue_log:       List[str]   # every problem found in ANY draft so far
    critic_passed:   bool
    revision_count:  int         # how many times the writer has fixed a draft
    critic_history:  list        # one entry per review, for the UI / API
    best_itinerary:  str         # draft with the fewest problems so far
    best_issues:     List[str]
    best_issue_count: int

    # Final output
    itinerary:    str
    first_draft:  str          # the draft before any critic fixes (used by eval/)
    messages:     list


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Initialise
# ─────────────────────────────────────────────────────────────────────────────
def node_init(state: TravelState) -> dict:
    profile  = budget_profile(state["budget_inr"], num_people=state.get("num_people", 1))
    start_dt = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end_dt   = datetime.strptime(state["end_date"],   "%Y-%m-%d")
    # 10 Nov → 15 Nov is 6 days (both ends count) and 5 nights.
    num_nights = max((end_dt - start_dt).days, 0)
    num_days   = num_nights + 1
    print(f"\n[init] {state['destination']} | {num_days} days / {num_nights} nights | ₹{state['budget_inr']:,.0f} | {profile['tier']} tier")
    return {
        "budget_tier": profile["tier"],
        "num_days":    num_days,
        "num_nights":  num_nights,
        "critique": [], "issue_log": [], "critic_passed": False, "revision_count": 0,
        "critic_history": [], "best_itinerary": "", "best_issues": [], "best_issue_count": 10**6,
        "flight_data": "", "train_data": "", "bus_data": "",
        "hotel_data": "", "weather_data": "", "attraction_data": "",
        "food_data": "", "wiki_data": "",
        "petrol_data": "", "atm_data": "", "hospital_data": "",
        "bus_station_data": "", "train_station_data": "",
        "itinerary": "", "first_draft": "", "messages": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Research nodes — all run in parallel after init
# ─────────────────────────────────────────────────────────────────────────────
def node_wiki(state: TravelState) -> dict:
    print(f"[research] Wikipedia info for {state['destination']}...")
    return {"wiki_data": get_destination_info.invoke(state["destination"])}


def node_weather(state: TravelState) -> dict:
    print(f"[research] Weather forecast...")
    q = f"{state['destination']}|{state['start_date']}|{state['end_date']}"
    return {"weather_data": get_weather_forecast.invoke(q)}


def node_flights(state: TravelState) -> dict:
    # Outbound and return live in one node because they share the flight_data key.
    print(f"[research] Searching flights {state['origin']} ⇄ {state['destination']}...")
    q_out = f"{state['origin']}|{state['destination']}|{state['start_date']}|{state['budget_inr']}|{state['num_people']}"
    q_ret = f"{state['destination']}|{state['origin']}|{state['end_date']}|{state['budget_inr']}|{state['num_people']}"
    outbound      = search_flights.invoke(q_out)
    return_flight = search_flights.invoke(q_ret)
    combined = json.dumps({
        "outbound": json.loads(outbound) if outbound else {},
        "return": json.loads(return_flight) if return_flight else {},
    })
    return {"flight_data": combined}


def node_trains(state: TravelState) -> dict:
    print(f"[research] Searching trains...")
    q = f"{state['origin']}|{state['destination']}|{state['start_date']}|{state['budget_inr']}"
    return {"train_data": search_trains.invoke(q)}


def node_buses(state: TravelState) -> dict:
    print(f"[research] Searching buses...")
    q = f"{state['origin']}|{state['destination']}|{state['start_date']}|{state['budget_inr']}"
    return {"bus_data": search_buses.invoke(q)}


def node_hotels(state: TravelState) -> dict:
    print(f"[research] Searching hotels in {state['destination']}...")
    q = f"{state['destination']}|{state['start_date']}|{state['end_date']}|{state['budget_inr']}|{state['num_people']}"
    return {"hotel_data": search_hotels.invoke(q)}


def node_attractions(state: TravelState) -> dict:
    print(f"[research] Finding attractions and hidden gems...")
    interests = ", ".join(state.get("preferences", ["sightseeing"]))
    q = f"{state['destination']}|{interests}|{state['budget_tier']}"
    return {"attraction_data": search_attractions.invoke(q)}


def node_food(state: TravelState) -> dict:
    print(f"[research] Finding local food and restaurants...")
    q = f"{state['destination']}|{state['budget_tier']}"
    return {"food_data": search_local_food.invoke(q)}


def node_local_facilities(state: TravelState) -> dict:
    print(f"[research] Finding petrol pumps, ATMs, hospitals (OpenStreetMap)...")
    return {
        "petrol_data":   get_petrol_pumps.invoke(state["destination"]),
        "atm_data":      get_atms.invoke(state["destination"]),
        "hospital_data": get_hospitals.invoke(state["destination"]),
    }


def node_transport_stations(state: TravelState) -> dict:
    print(f"[research] Finding bus and train stations (OpenStreetMap)...")
    return {
        "bus_station_data":   get_bus_stations.invoke(state["destination"]),
        "train_station_data": get_train_stations.invoke(state["destination"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ITINERARY PROMPT — very specific instructions
# ─────────────────────────────────────────────────────────────────────────────
ITINERARY_SYSTEM_PROMPT = """You are an expert Indian travel planner with 20 years of experience.
Write a SPECIFIC, PRACTICAL day-by-day itinerary that a normal traveller can read on a phone.

## CONTENT RULES
- Use real names and prices from the research data. Prefer places that appear in the data.
- Never invent a name. If the research has no named option for something (e.g. a restaurant
  for a meal), describe it plainly instead ("a local dhaba on Mall Road") — don't bold it.
- Transport: recommend ONE best way there and back (name + number + price per person),
  then one short line on why it beats the alternatives for this budget.
- Stays: 3 options by NAME from the HOTEL OPTIONS data. Include a homestay only if the data
  names a real one — otherwise list 3 hotels. For each: price/night, star rating,
  what's included, and where to book (MakeMyTrip / Booking.com / OYO / Zostel).
- Each day: 4-7 timed stops with distance/travel time, entry fee, and meals with
  restaurant name + dish + price per person. Timings must be realistic
  (allow at least 2 hours between connecting flights; add travel time between places).
- Budget: costs for ALL travellers, and the TOTAL must be at or under the given budget.
  (The Total row is recalculated by code from your line items, so get each line item right.)
- Include permits and safety warnings if relevant.
- Season: check the travel month. Never plan passes, treks, rafting, ropeways or roads that are
  closed or unsafe in that month (e.g. Rohtang Pass is closed to tourists Nov–Apr). If unsure
  whether something is open, pick a safe alternative that is open all year.

## OUTPUT FORMAT — follow this exactly
- Use the exact section headings below, in this order, with the emoji.
- Use short bullet points (one line each). No paragraphs longer than 2 lines.
- The ONLY table allowed is the 2-column budget table. Never put days, hotels or transport in a table.
- Put times in bold at the start of a bullet, 12-hour format: **8:00 AM** — ...
- Overnight bus/train: the day it leaves ENDS with the departure; the arrival is the FIRST
  bullet of the next day, followed by that day's plans. Example:
    ### Day 1 — Thu, 10 Dec · Travel day
    - **4:30 PM** — Board overnight Volvo bus from Bengaluru
    ### Day 2 — Fri, 11 Dec · Munnar town
    - **6:00 AM** — Arrive Munnar, check in
- No activities after 11 PM. Check real timings (e.g. Ganga Aarti is at sunset, not midnight).
- Use the day headings given in the request, one per day. Never skip or merge days.

# {emoji} {Destination} — {N}-Day Trip
> **{Origin} → {Destination}** · {dates} · {people} · Budget ₹X · Estimated total ₹Y

## ✨ At a glance
- **Getting there:** one line
- **Stay:** recommended stay, one line
- **Weather:** one line
- **Total cost:** ₹Y of ₹X budget (₹Z left over)

## ✈️ Getting there & back
- **Go:** ...
- **Return:** ...
- **Why this option:** one line

## 🏨 Where to stay
1. **Name** — ₹price/night · rating · included · book on Platform
2. ...
3. ...

## 🗓️ Day-by-day plan
### Day 1 — {weekday, date} · {short theme}
- **8:00 AM** — ...
- **1:00 PM** — Lunch at **Restaurant** — Dish, ₹price/person
- 💡 *Tip:* one line
(repeat for every day)

## 🍽️ Food to try
- **Dish** at **Restaurant** — ₹price/person (3-5 bullets)

## 💰 Budget breakdown
| Item | Cost |
|---|---|
| Transport (to/from) | ₹... |
| Stay (N nights) | ₹... |
| Food (N days × people) | ₹... |
| Local travel + activities | ₹... |
| Buffer | ₹... |
| **Total** | **₹...** |

## 🎒 Weather & packing
- 2-4 bullets based on the forecast data

## 🏧 Useful nearby
- **ATM:** name — area
- **Petrol:** name — distance
- **Hospital:** name — area, phone if known
- **Bus stand / Railway station:** names

## 💡 Local tips
- 5 short tips, including permits and safety
"""


# ─────────────────────────────────────────────────────────────────────────────
# Final node: Generate Itinerary (runs once every research node is done)
# ─────────────────────────────────────────────────────────────────────────────
def _prompt_data(value: str, max_chars: int = 1200) -> str:
    """Keep the final request within Groq's free-tier token-per-minute limit."""
    value = str(value or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n[Additional results omitted to fit the model limit.]"


def _travel_months(state: TravelState) -> str:
    start = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end   = datetime.strptime(state["end_date"], "%Y-%m-%d")
    return start.strftime("%B") if start.month == end.month else f"{start:%B}–{end:%B}"


def _season_block(state: TravelState) -> str:
    facts = facts_for(state["destination"])
    if not facts:
        return ""
    return ("\n## VERIFIED SEASON FACTS (never plan anything marked CLOSED)\n"
            + describe(facts, critic.trip_months(state["start_date"], state["end_date"])) + "\n")


def _fit_max_tokens(messages: list, wanted: int) -> int:
    """
    Groq's free tier rejects any single request whose prompt + max_tokens is over the
    tokens-per-minute limit (8000). Count the prompt and shrink the output allowance to fit.
    """
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")       # same tokenizer family as gpt-oss
    prompt = sum(len(enc.encode(m.content)) + 8 for m in messages) + 60
    return max(1500, min(wanted, GROQ_TOKENS_PER_MINUTE - 100 - prompt))


def _trip_block(state: TravelState) -> str:
    """Trip facts + exact day headings. Used by both the first draft and revisions."""
    profile  = budget_profile(state["budget_inr"], state["num_days"], state["num_people"])
    # Exact day headings, so the model can't skip, merge or misdate days.
    start_dt = datetime.strptime(state["start_date"], "%Y-%m-%d")
    day_headings = "\n".join(
        f"- Day {i + 1} — {(start_dt + timedelta(days=i)).strftime('%a, %d %b')}"
        for i in range(state["num_days"])
    )
    return f"""
## TRIP DETAILS
- **Destination:** {state['destination']}
- **Departing From:** {state['origin']}
- **Dates:** {state['start_date']} to {state['end_date']} ({state['num_days']} days, {state['num_nights']} nights — the stay costs {state['num_nights']} nights)
- **Total Budget:** ₹{state['budget_inr']:,.0f} for {state['num_people']} person(s)
- **Budget Tier:** {state['budget_tier'].upper()} | Hotel Range: ₹{profile['hotel_min']:,}–₹{profile['hotel_max']:,}/night
- **Interests:** {', '.join(state.get('preferences', []))}

## DAY HEADINGS (use exactly these, one ### heading each, then add a short theme)
{day_headings}
Day 1 starts with the journey from {state['origin']}. Day {state['num_days']} ends with the return journey.
Travel month: {_travel_months(state)} — only plan activities that are open and safe in this month.
{_season_block(state)}"""


def node_generate_itinerary(state: TravelState) -> dict:
    if state.get("critique") and state.get("itinerary"):
        return _revise_itinerary(state)

    print(f"[generate] All research done — generating detailed itinerary with LLM...")
    profile = budget_profile(state["budget_inr"], state["num_days"], state["num_people"])

    # Parse weather for summary
    try:
        weather_parsed = json.loads(state["weather_data"])
        weather_summary = weather_parsed.get("summary", {})
        packing = weather_parsed.get("packing_advice", [])
        forecast_days = weather_parsed.get("forecast", [])
    except Exception:
        weather_summary = {}
        packing = []
        forecast_days = []

    # Parse wiki for description
    try:
        wiki_parsed = json.loads(state["wiki_data"])
        wiki_desc = wiki_parsed.get("description", "")[:800]
    except Exception:
        wiki_desc = ""

    # Parse facilities
    try:
        petrol_parsed  = json.loads(state["petrol_data"])
        atm_parsed     = json.loads(state["atm_data"])
        hospital_parsed= json.loads(state["hospital_data"])
        bus_st_parsed  = json.loads(state["bus_station_data"])
        train_st_parsed= json.loads(state["train_station_data"])
    except Exception:
        petrol_parsed = atm_parsed = hospital_parsed = bus_st_parsed = train_st_parsed = {}

    user_message = _trip_block(state) + f"""
---

## DESTINATION OVERVIEW (Wikipedia)
{wiki_desc}

---

## WEATHER FORECAST
{json.dumps(forecast_days, indent=2)}
**Summary:** {json.dumps(weather_summary)}
**Recommended Packing:** {', '.join(packing)}

---

## FLIGHT OPTIONS (Outbound + Return)
{_prompt_data(state['flight_data'])}

---

## TRAIN OPTIONS
{_prompt_data(state['train_data'])}

---

## BUS OPTIONS
{_prompt_data(state['bus_data'])}

---

## HOTEL OPTIONS (Budget: ₹{profile['hotel_min']:,}–₹{profile['hotel_max']:,}/night)
{_prompt_data(state['hotel_data'])}

---

## ATTRACTIONS + HIDDEN GEMS
{_prompt_data(state['attraction_data'])}

---

## LOCAL FOOD + RESTAURANTS
{_prompt_data(state['food_data'])}

---

## PETROL PUMPS NEAR {state['destination'].upper()}
{_prompt_data(json.dumps(petrol_parsed.get('petrol_pumps', []), indent=2), 700)}

---

## ATMs NEAR {state['destination'].upper()}
{_prompt_data(json.dumps(atm_parsed.get('atms', []), indent=2), 700)}

---

## HOSPITALS NEAR {state['destination'].upper()}
{_prompt_data(json.dumps(hospital_parsed.get('hospitals_and_clinics', []), indent=2), 700)}

---

## BUS STATIONS
{_prompt_data(json.dumps(bus_st_parsed.get('bus_stations', []), indent=2), 700)}

---

## RAILWAY STATIONS
{_prompt_data(json.dumps(train_st_parsed.get('railway_stations', []), indent=2), 700)}

---

Now write the complete itinerary using the OUTPUT FORMAT from your instructions, with all {state['num_days']} days.
The total cost must stay within ₹{state['budget_inr']:,.0f}.
"""

    messages = [
        SystemMessage(content=ITINERARY_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    llm = get_llm(temperature=0.4, max_tokens=_fit_max_tokens(messages, 4000), reasoning_effort="low")
    response  = llm.invoke(messages, config={"run_name": "writer: first draft"})
    itinerary = response.content if hasattr(response, "content") else str(response)
    itinerary, _ = recompute_totals(itinerary, state["budget_inr"])

    return {"itinerary": itinerary, "first_draft": itinerary, "messages": messages + [response]}


def _revise_itinerary(state: TravelState) -> dict:
    """
    Fix a draft the critic rejected. The research data is NOT re-sent: the draft
    already contains the chosen places, and prompt + draft + 3500 output tokens
    must fit inside Groq's 8000 tokens-per-request limit.
    """
    attempt = state["revision_count"] + 1
    print(f"[revise] Fixing {len(state['critique'])} problem(s) — revision {attempt}/{MAX_REVISIONS}...")
    problems = "\n".join(f"{i}. {p}" for i, p in enumerate(state["critique"], 1))
    earlier  = [p for p in state.get("issue_log", []) if p not in state["critique"]]
    earlier_block = ("\n## PROBLEMS FROM EARLIER DRAFTS (already fixed — do NOT bring these back)\n"
                     + "\n".join(f"- {p}" for p in earlier) + "\n") if earlier else ""
    user_message = _trip_block(state) + f"""
---

## YOUR PREVIOUS DRAFT
{state['itinerary']}

---

## PROBLEMS FOUND BY THE REVIEWER
{problems}
{earlier_block}
Rewrite the COMPLETE itinerary, fixing every problem above. Keep everything that was
not mentioned the same. Follow the OUTPUT FORMAT exactly, with all {state['num_days']} days,
and keep the budget table's line items accurate.
"""
    messages = [SystemMessage(content=ITINERARY_SYSTEM_PROMPT), HumanMessage(content=user_message)]
    llm = get_llm(temperature=0.3, max_tokens=_fit_max_tokens(messages, 3500), reasoning_effort="low")
    response = llm.invoke(messages, config={"run_name": f"writer: fix round {attempt}"})
    itinerary = response.content if hasattr(response, "content") else str(response)
    itinerary, _ = recompute_totals(itinerary, state["budget_inr"])
    return {"itinerary": itinerary, "messages": messages + [response], "revision_count": attempt}


# ─────────────────────────────────────────────────────────────────────────────
# Critic agent — reviews every draft (see agents/critic.py)
# ─────────────────────────────────────────────────────────────────────────────
def node_critic(state: TravelState) -> dict:
    print(f"[critic] Reviewing draft (revision {state['revision_count']})...")
    result = critic.review(state["itinerary"], state, earlier_issues=state.get("issue_log", []))
    issues = result["rule_issues"] + result["llm_issues"]
    if result["passed"]:
        print("[critic] ✅ Approved")
    else:
        print(f"[critic] ❌ {len(issues)} problem(s) found:")
        for issue in issues:
            print(f"   - {issue}")

    update = {
        "critique":       issues,
        "issue_log":      state.get("issue_log", []) + [i for i in issues if i not in state.get("issue_log", [])],
        "critic_passed":  result["passed"],
        "critic_history": state["critic_history"] + [{"revision": state["revision_count"], **result}],
    }
    # Remember the best draft so far, in case later revisions make things worse.
    # "<=" means a tie goes to the NEWER draft: it has already fixed the earlier
    # drafts' problems, while the older draft still contains them.
    if len(issues) <= state["best_issue_count"]:
        update["best_itinerary"]   = state["itinerary"]
        update["best_issues"]      = issues
        update["best_issue_count"] = len(issues)
    return update


def _same_issue(a: str, b: str) -> bool:
    """Same problem, maybe reworded: same day number(s) and mostly the same words."""
    days = lambda s: set(re.findall(r"day\W*(\d+)", s.lower()))
    words = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))
    wa, wb = words(a), words(b)
    overlap = len(wa & wb) / max(1, len(wa | wb))
    return days(a) == days(b) and overlap >= 0.5


def _is_stuck(state: TravelState) -> bool:
    """True if every problem this round was already reported last round — another fix won't help."""
    history = state["critic_history"]
    if len(history) < 2:
        return False
    now  = history[-1]["rule_issues"] + history[-1]["llm_issues"]
    prev = history[-2]["rule_issues"] + history[-2]["llm_issues"]
    return bool(now) and all(any(_same_issue(a, b) for b in prev) for a in now)


def route_after_critic(state: TravelState) -> str:
    """The decision point: approve, or send the draft back for another fix."""
    if state["critic_passed"] or state["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    if _is_stuck(state):
        print("[critic] Same problems as last round — stopping early to save tokens.")
        return "finalize"
    return "generate_itinerary"


def node_finalize(state: TravelState) -> dict:
    if state["critic_passed"]:
        return {}
    # Out of retries: show the draft with the fewest problems.
    print(f"[finalize] Using best draft ({state['best_issue_count']} problem(s) left after "
          f"{state['revision_count']} revision(s)).")
    return {"itinerary": state["best_itinerary"], "critique": state["best_issues"]}


# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_NODES = {
    "wiki":               node_wiki,
    "weather":            node_weather,
    "flights":            node_flights,
    "trains":             node_trains,
    "buses":              node_buses,
    "hotels":             node_hotels,
    "attractions":        node_attractions,
    "food":               node_food,
    "local_facilities":   node_local_facilities,
    "transport_stations": node_transport_stations,
}


def build_graph(research_nodes: dict | None = None):
    """research_nodes lets eval/ swap in cached research instead of live searches."""
    research_nodes = research_nodes or RESEARCH_NODES
    g = StateGraph(TravelState)

    g.add_node("init",               node_init)
    g.add_node("generate_itinerary", node_generate_itinerary)
    g.add_node("critic",             node_critic)
    g.add_node("finalize",           node_finalize)
    for name, fn in research_nodes.items():
        g.add_node(name, fn)

    g.set_entry_point("init")

    # Fan-out: init → every research node (they start together).
    for name in research_nodes:
        g.add_edge("init", name)

    # Fan-in: generate_itinerary waits until ALL research nodes have finished.
    g.add_edge(list(research_nodes), "generate_itinerary")

    # Self-checking loop: every draft goes to the critic, which either
    # approves it (→ finalize) or sends it back to be fixed.
    g.add_edge("generate_itinerary", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"generate_itinerary": "generate_itinerary", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile()


travel_graph = build_graph()
