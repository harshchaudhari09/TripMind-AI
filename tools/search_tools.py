"""
tools/search_tools.py

Upgraded Tavily search tools that find SPECIFIC hotel names,
flight options and train names with REAL PRICES.
"""
import json
from langchain.tools import tool
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY, TAVILY_SEARCH_DEPTH, budget_profile

def _tavily():
    return TavilyClient(api_key=TAVILY_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTS — specific airline names + prices
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_flights(query: str) -> str:
    """
    Search for specific flights with airline names, flight numbers and prices.
    Input format: "FROM|TO|DATE|BUDGET_INR|NUM_PEOPLE"
    Example: "Delhi|Goa|2024-12-20|80000|2"
    Returns specific flight options with names and prices in INR.
    """
    try:
        parts      = query.split("|")
        origin     = parts[0].strip()
        dest       = parts[1].strip()
        date       = parts[2].strip()
        budget     = float(parts[3].strip()) if len(parts) > 3 else 50_000
        num_people = int(parts[4].strip()) if len(parts) > 4 else 1
        profile    = budget_profile(budget)

        # Very specific search query to get real names and prices
        search_query = (
            f"flights from {origin} to {dest} on {date} "
            f"{profile['flight_class']} class IndiGo SpiceJet Air India Vistara "
            f"cheapest price INR per person booking MakeMyTrip Cleartrip"
        )

        result = _tavily().search(
            query=search_query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=7,
            include_answer=True,
        )

        return json.dumps({
            "type": "flights",
            "route": f"{origin} → {dest}",
            "date": date,
            "class": profile["flight_class"],
            "budget_per_person_inr": budget // max(num_people, 1),
            "search_answer": result.get("answer", ""),
            "sources": [
                {
                    "title":   r["title"],
                    "url":     r["url"],
                    "content": r["content"][:600],
                }
                for r in result.get("results", [])[:6]
            ],
            "booking_links": {
                "MakeMyTrip": f"https://www.makemytrip.com/flights/",
                "Cleartrip":  f"https://www.cleartrip.com/flights/",
                "Ixigo":      f"https://www.ixigo.com/flights/",
                "Skyscanner": f"https://www.skyscanner.co.in/",
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# TRAINS — specific train names, numbers and prices
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_trains(query: str) -> str:
    """
    Search for specific trains with train names, numbers and ticket prices.
    Input format: "FROM|TO|DATE|BUDGET_INR"
    Example: "Delhi|Jaipur|2024-12-20|50000"
    Returns specific train names, numbers, classes and prices.
    """
    try:
        parts   = query.split("|")
        origin  = parts[0].strip()
        dest    = parts[1].strip()
        date    = parts[2].strip()
        budget  = float(parts[3].strip()) if len(parts) > 3 else 50_000
        profile = budget_profile(budget)

        # class based on budget
        train_class = profile["train_search"]

        search_query = (
            f"train from {origin} to {dest} {train_class} "
            f"train name train number ticket price INR IRCTC schedule "
            f"available seats {date} Rajdhani Shatabdi Duronto Garib Rath"
        )

        result = _tavily().search(
            query=search_query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=6,
            include_answer=True,
        )

        return json.dumps({
            "type": "trains",
            "route": f"{origin} → {dest}",
            "date": date,
            "recommended_class": train_class,
            "search_answer": result.get("answer", ""),
            "sources": [
                {
                    "title":   r["title"],
                    "url":     r["url"],
                    "content": r["content"][:600],
                }
                for r in result.get("results", [])[:5]
            ],
            "booking_links": {
                "IRCTC":       "https://www.irctc.co.in",
                "RailYatri":   "https://www.railyatri.in",
                "ConfirmTkt":  "https://www.confirmtkt.com",
                "Ixigo Trains":"https://www.ixigo.com/trains/",
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# BUSES — specific bus operators and prices
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_buses(query: str) -> str:
    """
    Search for bus options between two destinations with operator names and prices.
    Input format: "FROM|TO|DATE|BUDGET_INR"
    Example: "Delhi|Manali|2024-12-20|30000"
    """
    try:
        parts  = query.split("|")
        origin = parts[0].strip()
        dest   = parts[1].strip()
        date   = parts[2].strip()
        budget = float(parts[3].strip()) if len(parts) > 3 else 30_000

        bus_type = "Volvo AC sleeper" if budget > 30_000 else "ordinary semi-deluxe"

        search_query = (
            f"{bus_type} bus from {origin} to {dest} on {date} "
            f"HRTC RSRTC JKSRTC RedBus operator name price INR ticket booking"
        )

        result = _tavily().search(
            query=search_query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        return json.dumps({
            "type": "buses",
            "route": f"{origin} → {dest}",
            "date": date,
            "bus_type": bus_type,
            "search_answer": result.get("answer", ""),
            "sources": [
                {
                    "title":   r["title"],
                    "url":     r["url"],
                    "content": r["content"][:500],
                }
                for r in result.get("results", [])[:4]
            ],
            "booking_links": {
                "RedBus":  "https://www.redbus.in",
                "AbhiBus": "https://www.abhibus.com",
                "HRTC":    "https://www.hrtchp.com",
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# HOTELS — specific hotel names and prices
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_hotels(query: str) -> str:
    """
    Search for specific hotels with names, ratings and prices per night.
    Input format: "DESTINATION|CHECK_IN|CHECK_OUT|BUDGET_INR|NUM_PEOPLE"
    Example: "Manali|2024-12-20|2024-12-25|80000|2"
    Returns specific hotel names, star ratings, price per night in INR.
    """
    try:
        parts    = query.split("|")
        dest     = parts[0].strip()
        checkin  = parts[1].strip()
        checkout = parts[2].strip()
        budget   = float(parts[3].strip()) if len(parts) > 3 else 50_000
        people   = int(parts[4].strip()) if len(parts) > 4 else 2
        profile  = budget_profile(budget)

        # Very targeted search for real hotel names and prices
        search_query = (
            f"best {profile['hotel_keywords']} hotels in {dest} "
            f"price per night INR {profile['hotel_min']} to {profile['hotel_max']} "
            f"check in {checkin} {people} guests "
            f"hotel name rating amenities booking Booking.com MakeMyTrip OYO"
        )

        result = _tavily().search(
            query=search_query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=8,
            include_answer=True,
        )

        # Second search for homestays if budget is mid or below
        homestay_data = {}
        if profile["tier"] in ("budget", "mid", "comfort"):
            homestay_query = (
                f"best homestays in {dest} price per night INR "
                f"{profile['hotel_min']} to {profile['hotel_max']} "
                f"Airbnb booking authentic local experience"
            )
            homestay_result = _tavily().search(
                query=homestay_query,
                search_depth="basic",
                max_results=4,
                include_answer=True,
            )
            homestay_data = {
                "homestay_answer": homestay_result.get("answer", ""),
                "homestay_sources": [
                    {"title": r["title"], "url": r["url"], "content": r["content"][:400]}
                    for r in homestay_result.get("results", [])[:3]
                ],
            }

        return json.dumps({
            "type": "hotels",
            "destination": dest,
            "checkin": checkin,
            "checkout": checkout,
            "budget_tier": profile["tier"],
            "price_range_per_night": f"₹{profile['hotel_min']:,} – ₹{profile['hotel_max']:,}",
            "hotel_search_answer": result.get("answer", ""),
            "hotel_sources": [
                {
                    "title":   r["title"],
                    "url":     r["url"],
                    "content": r["content"][:600],
                }
                for r in result.get("results", [])[:6]
            ],
            **homestay_data,
            "booking_links": {
                "MakeMyTrip": f"https://www.makemytrip.com/hotels/",
                "Booking.com": f"https://www.booking.com/",
                "Goibibo":    f"https://www.goibibo.com/hotels/",
                "OYO":        f"https://www.oyorooms.com/",
                "Airbnb":     f"https://www.airbnb.co.in/",
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# ATTRACTIONS + HIDDEN GEMS
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_attractions(query: str) -> str:
    """
    Search for tourist attractions, hidden gems and local experiences.
    Input format: "DESTINATION|INTERESTS|BUDGET_TIER"
    Example: "Spiti Valley|trekking,photography,buddhism|comfort"
    """
    try:
        parts     = query.split("|")
        dest      = parts[0].strip()
        interests = parts[1].strip() if len(parts) > 1 else "general sightseeing"
        tier      = parts[2].strip() if len(parts) > 2 else "mid"

        # Main attractions search
        main_query = (
            f"top things to do in {dest} for {interests} tourists "
            f"entry fees timings tips 2024 travel guide"
        )

        # Hidden gems search
        gems_query = (
            f"hidden gems offbeat secret places near {dest} "
            f"lesser known spots locals only no tourists unique experience"
        )

        main_result = _tavily().search(query=main_query, search_depth=TAVILY_SEARCH_DEPTH, max_results=5, include_answer=True)
        gems_result = _tavily().search(query=gems_query, search_depth=TAVILY_SEARCH_DEPTH, max_results=4, include_answer=True)

        return json.dumps({
            "destination": dest,
            "interests": interests,
            "attractions_answer": main_result.get("answer", ""),
            "attractions_sources": [
                {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
                for r in main_result.get("results", [])[:4]
            ],
            "hidden_gems_answer": gems_result.get("answer", ""),
            "hidden_gems_sources": [
                {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
                for r in gems_result.get("results", [])[:3]
            ],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FOOD
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_local_food(query: str) -> str:
    """
    Search for local food, restaurants and street food at destination.
    Input format: "DESTINATION|BUDGET_TIER"
    Example: "Jaisalmer|mid"
    """
    try:
        parts  = query.split("|")
        dest   = parts[0].strip()
        tier   = parts[1].strip() if len(parts) > 1 else "mid"

        search_query = (
            f"best local food restaurants street food in {dest} "
            f"must try dishes restaurant names prices {tier} budget "
            f"authentic local cuisine 2024"
        )

        result = _tavily().search(
            query=search_query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=5,
            include_answer=True,
        )

        return json.dumps({
            "destination": dest,
            "food_answer": result.get("answer", ""),
            "sources": [
                {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
                for r in result.get("results", [])[:4]
            ],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# All tools export
ALL_SEARCH_TOOLS = [
    search_flights,
    search_trains,
    search_buses,
    search_hotels,
    search_attractions,
    search_local_food,
]
