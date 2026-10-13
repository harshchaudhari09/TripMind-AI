"""The evaluation's place extraction: count real places, not dishes, times or activities."""
from eval import scoring

TRIP = {"destination": "Manali, Himachal Pradesh", "origin": "Delhi"}

PLAN = """## 🏨 Where to stay
1. **The Orchid Manali** — ₹6,000/night
## 🗓️ Day-by-day plan
### Day 1 — Mon · x
- **12:30 PM** — Lunch at **Manali Dhaam** (Solang) — **Thukpa**, ₹350 pp
- **3:00 PM** — Coffee at **Cafe 1947** — Traditional **Cold Brew**, ₹180
- **5:00 PM** — Guided **Himachali cooking class** at **Himachali Rasoi**
- **7:00 PM** — Take a **taxi** to **Hadimba Temple**
## 🍽️ Food to try
- **Siddu** at **Johnson's Cafe** — ₹300
"""


def test_places_are_found_and_dishes_skipped():
    places = scoring.extract_places(PLAN, TRIP)
    for real in ["The Orchid Manali", "Manali Dhaam", "Cafe 1947", "Himachali Rasoi", "Hadimba Temple", "Johnson's Cafe"]:
        assert real in places
    for not_a_place in ["Thukpa", "Cold Brew", "Siddu", "12:30 PM", "taxi", "Himachali cooking class"]:
        assert not_a_place not in places


def test_name_variants_handle_brackets_dashes_and_possessives():
    v = scoring._variants("Hotel Kamal – Near Taj Mahal")
    assert "hotel kamal" in v
    assert any("1947" in x for x in scoring._variants("Old Manali’s Café 1947"))


def test_place_found_in_research_text():
    research = " " + scoring.normalise("Top picks: Hotel Palam Hills, Leh. Also Zasgyath (Grand Dragon).") + " "
    assert scoring.in_research("Hotel Palam Hills", research)
    assert scoring.in_research("Zasgyath (Grand Dragon)", research)
    assert not scoring.in_research("Snowy Pines Homestay", research)
