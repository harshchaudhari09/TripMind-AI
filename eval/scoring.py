"""
eval/scoring.py

Scores one itinerary on three questions:
1. Correct number of days?   — every '### Day N' heading present exactly once
2. Within budget?            — budget table exists, adds up, and Total ≤ budget
3. No made-up places?        — every named stay / restaurant / attraction is either
                               a) in the research data the app collected, or
                               b) found on OpenStreetMap near the destination.
                               Names that fail both are "unverified" (possibly made up).

Everything here is plain Python — no LLM calls — so scores are repeatable.
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
import unicodedata

import requests

from agents import critic

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PLACES_CACHE = os.path.join(CACHE_DIR, "places.json")
# Bump when the lookup logic changes, so old "not found" answers are retried.
MAP_LOOKUP_VERSION = 2

# Bold text that is not a place name.
_PLATFORMS = {"booking.com", "makemytrip", "oyo", "goibibo", "agoda", "airbnb", "irctc",
              "redbus", "cleartrip", "yatra", "zostel", "hostelworld", "tripadvisor"}
_ACTIVITY = re.compile(r"\b(class|tour|walk|ride|safari|session|lesson|workshop|show|in-house|market stroll|"
                       r"sledding|skiing|paragliding|rafting|zorbing|snorkell?ing|scuba|kayaking)\b", re.I)
_TRANSPORT = re.compile(r"\b(flight|airlines?|indigo|air india|vistara|spicejet|akasa|express|"
                        r"shatabdi|rajdhani|vande bharat|train|bus|volvo|taxi|cab|auto|jeep|"
                        r"ferry|airport|railway|station|metro)\b", re.I)
# Words removed when matching "Hotel Palam Hills" against "Palam Hills Hotel".
_GENERIC = {"hotel", "hotels", "resort", "resorts", "homestay", "home", "stay", "guest", "house",
            "guesthouse", "hostel", "cafe", "restaurant", "dhaba", "the", "inn", "lodge", "villa",
            "camp", "camps", "and", "by", "at", "of", "residency", "suites", "boutique",
            "private", "room", "rooms"}


def normalise(text: str) -> str:
    text = re.sub(r"[’'`]", "", text)          # "Johnson’s" → "Johnsons", not "johnson s"
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1 + 2: days and budget (reuses the critic's rule checks)
# ─────────────────────────────────────────────────────────────────────────────
def score_structure(itinerary: str, num_days: int, budget_inr: float, trip: dict | None = None) -> dict:
    text = critic._normalise(itinerary)
    items, total = critic.parse_budget_table(text)
    line_sum = sum(a for _, a in items)
    has_table = bool(items) and total is not None
    return {
        "days_ok":       not critic.check_days(text, num_days),
        "days_found":    sorted(critic._split_days(text).keys()),
        "has_budget":    has_table,
        "budget_total":  total,
        "adds_up":       has_table and abs(line_sum - total) <= max(500, 0.02 * total),
        "within_budget": has_table and max(total, line_sum) <= budget_inr,
        "time_order_ok": not critic.check_time_order(text),
        # Places from the verified seasonal list planned while closed (config/seasonal_facts.py)
        "season_ok":     not critic.check_season(text, trip) if trip else True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3: place extraction
# ─────────────────────────────────────────────────────────────────────────────
def _sections(text: str) -> dict:
    """{'where to stay': '...', 'day by day plan': '...', ...} keyed by normalised '## ' heading."""
    parts = re.split(r"^##\s+(?!#)(.*)$", text, flags=re.MULTILINE)
    return {normalise(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _is_place(name: str, trip: dict) -> bool:
    n = name.strip()
    low = normalise(n)
    if (len(low) < 3 or ":" in n or "₹" in n or re.match(r"^\d", n)
            or low in _PLATFORMS or _TRANSPORT.search(n) or _ACTIVITY.search(n)
            or low in (normalise(trip["destination"]), normalise(trip["origin"]))
            or low.split()[0] in ("day", "total", "tip", "go", "return")):
        return False
    return True


def extract_places(itinerary: str, trip: dict) -> list[str]:
    text = critic._normalise(itinerary)
    found = []
    for heading, body in _sections(text).items():
        if "stay" in heading:
            # "1. **Hotel X** — ₹…" → the first bold on each numbered line
            found += re.findall(r"^\s*\d+\.\s*\*\*(.+?)\*\*", body, re.MULTILINE)
        elif "day" in heading:
            for line in body.splitlines():
                spans = list(re.finditer(r"\*\*(.+?)\*\*", line))
                for i, m in enumerate(spans):
                    before = line[:m.start()]
                    prev = spans[i - 1] if i else None
                    gap = line[prev.end():m.start()] if prev else ""
                    # Dishes, not places: "at **Café** (Old Manali) — Traditional **Chana Madra**".
                    # Only when the previous bold is a place — "**11:30 AM** — Explore **X**" is a place.
                    after_place = prev and not re.match(r"\d{1,2}:\d{2}", prev.group(1))
                    if ((after_place and re.match(r"^[^*]{0,25}[—–-]\s*(\w+\s+){0,2}$", gap))
                            or re.search(r"(\b(try|taste|eat|order|have|sip|savou?r|snack)\b[^*]{0,15}|&\s*)$", before, re.I)):
                        continue
                    found.append(m.group(1))
        elif "food" in heading:
            # "**Dish** at **Restaurant**" → only the restaurant is a place
            found += re.findall(r"\bat\s+\*\*(.+?)\*\*", body)
    places, seen = [], set()
    for name in found:
        name = re.sub(r"\s+", " ", name).strip(" *")
        if _is_place(name, trip) and normalise(name) not in seen:
            seen.add(normalise(name))
            places.append(name)
    return places


# ─────────────────────────────────────────────────────────────────────────────
# 3: place verification
# ─────────────────────────────────────────────────────────────────────────────
def research_text(state: dict) -> str:
    """All research results as one normalised string (JSON decoded so names aren't escaped)."""
    chunks = []
    for key, value in state.items():
        if key.endswith("_data") and value:
            try:
                chunks.append(json.dumps(json.loads(value), ensure_ascii=False))
            except (TypeError, ValueError):
                chunks.append(str(value))
    return " " + normalise(" ".join(chunks)) + " "


def _variants(name: str) -> list[str]:
    """'Zasgyath (Grand Dragon) Hotel' → ['zasgyath grand dragon hotel', 'zasgyath', 'grand dragon', ...]"""
    names = [name] + re.findall(r"\(([^)]+)\)", name) + [re.sub(r"\([^)]*\)", "", name)]
    names += [re.split(r"\s+[—–-]\s+", name)[0]]              # "Hotel Kamal – Near Taj" → "Hotel Kamal"
    names += [m for m in re.findall(r"[’']s\s+(.+)$", name)]     # "Old Manali’s Café 1947" → "Café 1947"
    out = []
    for n in names:
        full = normalise(n)
        core = " ".join(w for w in full.split() if w not in _GENERIC)
        out += [v for v in (full, core) if len(v) >= 4]
    return list(dict.fromkeys(out))


def in_research(name: str, research: str) -> bool:
    return any(f" {v} " in research for v in _variants(name))


class MapChecker:
    """Looks names up on OpenStreetMap (Nominatim) inside a box around the destination.
    Nominatim allows ~1 request/second, so calls are spaced out and cached on disk."""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            self.cache = json.load(open(PLACES_CACHE))
        except (OSError, ValueError):
            self.cache = {}
        self._last = 0.0
        self._lock = threading.Lock()

    def _get(self, params: dict):
        with self._lock:
            wait = 1.1 - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
        resp = requests.get("https://nominatim.openstreetmap.org/search",
                            params={**params, "format": "json", "limit": 1},
                            headers={"User-Agent": "TripMindAI-eval/1.0 student-project"}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def centre(self, destination: str):
        key = f"__centre__|{normalise(destination)}"
        if key not in self.cache:
            data = self._get({"q": destination})
            self.cache[key] = [float(data[0]["lat"]), float(data[0]["lon"])] if data else None
            self.save()
        return self.cache[key]

    def exists_near(self, name: str, destination: str, box_deg: float = 0.6) -> bool | None:
        """True/False, or None if the lookup failed (network) — None is not counted either way."""
        key = f"v{MAP_LOOKUP_VERSION}|{normalise(name)}|{normalise(destination)}"
        if key in self.cache:
            return self.cache[key]
        try:
            centre = self.centre(destination)
            if not centre:
                return None
            lat, lon = centre
            # ~0.6° ≈ 65 km box; bounded=1 means "only results inside this box".
            viewbox = f"{lon - box_deg},{lat + box_deg},{lon + box_deg},{lat - box_deg}"
            found = False
            # Try the name exactly as written first, then the simplified "core" name.
            queries = list(dict.fromkeys([re.sub(r"\s*\([^)]*\)", "", name).strip()] + _variants(name)[1:]))[:3]
            for query in queries:
                if self._get({"q": query, "viewbox": viewbox, "bounded": 1}):
                    found = True
                    break
            self.cache[key] = found
            self.save()
            return found
        except Exception:
            return None

    def save(self):
        with open(PLACES_CACHE, "w") as f:
            json.dump(self.cache, f, indent=1, ensure_ascii=False)


def score_places(itinerary: str, trip: dict, research: str, maps: MapChecker | None) -> dict:
    places = extract_places(itinerary, trip)
    detail = []
    for name in places:
        if in_research(name, research):
            status = "in_research"
        elif maps is not None:
            ok = maps.exists_near(name, trip["destination"])
            status = "on_map" if ok else ("unverified" if ok is False else "lookup_failed")
        else:
            status = "unverified"
        detail.append({"name": name, "status": status})
    checked = [d for d in detail if d["status"] != "lookup_failed"]
    verified = [d for d in checked if d["status"] in ("in_research", "on_map")]
    return {
        "places_total":      len(checked),
        "places_verified":   len(verified),
        "places_pct":        round(100 * len(verified) / len(checked), 1) if checked else None,
        "unverified_places": [d["name"] for d in checked if d["status"] == "unverified"],
        "places_detail":     detail,
    }


def score(itinerary: str, trip: dict, num_days: int, research: str, maps: MapChecker | None) -> dict:
    return {**score_structure(itinerary, num_days, trip["budget_inr"], trip),
            **score_places(itinerary, trip, research, maps)}
