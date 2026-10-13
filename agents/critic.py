"""
agents/critic.py

The critic agent reviews a draft itinerary before the user sees it.

It works in two layers:
1. Rule checks (plain Python) — things that can be counted exactly:
   - every day is present, once, in order
   - the budget table adds up and stays within the user's budget
   - times inside each day go forward (no 2 PM before 11 AM)
2. LLM review — judgement calls code can't make:
   - is the timing realistic (travel time, flight connections)?
   - does anything contradict itself (arrival date vs Day 1)?
   - is anything impossible for the season (closed passes, treks)?

The graph uses the result to decide: approve the draft, or send it back to
the writer with the list of problems.
"""
from __future__ import annotations
import json
import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from datetime import date, datetime

from config.settings import CRITIC_MODEL
from config.seasonal_facts import facts_for, describe
from utils.llm_factory import get_llm


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: rule checks
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    # LLMs often use narrow/non-breaking spaces ("6:30 AM") and
    # non-breaking hyphens ("5‑Day"); turn them into plain characters.
    return (text.replace(" ", " ").replace(" ", " ")
                .replace("‑", "-").replace("–", "-"))


def _split_days(text: str) -> dict:
    """Return {day_number: section_text} for every '### Day N' heading."""
    matches = list(re.finditer(r"^#{2,4}\s*Day\s+(\d+)\b.*$", text, re.MULTILINE))
    days = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # A day ends at the next day heading or the next '## ' section.
        section = text[m.end():end]
        next_section = re.search(r"^##\s", section, re.MULTILINE)
        if next_section:
            section = section[:next_section.start()]
        days.setdefault(int(m.group(1)), []).append(section)
    return days


def check_days(text: str, num_days: int) -> List[str]:
    days = _split_days(text)
    issues = []
    missing = [d for d in range(1, num_days + 1) if d not in days]
    extra   = [d for d in days if d > num_days]
    repeated = [d for d, sections in days.items() if len(sections) > 1]
    if missing:
        issues.append(f"Missing day(s): {', '.join(f'Day {d}' for d in missing)}. "
                      f"The trip has {num_days} days and each needs its own '### Day N' section.")
    if extra:
        issues.append(f"Extra day(s) beyond the trip length: {', '.join(f'Day {d}' for d in extra)}.")
    if repeated:
        issues.append(f"Day heading used more than once: {', '.join(f'Day {d}' for d in repeated)}.")
    return issues


def _money(cell: str) -> float | None:
    """'**₹44,800**' → 44800. For a range like '₹2,000–₹3,000' take the higher end."""
    nums = re.findall(r"₹\s*([\d,]+(?:\.\d+)?)", cell)
    if not nums:
        nums = re.findall(r"\b(\d[\d,]{2,})\b", cell)
    values = [float(n.replace(",", "")) for n in nums if n.replace(",", "")]
    return max(values) if values else None


def parse_budget_table(text: str) -> tuple[list, float | None]:
    """Return ([(item, amount), ...], total) from the '| Item | Cost |' table."""
    items, total = [], None
    for line in text.splitlines():
        if not line.strip().startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower() in ("item", "**item**"):
            continue
        amount = _money(cells[-1])
        if amount is None:
            continue
        if "total" in cells[0].lower():
            total = amount
        else:
            items.append((cells[0].replace("*", ""), amount))
    return items, total


def check_budget(text: str, budget_inr: float) -> List[str]:
    items, total = parse_budget_table(text)
    if not items or total is None:
        return ["Budget table is missing or has no Total row. Add the '| Item | Cost |' table with a **Total** row."]
    issues = []
    line_sum = sum(a for _, a in items)
    if abs(line_sum - total) > max(500, 0.02 * total):
        issues.append(f"Budget table doesn't add up: the line items sum to ₹{line_sum:,.0f} "
                      f"but the Total says ₹{total:,.0f}. Fix the numbers so they match.")
    worst_total = max(total, line_sum)
    if worst_total > budget_inr:
        issues.append(f"Over budget: total is ₹{worst_total:,.0f} but the budget is ₹{budget_inr:,.0f}. "
                      f"Cut ₹{worst_total - budget_inr:,.0f} (cheaper stay, transport or activities).")
    return issues


def _minutes(h: str, m: str, ampm: str) -> int:
    hour = int(h) % 12 + (12 if ampm.upper() == "PM" else 0)
    return hour * 60 + int(m)


_DEPARTURE = re.compile(r"\b(board|boards|depart|departs|departure|bus|train|flight|overnight|sleeper|express)\b", re.I)


def check_time_order(text: str) -> List[str]:
    issues = []
    for day, sections in sorted(_split_days(text).items()):
        stops = [(m.groups(), line) for line in sections[0].splitlines()
                 for m in [re.search(r"\*\*\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])\s*\*\*", line)] if m]
        times = [s[0] for s in stops]
        mins = [_minutes(*t) for t in times]
        for i in range(1, len(mins)):
            if mins[i] >= mins[i - 1]:
                continue
            p, t = (f"{x[0]}:{x[1]} {x[2].upper()}" for x in (times[i - 1], times[i]))
            departs = _DEPARTURE.search(stops[i - 1][1])
            if departs and mins[i - 1] >= 12 * 60 and mins[i] < 12 * 60:
                # Afternoon/evening → early morning = an overnight bus/train (very common in
                # India). Fine if the arrival is the day's last stop; but the morning after
                # belongs to the next day, so any stops after the arrival are misplaced.
                later = len(times) - i - 1
                if later:
                    issues.append(f"Day {day}: an overnight journey leaves at {p} and arrives at {t} the next "
                                  f"morning, but {later} more stop(s) follow under Day {day}. Move the arrival "
                                  f"and everything after it to Day {day + 1}; Day {day} should end with the departure.")
            elif mins[i - 1] >= 22 * 60 and mins[i] < 4 * 60:
                issues.append(f"Day {day}: stops continue past midnight ({p} → {t}). Activities after "
                              f"midnight are unrealistic: end the day by about 11 PM and move the rest to "
                              f"a sensible time (check when the activity really happens).")
            else:
                issues.append(f"Day {day}: times go backwards ({p} is followed by {t}). "
                              f"Put the stops in time order.")
            break
    return issues


_FLIGHT_DEPARTS = re.compile(r"\b(flight|plane)\b.*\b(departs?|departure|takes? off|boarding)\b|\b(departs?|take off)\b.*\bflight\b", re.I)
_TRAIN_DEPARTS  = re.compile(r"\b(train|express|shatabdi|rajdhani)\b.*\b(departs?|departure|leaves)\b|\bboard\b.*\btrain\b", re.I)
_HEAD_TO = re.compile(r"\b(leave|leaves|head|heads|taxi|cab|auto|drive|transfer|check.?out)\b", re.I)


def check_departure_buffers(text: str) -> List[str]:
    """
    Leaving for the airport/station too close to departure. The LLM critic is unreliable at
    this arithmetic (it missed a 45-minute gap and once complained about a 6-hour one), so
    it's done here: ≥ 2 h before a flight, ≥ 30 min before a train, counted from the stop
    where you head to the airport/station.
    """
    issues = []
    for day, sections in sorted(_split_days(text).items()):
        stops = [(_minutes(*m.groups()), line) for line in sections[0].splitlines()
                 for m in [re.search(r"\*\*\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])\s*\*\*", line)] if m]
        for (t1, l1), (t2, l2) in zip(stops, stops[1:]):
            gap = t2 - t1
            if gap < 0:
                continue
            if _FLIGHT_DEPARTS.search(l2) and "airport" in l1.lower() and _HEAD_TO.search(l1) and gap < 120:
                issues.append(f"Day {day}: only {gap} min between heading to the airport and the flight "
                              f"departing. Leave for the airport at least 2.5 hours before a domestic flight.")
            elif _TRAIN_DEPARTS.search(l2) and "station" in l1.lower() and _HEAD_TO.search(l1) and gap < 30:
                issues.append(f"Day {day}: only {gap} min between heading to the station and the train "
                              f"departing. Leave at least 45 minutes before the train.")
    return issues


def trip_months(start_date: str, end_date: str) -> List[int]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date, "%Y-%m-%d").date()
    months, d = [], date(start.year, start.month, 1)
    while d <= end:
        months.append(d.month)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return months


_NEGATION = re.compile(r"\b(closed|skip|skipping|instead|not|no|avoid|shut|alternative)\b", re.I)


def check_season(text: str, trip: dict) -> List[str]:
    """Flag places from the verified seasonal list that are planned while they're closed."""
    months = trip_months(trip["start_date"], trip["end_date"])
    issues = []
    for fact in facts_for(trip["destination"]):
        shut = [m for m in months if m in fact["closed"]]
        if not shut or not fact["match"]:
            continue
        for day, sections in sorted(_split_days(text).items()):
            # A bullet that mentions the place without saying it's closed/skipped = it's planned.
            planned = [ln for ln in sections[0].splitlines()
                       if any(w in ln.lower() for w in fact["match"]) and not _NEGATION.search(ln)]
            if planned:
                issues.append(f"Day {day}: {fact['name']} is planned, but it is not possible in this "
                              f"month ({fact['note']}). Replace it with something open in that month.")
                break
    return issues


def rule_checks(itinerary: str, num_days: int, budget_inr: float, trip: dict | None = None) -> List[str]:
    text = _normalise(itinerary)
    issues = (check_days(text, num_days) + check_budget(text, budget_inr)
              + check_time_order(text) + check_departure_buffers(text))
    return issues + (check_season(text, trip) if trip else [])


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: LLM review
# ─────────────────────────────────────────────────────────────────────────────
CRITIC_SYSTEM_PROMPT = """You are an experienced Indian travel editor reviewing a draft itinerary
before a real traveller follows it. Your job is to catch BLOCKING problems only — things that
make the plan impossible, unsafe, or self-contradictory. Go through it DAY BY DAY and check:

1. Travel time: a day trip whose round-trip travel is more than ~6 hours, or times that leave
   clearly too little time to cover the distance (mountain roads average 25-30 km/h).
2. Connections: flight connections under 2 hours; arriving before you could have got there;
   activities after the return transport has left.

Before reporting ANY timing problem, write down the actual time available and the time needed,
and only report it if available < needed. Example: leaving the hotel at 1:00 PM for a 7:00 PM
flight gives 6 hours — far more than needed, so it is NOT a problem. Never suggest leaving
later to "allow more time".
3. Contradictions: dates/times in "Getting there & back" that don't match the day plan; the
   return journey not on the last day.
4. Safety: long drives over high passes after dark; going above 3,500 m within the first
   2 days after arriving from the plains.

SEASONS: do NOT report seasonal closures at all — code already checks them against a verified
list. Treat every place as open unless VERIFIED SEASON FACTS marks it CLOSED. Never write that
something "is closed", "may be closed" or "is likely closed".

Do NOT report (these are not blocking): style or wording, prices, budget arithmetic, hotel
check-in times, weather contingencies or "consider a backup", travel estimates that are off by
less than ~30 minutes, general altitude advice below 3,500 m, or suggestions to make a good plan
nicer. An overnight bus/train that departs in the evening and arrives next morning is fine.

Each issue must name the day, say what is wrong AND how to fix it, in one sentence.
Any replacement you suggest must not be marked CLOSED in VERIFIED SEASON FACTS, and must not be
something listed under "ALREADY FLAGGED".

Reply with JSON only, no other text:
{"approved": true or false, "issues": ["...", "..."]}
If nothing is blocking, reply {"approved": true, "issues": []} — approving a good plan is the
correct outcome. Give at most 4 issues, most serious first."""


def llm_review(itinerary: str, trip: dict, earlier_issues: List[str] | None = None) -> List[str]:
    # 1500 tokens leaves room for the model's hidden reasoning AND the JSON answer.
    llm = get_llm(temperature=0, max_tokens=1500, reasoning_effort="low", model=CRITIC_MODEL)
    facts = (f"Trip: {trip['origin']} → {trip['destination']}, {trip['start_date']} to {trip['end_date']} "
             f"({trip['num_days']} days, {trip['num_days'] - 1} nights), {trip['num_people']} traveller(s), "
             f"budget ₹{trip['budget_inr']:,.0f}.")
    season = describe(facts_for(trip["destination"]), trip_months(trip["start_date"], trip["end_date"]))
    season_block = f"\n\nVERIFIED SEASON FACTS for this trip:\n{season}" if season else ""
    flagged = ("\n\nALREADY FLAGGED in earlier drafts (never suggest these as replacements):\n"
               + "\n".join(f"- {i}" for i in earlier_issues)) if earlier_issues else ""
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"{facts}{season_block}{flagged}\n\n--- DRAFT ITINERARY ---\n{itinerary}"),
    ]
    # Small models occasionally reply without valid JSON, so allow one retry.
    for attempt in (1, 2):
        try:
            response = llm.invoke(messages, config={"run_name": "critic: AI review"})
            raw = response.content if hasattr(response, "content") else str(response)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                reason = response.response_metadata.get("finish_reason", "?")
                raise ValueError(f"no JSON in reply (finish_reason={reason}, reply={raw[:80]!r})")
            data = json.loads(match.group(0))
            return [str(i) for i in data.get("issues", [])][:4] if not data.get("approved", False) else []
        except Exception as e:
            print(f"  [critic] LLM review attempt {attempt} failed: {e}")
    # A broken review must never block the user's itinerary; rule checks still apply.
    return []


def review(itinerary: str, trip: dict, earlier_issues: List[str] | None = None) -> dict:
    """Run both layers. Returns {"passed": bool, "rule_issues": [...], "llm_issues": [...]}."""
    rule_issues = rule_checks(itinerary, trip["num_days"], trip["budget_inr"], trip)
    llm_issues  = llm_review(itinerary, trip, earlier_issues)
    return {
        "passed":      not rule_issues and not llm_issues,
        "rule_issues": rule_issues,
        "llm_issues":  llm_issues,
    }
