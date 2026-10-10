"""
config/seasonal_facts.py

A small, hand-checked list of seasonal closures (and of places that are open all year).

Why this exists: the critic's LLM kept inventing closures ("Khardung La is closed in
November" — it isn't). Season problems are now checked against THIS list by plain code,
and the LLM is told to use only these facts, never its own guesses.

Months are approximate: opening/closing dates shift by a few weeks each year with snow.
Add new places here as you find them — each entry needs:
  name         display name
  match        lowercase words that identify it in an itinerary
  near         lowercase words in the destination that make this fact relevant
  closed       month numbers (1 = Jan) when it's closed / not possible; [] = open all year
  note         one line shown to the writer and the critic
"""
from __future__ import annotations

HIMACHAL = ["manali", "kullu", "solang", "himachal", "lahaul", "spiti", "kaza", "keylong", "sissu"]
LADAKH   = ["leh", "ladakh", "nubra", "pangong", "zanskar", "kargil", "hunder", "diskit"]
UTTARAKHAND = ["rishikesh", "haridwar", "kedarnath", "badrinath", "auli", "joshimath", "chamoli", "uttarakhand"]

SEASONAL_FACTS = [
    # ── Himachal ────────────────────────────────────────────────────────────
    {"name": "Rohtang Pass", "match": ["rohtang"], "near": HIMACHAL,
     "closed": [11, 12, 1, 2, 3, 4],
     "note": "closed to tourists roughly mid-Nov to Apr; Atal Tunnel to Sissu usually stays open"},
    {"name": "Kunzum Pass (Manali–Kaza road)", "match": ["kunzum"], "near": HIMACHAL,
     "closed": [11, 12, 1, 2, 3, 4, 5], "note": "snowbound roughly Nov to May"},
    {"name": "Hampta Pass trek", "match": ["hampta"], "near": HIMACHAL,
     "closed": [11, 12, 1, 2, 3, 4, 5], "note": "trek season is roughly Jun to Oct"},
    {"name": "Manali–Leh highway (Baralacha La, Tanglang La)", "match": ["baralacha", "tanglang"],
     "near": HIMACHAL + LADAKH, "closed": [11, 12, 1, 2, 3, 4],
     "note": "closed roughly Nov to Apr"},
    {"name": "Solang Valley", "match": [], "near": HIMACHAL, "closed": [],
     "note": "open all year; snow activities in winter (ropeway may pause in bad weather)"},
    {"name": "Atal Tunnel", "match": [], "near": HIMACHAL, "closed": [],
     "note": "open all year; may shut briefly after heavy snowfall"},
    # ── Ladakh ──────────────────────────────────────────────────────────────
    {"name": "Khardung La / Nubra Valley", "match": [], "near": LADAKH, "closed": [],
     "note": "kept open all year by BRO; may shut briefly after snowfall"},
    {"name": "Chang La / Pangong Lake", "match": [], "near": LADAKH, "closed": [],
     "note": "reachable all year; the lake freezes roughly Jan to Mar"},
    {"name": "Markha Valley trek", "match": ["markha"], "near": LADAKH,
     "closed": [11, 12, 1, 2, 3, 4, 5], "note": "trek season is roughly Jun to Sep"},
    {"name": "Chadar trek (frozen Zanskar)", "match": ["chadar"], "near": LADAKH,
     "closed": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "note": "only possible roughly Jan to Feb"},
    {"name": "Zanskar river rafting", "match": ["zanskar rafting", "rafting on the zanskar", "rafting on zanskar"],
     "near": LADAKH, "closed": [10, 11, 12, 1, 2, 3, 4, 5], "note": "summer only, roughly Jun to Sep"},
    {"name": "Zoji La (Srinagar–Leh road)", "match": ["zoji"], "near": LADAKH,
     "closed": [1, 2, 3], "note": "usually closed in mid-winter"},
    # ── Uttarakhand ─────────────────────────────────────────────────────────
    {"name": "River rafting in Rishikesh", "match": ["rafting"], "near": ["rishikesh"],
     "closed": [7, 8], "note": "suspended during the monsoon, roughly Jul to mid-Sep"},
    {"name": "Kedarnath Temple", "match": ["kedarnath"], "near": UTTARAKHAND,
     "closed": [11, 12, 1, 2, 3, 4], "note": "closed for winter, roughly Nov to Apr"},
    {"name": "Badrinath Temple", "match": ["badrinath"], "near": UTTARAKHAND,
     "closed": [12, 1, 2, 3, 4], "note": "closes around mid-Nov, reopens around late Apr/May"},
    {"name": "Valley of Flowers", "match": ["valley of flowers"], "near": UTTARAKHAND,
     "closed": [11, 12, 1, 2, 3, 4, 5], "note": "open roughly Jun to Oct"},
    {"name": "Hemkund Sahib", "match": ["hemkund"], "near": UTTARAKHAND,
     "closed": [11, 12, 1, 2, 3, 4], "note": "open roughly late May to Oct"},
    # ── Goa ─────────────────────────────────────────────────────────────────
    {"name": "Sea water sports in Goa", "match": ["parasailing", "jet ski", "banana ride", "water sports"],
     "near": ["goa"], "closed": [6, 7, 8, 9], "note": "banned during the monsoon, roughly Jun to Sep"},
]

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def facts_for(destination: str) -> list[dict]:
    dest = destination.lower()
    return [f for f in SEASONAL_FACTS if any(word in dest for word in f["near"])]


def describe(facts: list[dict], months: list[int]) -> str:
    """One line per fact, e.g. '- Rohtang Pass: CLOSED in Nov — closed to tourists ...'."""
    lines = []
    for f in facts:
        shut = [m for m in months if m in f["closed"]]
        status = f"CLOSED in {', '.join(_MONTHS[m - 1] for m in shut)}" if shut else "open in these months"
        lines.append(f"- {f['name']}: {status} — {f['note']}")
    return "\n".join(lines)
