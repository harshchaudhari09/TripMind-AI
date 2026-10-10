"""
config/settings.py
"""
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "groq")

# The critic uses a smaller model on purpose: Groq's free tier limits tokens
# per minute *per model*, so the critic doesn't eat the writer's allowance.
CRITIC_MODEL   = os.getenv("CRITIC_MODEL", "openai/gpt-oss-20b")
# How many times the critic may send a draft back for fixes.
MAX_REVISIONS  = int(os.getenv("MAX_REVISIONS", "2"))
# Groq free tier: tokens per minute per model. A single request (prompt + max_tokens)
# larger than this is rejected outright, so prompts are sized to fit.
GROQ_TOKENS_PER_MINUTE = int(os.getenv("GROQ_TOKENS_PER_MINUTE", "8000"))

# ── Public demo protection (free tiers are small: Groq 200k tokens/day per model,
#    Tavily 1,000 credits/month ≈ 16 per trip on "advanced") ─────────────────────
# When the main model hits a limit, the writer switches to this model (own daily allowance).
FALLBACK_MODEL       = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
# "basic" Tavily searches cost 1 credit instead of 2 — roughly halves Tavily use per trip.
TAVILY_SEARCH_DEPTH  = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
# Per-visitor and whole-site limits for the web API (0 = unlimited). Counters live in
# memory, so they reset when the server restarts.
PLANS_PER_IP_PER_DAY = int(os.getenv("PLANS_PER_IP_PER_DAY", "0"))
CHATS_PER_IP_PER_DAY = int(os.getenv("CHATS_PER_IP_PER_DAY", "0"))
DAILY_PLAN_LIMIT     = int(os.getenv("DAILY_PLAN_LIMIT", "0"))


def budget_profile(total_inr: float, num_days: int = 1, num_people: int = 1) -> dict:
    """
    Break down total budget into per-night hotel range,
    flight class, food budget, and search keywords.
    Accounts for number of days and people.
    """
    per_person_per_day = total_inr / max(num_people, 1) / max(num_days, 1)

    if total_inr <= 15_000:
        return {
            "tier": "budget",
            "hotel_min": 500,   "hotel_max": 1_500,
            "flight_class": "economy",
            "food_per_day": 300,
            "hotel_keywords": "budget hostel guesthouse cheap",
            "hotel_search": f"cheap budget hostel guesthouse under 1500 per night",
            "flight_search": "cheapest economy flight",
            "train_search": "sleeper class cheapest train",
        }
    elif total_inr <= 50_000:
        return {
            "tier": "mid",
            "hotel_min": 2_000, "hotel_max": 5_000,
            "flight_class": "economy",
            "food_per_day": 800,
            "hotel_keywords": "3 star hotel mid-range",
            "hotel_search": "3 star hotel 2000 to 5000 per night",
            "flight_search": "economy flight best price",
            "train_search": "3AC train ticket price",
        }
    elif total_inr <= 1_00_000:
        return {
            "tier": "comfort",
            "hotel_min": 5_000, "hotel_max": 10_000,
            "flight_class": "economy",
            "food_per_day": 1_500,
            "hotel_keywords": "4 star hotel boutique comfortable",
            "hotel_search": "4 star hotel 5000 to 10000 per night",
            "flight_search": "economy flight with meals",
            "train_search": "2AC train ticket price",
        }
    elif total_inr <= 3_00_000:
        return {
            "tier": "premium",
            "hotel_min": 10_000, "hotel_max": 25_000,
            "flight_class": "business",
            "food_per_day": 3_000,
            "hotel_keywords": "5 star luxury resort premium",
            "hotel_search": "5 star luxury hotel 10000 to 25000 per night",
            "flight_search": "business class flight",
            "train_search": "1AC first class train ticket",
        }
    else:
        return {
            "tier": "luxury",
            "hotel_min": 25_000, "hotel_max": 1_00_000,
            "flight_class": "business",
            "food_per_day": 5_000,
            "hotel_keywords": "luxury 5 star resort villa private pool",
            "hotel_search": "luxury 5 star resort villa above 25000 per night",
            "flight_search": "first class business class flight",
            "train_search": "1AC luxury train Maharaja Express",
        }
