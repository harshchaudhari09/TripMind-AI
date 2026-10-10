"""
api.py — FastAPI server for TripMind AI

Serves:
  GET  /            → the web UI (static/index.html)
  GET  /health       → health check
  POST /plan        → run the parallel LangGraph pipeline, return an itinerary
  POST /chat        → refine an existing itinerary through conversation
"""
import sys, os, time, logging, threading
from collections import defaultdict, deque
from datetime import date
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional

from config.settings import (GROQ_API_KEY, TAVILY_API_KEY, PLANS_PER_IP_PER_DAY,
                             CHATS_PER_IP_PER_DAY, DAILY_PLAN_LIMIT)

MAX_TRIP_DAYS = 10   # longer trips use too many tokens for the free tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tripmind")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="TripMind AI", description="Agentic AI Travel Planner", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class TripRequest(BaseModel):
    destination:  str         = Field(..., example="Manali, Himachal Pradesh")
    origin:       str         = Field(..., example="Delhi, India")
    start_date:   str         = Field(..., example="2025-12-20")
    end_date:     str         = Field(..., example="2025-12-25")
    budget_inr:   float       = Field(..., example=80000)
    num_people:   int         = Field(default=2, example=2)
    preferences:  List[str]   = Field(default=["trekking", "nature", "local food"])

    # Public-demo guardrails: reject requests that would waste the free API allowance.
    @field_validator("destination", "origin")
    @classmethod
    def _place(cls, v: str) -> str:
        v = v.strip()
        if not 2 <= len(v) <= 80:
            raise ValueError("must be 2–80 characters")
        return v

    @model_validator(mode="after")
    def _trip(self):
        try:
            start, end = date.fromisoformat(self.start_date), date.fromisoformat(self.end_date)
        except ValueError:
            raise ValueError("dates must be YYYY-MM-DD")
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        if (end - start).days + 1 > MAX_TRIP_DAYS:
            raise ValueError(f"trips can be at most {MAX_TRIP_DAYS} days in this demo")
        if not 1 <= self.num_people <= 10:
            raise ValueError("num_people must be between 1 and 10")
        if not 1000 <= self.budget_inr <= 10_000_000:
            raise ValueError("budget_inr must be between ₹1,000 and ₹1,00,00,000")
        self.preferences = [p.strip()[:40] for p in self.preferences if p.strip()][:8]
        return self


class TripResponse(BaseModel):
    itinerary:       str
    budget_tier:     str
    num_days:        int
    destination:     str
    latency_ms:      float
    # Critic agent results
    critic_passed:    bool       = False
    revisions:        int        = 0
    remaining_issues: List[str]  = Field(default_factory=list)


class ChatRequest(BaseModel):
    message:              str
    current_itinerary:    str
    trip_details:         Dict[str, Any]
    conversation_history:  List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply:                 str
    updated_itinerary:     Optional[str] = None
    conversation_history:   List[Dict[str, str]]


# ─────────────────────────────────────────────────────────────────────────────
# Usage limits — the free Groq/Tavily allowances are shared by every visitor, so
# one person shouldn't be able to use them all up. Counts are kept in memory for
# 24 hours (they reset if the server restarts — fine for a demo).
# ─────────────────────────────────────────────────────────────────────────────
_usage: Dict[str, deque] = defaultdict(deque)
_usage_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the visitor's IP is the first entry of X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def _take_quota(kind: str, ip: str, per_ip: int, site_wide: int = 0) -> None:
    now, window = time.time(), 24 * 3600
    keys = [(f"{kind}:{ip}", per_ip, "you"), (f"{kind}:*", site_wide, "site")]
    with _usage_lock:
        for key, limit, who in keys:
            q = _usage[key]
            while q and now - q[0] > window:
                q.popleft()
            if limit and len(q) >= limit:
                if who == "you":
                    detail = (f"You've used your {limit} free {kind}s for today on this demo. "
                              f"Try the example trips, or run TripMind yourself from GitHub.")
                else:
                    detail = (f"The demo has reached today's limit of {limit} {kind}s (free API allowance). "
                              f"Try the example trips, or come back tomorrow.")
                raise HTTPException(status_code=429, detail=detail)
        for key, _, _ in keys:
            _usage[key].append(now)


# ─────────────────────────────────────────────────────────────────────────────
# Startup checks — fail loudly and clearly instead of a cryptic 500 later
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def check_keys():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        logger.warning(
            "Missing %s in .env — /plan and /chat will fail until these are set. "
            "Copy .env.example to .env and add your keys.", ", ".join(missing)
        )
    else:
        logger.info("API keys loaded. TripMind is ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "groq_configured": bool(GROQ_API_KEY),
        "tavily_configured": bool(TAVILY_API_KEY),
        "limits": {"plans_per_visitor_per_day": PLANS_PER_IP_PER_DAY,
                   "site_plans_per_day": DAILY_PLAN_LIMIT, "max_trip_days": MAX_TRIP_DAYS},
    }


@app.post("/plan", response_model=TripResponse)
def plan_trip(req: TripRequest, request: Request):
    if not GROQ_API_KEY or not TAVILY_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Missing GROQ_API_KEY or TAVILY_API_KEY. Add them to .env and restart.",
        )
    _take_quota("trip plan", _client_ip(request), PLANS_PER_IP_PER_DAY, DAILY_PLAN_LIMIT)

    t0 = time.perf_counter()

    initial_state = {
        "destination":  req.destination,
        "origin":       req.origin,
        "start_date":   req.start_date,
        "end_date":     req.end_date,
        "budget_inr":   req.budget_inr,
        "num_people":   req.num_people,
        "preferences":  req.preferences,
        "budget_tier": "", "num_days": 0,
        "flight_data": "", "train_data": "", "bus_data": "",
        "hotel_data": "", "weather_data": "", "attraction_data": "",
        "food_data": "", "wiki_data": "",
        "petrol_data": "", "atm_data": "", "hospital_data": "",
        "bus_station_data": "", "train_station_data": "",
        "itinerary": "", "messages": [],
    }

    try:
        # imported lazily so a missing/invalid key fails via the check above,
        # not at module import time
        from agents.graph import travel_graph
        final = travel_graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Trip planning failed")
        raise HTTPException(status_code=500, detail=str(e))

    return TripResponse(
        itinerary=final["itinerary"],
        budget_tier=final["budget_tier"],
        num_days=final["num_days"],
        destination=req.destination,
        latency_ms=(time.perf_counter() - t0) * 1000,
        critic_passed=final.get("critic_passed", False),
        revisions=final.get("revision_count", 0),
        remaining_issues=[] if final.get("critic_passed") else final.get("critique", []),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Missing GROQ_API_KEY. Add it to .env and restart.")
    _take_quota("chat message", _client_ip(request), CHATS_PER_IP_PER_DAY)

    try:
        from agents.chat_agent import refine_itinerary
        reply, updated_history = refine_itinerary(
            user_message=req.message,
            current_itinerary=req.current_itinerary,
            trip_details=req.trip_details,
            conversation_history=req.conversation_history,
        )
    except Exception as e:
        logger.exception("Chat refinement failed")
        raise HTTPException(status_code=500, detail=str(e))

    # chat_agent returns the full updated itinerary in `reply` when it rewrote one
    updated_itinerary = reply if ("## Day" in reply or "**Day" in reply) else None

    return ChatResponse(
        reply=reply,
        updated_itinerary=updated_itinerary,
        conversation_history=updated_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Static UI — mounted last so /health, /plan, /chat above take priority
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
