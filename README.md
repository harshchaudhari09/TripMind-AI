# ✈ TripMind AI — Self-Checking Travel Planner

A multi-agent travel planner built with **LangGraph + Groq + Tavily + OpenStreetMap**.
It researches a trip in parallel, writes a day-by-day plan, and has a **critic agent**
review and fix that plan before you see it.

> This is the part of TripMind that's truly agentic: it judges its own work and decides what to do next.

---

## Architecture

```
User input
    │
    ▼
  init ──┬─► wiki ──────────────┐
         ├─► weather            │   10 research steps run
         ├─► flights (⇄ return) │   at the same time
         ├─► trains / buses     │
         ├─► hotels             │
         ├─► attractions / food │
         └─► ATMs, hospitals,   │
             stations (OSM) ────┘
                    │
                    ▼
          generate_itinerary  (Groq gpt-oss-120b)
                    │
                    ▼
               ┌─ critic ─┐   rule checks (days, budget maths, time order)
               │          │   + AI review (timings, season, contradictions)
     needs fixes          approved / out of retries
   (max 2 rounds)                   │
          │                         ▼
          └──► revise ──► critic   finalize → best draft + critic verdict
```

**Critic agent** ([agents/critic.py](agents/critic.py))
- *Rule checks* (plain Python, always right): every day present once, within budget, times
  within a day go forward (overnight journeys allowed), and nothing planned that's closed that
  month according to a hand-checked list ([config/seasonal_facts.py](config/seasonal_facts.py)).
- *AI review* (`gpt-oss-20b`, a separate Groq rate-limit bucket): blocking problems only —
  impossible travel times, short connections, contradictions, safety. It may not judge
  seasons itself (it used to invent closures, e.g. "Khardung La is closed in November").
- Sends the draft back with notes up to `MAX_REVISIONS` times; the writer is told which
  earlier problems not to reintroduce. Stops early if the same problems repeat. Ties go to
  the newest draft.

**Code does the maths** ([agents/budget.py](agents/budget.py)): the AI writes the budget line
items, and Python recalculates every Total — the AI got the sum wrong in 1 of 4 evaluated trips.

---

## Evaluation

`eval/run_eval.py` runs TripMind on 15 sample trips ([eval/trips.json](eval/trips.json)) — easy
weekends, tight budgets, winter mountain trips, a 1-day and an 8-day trip — and scores each
plan **before** and **after** the critic agent:

| Check | How it's measured (plain Python, no AI) |
|---|---|
| Correct number of days | every `### Day N` heading present exactly once |
| Within budget | budget table exists, adds up, and Total ≤ budget |
| No made-up places | each named stay/restaurant/attraction is in the research data or on OpenStreetMap near the destination |

```bash
venv/bin/python -m eval.run_eval --limit 3   # quick check
venv/bin/python -m eval.run_eval             # all 15 (resumable)
```
Results: `eval/results/REPORT.md`. Research is cached, so re-runs only cost LLM calls.

---

## Deploy (Render, free)

The repo includes a [render.yaml](render.yaml) Blueprint:

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New → Blueprint** → pick the repo.
3. Paste `GROQ_API_KEY` and `TAVILY_API_KEY` when Render asks (never commit them).

**Public-demo protection** (free tiers are small — Groq: 200k tokens/day per model,
Tavily: 1,000 credits/month):
- Per-visitor and site-wide daily plan limits (`PLANS_PER_IP_PER_DAY`, `DAILY_PLAN_LIMIT`)
- Automatic switch to a backup model when the main one hits its limit (`FALLBACK_MODEL`)
- Cheaper "basic" Tavily searches (`TAVILY_SEARCH_DEPTH=basic`)
- Instant example trips that cost no tokens — published with
  `python -m eval.run_eval --export-examples <trip ids>` (only critic-approved plans that
  pass every rule check are exported)
- Trips are capped at 10 days

The free instance sleeps after 15 minutes idle, so the first visit can take ~1 minute.

---

## Budget Tiers (Auto-detected)

| Budget (INR)       | Tier     | Hotel/night         | Flight class |
|--------------------|----------|---------------------|--------------|
| ≤ 15,000           | Budget   | ₹500 – ₹1,500       | Economy      |
| 15,001 – 50,000    | Mid      | ₹2,000 – ₹5,000     | Economy      |
| 50,001 – 1,00,000  | Comfort  | ₹5,000 – ₹10,000    | Economy      |
| 1,00,001 – 3,00,000| Premium  | ₹10,000 – ₹25,000   | Business     |
| > 3,00,000         | Luxury   | ₹25,000 – ₹1,00,000 | Business     |

---

## Quick Start

### 1. Get free API keys

| Service   | URL                         | Free tier       |
|-----------|-----------------------------|-----------------|
| **Groq**  | https://console.groq.com    | 6K tokens/min   |
| **Tavily**| https://app.tavily.com      | 1K searches/mo  |
| **Serper**| https://serper.dev          | 2,500 searches  |

### 2. Setup

```bash
git clone <repo>
cd travel_planner

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run it

```bash
python main.py
```

This starts the server **and opens the web UI in your browser automatically** at `http://localhost:8000` — fill in the trip form, get a generated itinerary, then use the chat box underneath it to ask for changes ("make day 2 cheaper", "add a waterfall visit", etc.), same as before.

Other modes:

```bash
python main.py --cli              # old terminal-only flow
python main.py --no-browser       # server only, don't auto-open a tab
python main.py --port 8080        # run on a different port
uvicorn api:app --reload --port 8000   # dev mode with hot reload
```

### 4. Example API call

```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Goa, India",
    "origin": "Delhi, India",
    "start_date": "2024-12-20",
    "end_date": "2024-12-25",
    "budget_inr": 80000,
    "preferences": ["beaches", "seafood", "nightlife"]
  }'
```

---

## LLM Options

Edit `.env` → `LLM_PROVIDER`:

```env
# Option 1: Groq (recommended, free, fast)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxx
GROQ_MODEL=llama3-8b-8192    # or mixtral-8x7b-32768

# Option 2: Ollama (local, fully offline)
LLM_PROVIDER=ollama
# run: ollama pull phi3

# Option 3: HuggingFace (auto-fallback, no key needed for public models)
LLM_PROVIDER=huggingface
```

---

## Stage 2: LLMOps with ELK

```bash
# Start full stack with Elasticsearch + Kibana
docker compose up

# Init the ES index
python -c "from utils.llmops_logger import setup_elk_index; setup_elk_index()"

# Open Kibana dashboard
open http://localhost:5601
```

**Or use LangSmith (free, easier):**
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxx
LANGCHAIN_PROJECT=travel-planner
```
Then visit https://smith.langchain.com to see traces.

---

## Project Structure

```
travel_planner/
├── main.py              # Launcher — starts the server + opens the UI (or --cli for terminal mode)
├── api.py               # FastAPI server: serves the UI, /plan, /chat
├── static/
│   └── index.html       # Web UI (single file, no build step)
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py      # Budget tiers, API keys, model config
│
├── agents/
│   ├── graph.py         # LangGraph workflow (the core: init → 10 parallel research nodes → itinerary ⇄ critic loop)
│   ├── critic.py        # Critic agent: rule checks + LLM review of each draft
│   └── chat_agent.py    # Conversational refinement of a generated itinerary
│
├── tools/
│   └── search_tools.py  # Tavily tools: flights, trains, hotels, weather
│
├── utils/
│   ├── llm_factory.py   # Groq / Ollama switcher
│   └── llmops_logger.py # ELK + LangSmith observability
│
└── output/              # Itineraries saved here in CLI mode
```

## Troubleshooting "it worked before, then it didn't"

The most common causes, in order of likelihood:

1. **Expired or rate-limited keys.** Groq and Tavily free tiers expire/rotate. Check `http://localhost:8000/health` — if `groq_configured` or `tavily_configured` is `false`, your `.env` isn't being read (wrong filename, wrong folder, or the key value is empty).
2. **A committed `.env` with real keys.** If you ever pushed `.env` to a repo, treat those keys as burned and generate new ones — rotate them at console.groq.com and app.tavily.com. `.env` is now in `.gitignore` and a `.env.example` placeholder is included instead.
3. **Model deprecation.** Groq periodically retires model names. If `/plan` fails with a model-not-found error, check the current model list at console.groq.com and update `GROQ_MODEL` in `.env`.
4. **Stale `__pycache__`.** Delete any `__pycache__` folders if imports behave strangely after editing files.
