# ✈ TripMind AI — a travel planner that checks its own work

TripMind plans a complete day-by-day trip in India — transport, hotels, food, weather,
budget and nearby facilities — from a few inputs. What makes it different: **a critic agent
reviews every plan before you see it, and sends it back to be fixed** when something is wrong.

> This is the part of TripMind that's truly agentic: it judges its own work and decides what to do next.

[![tests](https://github.com/harshchaudhari09/TripMind-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/harshchaudhari09/TripMind-AI/actions/workflows/tests.yml)

🔗 **Live demo: [tripmind-ai-y4g3.onrender.com](https://tripmind-ai-y4g3.onrender.com/)** · 💻 Built with LangGraph, Groq, Tavily, OpenStreetMap and FastAPI

_Free hosting: the first visit after a quiet spell takes ~1 minute to wake up. Click an example trip to see a plan instantly; each visitor can plan 2 trips a day._

---

## What it does

You enter where you're going, from where, your dates, budget, group size and interests.
TripMind then:

1. **Researches in parallel** — 10 lookups run at the same time: flights (there and back),
   trains, buses, hotels, attractions, local food, live weather, Wikipedia, and nearby ATMs,
   hospitals, petrol pumps and stations from OpenStreetMap.
2. **Writes a plan** — a summary, transport, 3 stays, a timed day-by-day plan, food to try,
   a budget table, packing tips and local tips.
3. **Checks the plan** — a critic agent looks for mistakes and, if it finds any, sends the
   plan back with notes. Up to 2 rounds of fixes.
4. **Shows the result** with the critic's verdict: ✅ approved, or ⚠️ what to double-check.

You can then ask for changes in a chat box ("make day 2 cheaper", "add a waterfall") —
and every edited plan goes through the same critic before it's shown.

**🎤 Or just talk to it.** Tap the mic and say *"3 days in Goa from Mumbai next weekend, me and my
wife, budget 30k"* — TripMind fills in the form and starts planning. With a plan open, say
*"make day two cheaper"* and it edits the plan (critic-checked) and reads the change back to you.
Speech-to-text runs in the browser (Chrome, Edge, Safari); a small model works out whether you
want a new trip or a change, and code validates every date, budget and group size it extracts.

---

## Why a critic agent?

Language models write convincing plans that are often quietly wrong. In testing, first
drafts included:

| What the first draft said | What was wrong | Caught by |
|---|---|---|
| Budget total **₹28,850** of ₹30,000 | The items actually added up to **₹36,950** — over budget | Code |
| Day trip to **Rohtang Pass** in November | Closed to tourists Nov–Apr | Verified seasonal list |
| **Ganga Aarti at 1:30 AM** | It happens at sunset | Code (activities after midnight) |
| Overnight bus arrives 1:35 AM, then a full day of sightseeing under the **same** day | The next day's plans were filed under the travel day | Code (overnight journeys) |
| Hotel check-out, then 3 hours of driving around South Goa, back by 7:30 PM | Not enough time to cover the distances | AI reviewer |

Fixing these needs two kinds of checking, so the critic has two layers:

- **Rule checks in plain Python** — anything that can be *counted*: every day present,
  budget within limit, times in order, overnight journeys, enough time before a flight or
  train, and nothing planned that's closed that month (from a hand-checked list).
- **An AI reviewer** — judgement calls code can't make: realistic travel times, connections,
  contradictions between sections, and safety (e.g. going above 3,500 m too soon).

---

## How it works

```
                   ┌─► flights (there & back) ─┐
                   ├─► trains / buses          │
 trip details ─► init ─► hotels                ├─► writer agent ─► critic agent ──► final plan
                   ├─► attractions / food      │        ▲                │   (✅ or ⚠️ + notes)
                   ├─► weather · Wikipedia     │        │   needs fixes  │
                   └─► ATMs · hospitals ·  ────┘        └────────────────┘
                       stations (OpenStreetMap)          max 2 rounds; stops early
                                                         if the same problem repeats
```

- **LangGraph** runs the 10 research steps in parallel and wires the writer ⇄ critic loop.
- **Writer:** Groq `gpt-oss-120b`. **AI reviewer:** `gpt-oss-20b` (a separate rate-limit
  allowance, so checking doesn't slow down writing).
- **Code does the arithmetic:** the writer lists budget items; Python calculates every total.
- If a fix makes things worse, TripMind keeps the best draft it has seen.

---

## Results

TripMind is tested on **15 sample trips** ([eval/trips.json](eval/trips.json)) — easy weekends,
tight budgets (₹10,000), winter trips to Manali and Leh, a 1-day and an 8-day trip. Every plan
is graded automatically by plain Python (no AI), **before** and **after** the critic:

| Check | First draft | After the critic |
|---|---|---|
| Correct number of days | 100% | 100% |
| Within budget | 93% | **100%** |
| Times in order (incl. overnight journeys, nothing after midnight) | 93% | **100%** |
| No places closed that season | 100% | 100% |
| Named places verified (in research data or on OpenStreetMap)* | 84% | 85% |
| **Passes every check** | 87% | **100%** |

- The critic approved **15 of 15** plans, needing only **0.3 fix rounds** on average;
  a trip takes **~48 seconds**.
- An earlier version (before moving maths and season checks into code) passed every check
  on only **50%** of first drafts — see [eval/results/REPORT_before_3b.md](eval/results/REPORT_before_3b.md).

\* A lower bound: many "unverified" names are real places that OpenStreetMap lists under another
name or that lie just outside the search area (e.g. Diskit Monastery, Hampi's Royal Enclosure).
Full details: [eval/results/REPORT.md](eval/results/REPORT.md).

**What the numbers do and don't show:** the day, budget, time and season checks are the same
rules the critic enforces, so "100% after the critic" mainly proves the loop reliably catches
and fixes what it's built to catch. The place check is independent of the critic.

### Automated tests

53 offline tests ([tests/](tests/)) cover every rule check (using the real mistakes above as
test cases), the budget maths, the writer ⇄ critic loop with fake AI models, chat edits, the voice assistant, the
public-demo limits, and that every published example trip passes all checks. They run on every
push via GitHub Actions — no API keys needed:

```bash
pip install -r requirements-dev.txt && python -m pytest -q
```

---

## What I learned building it

- **Use code for anything that can be counted.** The model kept getting sums, dates and
  time gaps wrong; plain Python never does. The AI is reserved for judgement.
- **AI critics hallucinate too.** The reviewer once insisted Khardung La is "closed in
  November" (it's open all year). Season checks now use a verified list instead of the
  model's memory.
- **Measure, don't guess.** The evaluation found bugs no single test run showed — including
  a tie-break that silently discarded the fixed plan.
- **Search costs more than the AI.** Per trip, web search (~$0.07) costs ~14× the language
  model (~$0.005) — caching research is the biggest saving.

---

## Tech stack

| Part | Tools |
|---|---|
| Agent workflow | LangGraph, LangChain |
| Language models | Groq — `gpt-oss-120b` (writer), `gpt-oss-20b` (reviewer, backup) |
| Research | Tavily search, OpenStreetMap (Overpass + Nominatim), Open-Meteo, Wikipedia |
| Backend & UI | FastAPI, single-page HTML/JS |
| Hosting | Render (free tier), with per-visitor limits and instant example trips |

---

## Project structure

```
agents/
  graph.py         LangGraph workflow: parallel research → writer ⇄ critic loop
  critic.py        Critic agent: rule checks + AI review
  budget.py        Recalculates budget totals in code
  chat_agent.py    Chat-based changes to a finished plan (only changed parts are merged in)
  voice_intent.py  Voice assistant: understands spoken requests (new trip vs. edit)
tools/             Tavily, OpenStreetMap, weather and Wikipedia tools
config/
  settings.py      Models, limits, budget tiers
  seasonal_facts.py  Hand-checked seasonal closures (Rohtang, Khardung La, …)
eval/              15 test trips, automatic grading, reports
tests/             53 offline tests (pytest)
static/index.html  Web interface
api.py · main.py   Web server and launcher
```

---

## Run it yourself

You need free API keys from [Groq](https://console.groq.com) and [Tavily](https://app.tavily.com).

```bash
git clone https://github.com/harshchaudhari09/TripMind-AI.git
cd TripMind-AI
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your two API keys
python main.py                # opens the web app at http://127.0.0.1:8000
```

Other options: `python main.py --cli` (terminal version) · `python -m eval.run_eval --limit 3`
(grade 3 sample trips). Deploying your own copy: [render.yaml](render.yaml) is a ready-made
Render Blueprint — just add the two keys.

---

Built by **Harsh Chaudhari**.
