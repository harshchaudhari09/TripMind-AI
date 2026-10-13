"""
eval/run_eval.py — run TripMind on sample trips and score the results.

    venv/bin/python -m eval.run_eval                  # all trips in eval/trips.json
    venv/bin/python -m eval.run_eval --limit 3        # just the first 3 (a quick check)
    venv/bin/python -m eval.run_eval --only goa-beach,agra-daytrip
    venv/bin/python -m eval.run_eval --report         # rebuild the report from saved results
    venv/bin/python -m eval.run_eval --rerun          # re-plan trips that already have results

Each trip is scored twice — the FIRST draft (before the critic) and the FINAL plan (after the
critic loop) — so the report shows how much the critic agent helps.

Resumable: every finished trip is saved to eval/results/<id>.json and skipped next time, so if
Groq's daily limit stops a run you can simply run the same command again tomorrow.
Research (Tavily/OSM/Wikipedia) is cached in eval/cache/research/, so re-runs cost only LLM calls.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import scoring

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
TRIPS_FILE   = os.path.join(EVAL_DIR, "trips.json")
RESULTS_DIR  = os.path.join(EVAL_DIR, "results")
RESEARCH_DIR = os.path.join(EVAL_DIR, "cache", "research")
REPORT_FILE  = os.path.join(RESULTS_DIR, "REPORT.md")
EXAMPLES_FILE = os.path.join(os.path.dirname(EVAL_DIR), "static", "examples.json")

# Which state keys each research node fills — used to replay cached research.
NODE_KEYS = {
    "wiki": ["wiki_data"], "weather": ["weather_data"], "flights": ["flight_data"],
    "trains": ["train_data"], "buses": ["bus_data"], "hotels": ["hotel_data"],
    "attractions": ["attraction_data"], "food": ["food_data"],
    "local_facilities": ["petrol_data", "atm_data", "hospital_data"],
    "transport_stations": ["bus_station_data", "train_station_data"],
}


class DailyLimitReached(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Running one trip
# ─────────────────────────────────────────────────────────────────────────────
def plan_trip(trip: dict) -> tuple[dict, float, bool]:
    """Run the full graph. Returns (final_state, seconds, used_cached_research)."""
    from agents.graph import build_graph, RESEARCH_NODES

    cache_path = os.path.join(RESEARCH_DIR, f"{trip['id']}.json")
    cached = json.load(open(cache_path)) if os.path.exists(cache_path) else None
    collected = {}
    if cached:
        nodes = {name: (lambda keys: lambda state: {k: cached.get(k, "") for k in keys})(keys)
                 for name, keys in NODE_KEYS.items()}
    else:
        # Record each research result as it arrives, so it's saved even if a later
        # step (e.g. an LLM call) fails — research costs Tavily credits, don't lose it.
        def recording(fn):
            def node(state):
                result = fn(state)
                collected.update(result)
                return result
            return node
        nodes = {name: recording(fn) for name, fn in RESEARCH_NODES.items()}

    inputs = {k: trip[k] for k in ("destination", "origin", "start_date", "end_date",
                                   "budget_inr", "num_people", "preferences")}
    t0 = time.time()
    try:
        from utils.llmops_logger import trace_config
        config = trace_config(inputs, source="eval")
        config["tags"].append(trip["id"])
        final = build_graph(nodes).invoke(inputs, config=config)
    finally:
        expected = {k for keys in NODE_KEYS.values() for k in keys}
        if not cached and expected <= collected.keys():
            os.makedirs(RESEARCH_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(collected, f)
    return final, time.time() - t0, bool(cached)


def _is_daily_limit(msg: str) -> bool:
    # Groq says e.g. "on tokens per day (TPD): Limit 200000 ...". Match the phrase, not
    # the letters "tpd" — those can appear by chance inside an organisation id.
    return bool(re.search(r"per day|\((TPD|RPD)\)", msg, re.I))


def evaluate_trip(trip: dict, maps: scoring.MapChecker | None) -> dict:
    for attempt in (1, 2):
        try:
            final, seconds, cached = plan_trip(trip)
            break
        except Exception as e:
            msg = str(e)
            if _is_daily_limit(msg):
                raise DailyLimitReached(msg) from e
            if attempt == 1 and ("429" in msg or "rate limit" in msg.lower()):
                # Per-minute limit still busy after the client's own retries: let the
                # minute window reset, then try the trip again (research is cached now).
                print("  ⏳ Groq per-minute limit — waiting 60s, then retrying this trip...")
                time.sleep(60)
                continue
            return {"id": trip["id"], "error": msg[:500], "finished_at": datetime.now().isoformat()}

    research = scoring.research_text(final)
    return {
        "id":            trip["id"],
        "trip":          trip,
        "num_days":      final["num_days"],
        "seconds":       round(seconds, 1),
        "cached_research": cached,
        "critic_passed": final.get("critic_passed", False),
        "fix_rounds":    final.get("revision_count", 0),
        "remaining_issues": [] if final.get("critic_passed") else final.get("critique", []),
        "first_draft":   scoring.score(final["first_draft"], trip, final["num_days"], research, maps),
        "final":         scoring.score(final["itinerary"],   trip, final["num_days"], research, maps),
        "itinerary":     final["itinerary"],
        "first_draft_text": final["first_draft"],
        "finished_at":   datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def _pct(values) -> str:
    values = [v for v in values if v is not None]
    return f"{100 * sum(values) / len(values):.0f}%" if values else "–"


def _avg(values, fmt="{:.0f}") -> str:
    values = [v for v in values if v is not None]
    return fmt.format(sum(values) / len(values)) if values else "–"


def build_report(results: list[dict]) -> str:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    lines = [
        "# TripMind evaluation report",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · {len(ok)} trip(s) scored"
        + (f" · {len(failed)} failed" if failed else "") + "_",
        "",
        "## Summary: first draft vs. after the critic agent",
        "",
        "| Check | First draft | Final plan |",
        "|---|---|---|",
    ]
    for label, key in [("Correct number of days", "days_ok"),
                       ("Within budget", "within_budget"),
                       ("Budget table adds up", "adds_up"),
                       ("Times in order", "time_order_ok"),
                       ("No seasonally closed places", "season_ok")]:
        lines.append(f"| {label} | {_pct(r['first_draft'].get(key) for r in ok)} | {_pct(r['final'].get(key) for r in ok)} |")
    lines.append(f"| Places verified (avg) | {_avg((r['first_draft']['places_pct'] for r in ok), '{:.0f}%')} "
                 f"| {_avg((r['final']['places_pct'] for r in ok), '{:.0f}%')} |")
    all_pass = lambda s: (s["days_ok"] and s["within_budget"] and s["adds_up"]
                          and s["time_order_ok"] and s.get("season_ok", True))
    lines.append(f"| **Passes all checks above** | **{_pct(all_pass(r['first_draft']) for r in ok)}** "
                 f"| **{_pct(all_pass(r['final']) for r in ok)}** |")
    lines += [
        "",
        f"- Critic approved: **{_pct(r['critic_passed'] for r in ok)}** of trips "
        f"(avg {_avg((r['fix_rounds'] for r in ok), '{:.1f}')} fix rounds)",
        f"- Average time per trip: **{_avg(r['seconds'] for r in ok)} s**",
        "",
        "## Per trip (final plan)",
        "",
        "| Trip | Days | Budget used | Adds up | Places verified | Critic | Fixes | Time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        f, t = r["final"], r["trip"]
        budget = (f"{'✅' if f['within_budget'] else '❌'} ₹{f['budget_total']:,.0f} / ₹{t['budget_inr']:,.0f}"
                  if f["budget_total"] else "❌ no table")
        places = f"{f['places_verified']}/{f['places_total']}" if f["places_total"] else "–"
        lines.append(f"| {r['id']} | {'✅' if f['days_ok'] else '❌'} {len(f['days_found'])}/{r['num_days']} "
                     f"| {budget} | {'✅' if f['adds_up'] else '❌'} | {places} "
                     f"| {'✅' if r['critic_passed'] else '⚠️'} | {r['fix_rounds']} | {r['seconds']:.0f}s |")
    unverified = [(r["id"], n) for r in ok for n in r["final"]["unverified_places"]]
    if unverified:
        lines += ["", "## Unverified places (not in research data, not found on OpenStreetMap)", "",
                  "These *may* be made up — or real places OSM lists under another name. Spot-check them.", ""]
        lines += [f"- **{trip_id}**: {name}" for trip_id, name in unverified]
    flagged = [(r["id"], i) for r in ok for i in r["remaining_issues"]]
    if flagged:
        lines += ["", "## Issues the critic still flagged on the final plan", ""]
        lines += [f"- **{trip_id}**: {issue}" for trip_id, issue in flagged]
    if failed:
        lines += ["", "## Failed runs", ""] + [f"- **{r['id']}**: {r['error']}" for r in failed]
    lines += ["", "---", "",
              "**How scores work** — *Days*: every `### Day N` heading present once. *Within budget*: the "
              "budget table's Total (or its line items, whichever is higher) ≤ the budget. *Places verified*: "
              "named stays, restaurants and attractions found in the app's research data or on OpenStreetMap "
              "within ~65 km of the destination; this is a lower bound, since some real places aren't in OSM."]
    return "\n".join(lines) + "\n"


def print_summary(results: list[dict]):
    from rich.console import Console
    from rich.markdown import Markdown
    Console().print(Markdown(build_report(results).split("## Per trip")[0]))


def rescore(trip_ids: list[str], maps):
    """Apply the current scoring code to saved plans — useful after improving scoring.py."""
    for r in load_results(trip_ids):
        if "error" in r:
            continue
        research = scoring.research_text(json.load(open(os.path.join(RESEARCH_DIR, f"{r['id']}.json"))))
        r["final"] = scoring.score(r["itinerary"], r["trip"], r["num_days"], research, maps)
        if r.get("first_draft_text"):
            r["first_draft"] = scoring.score(r["first_draft_text"], r["trip"], r["num_days"], research, maps)
        with open(os.path.join(RESULTS_DIR, f"{r['id']}.json"), "w") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)
        print(f"re-scored {r['id']}")


def export_examples(trip_ids: list[str]):
    """
    Copy finished plans into static/examples.json — the instant "example trips" on the
    website, which cost no API tokens. Only plans that pass every rule check AND were
    approved by the critic are exported, so a flawed plan can't end up on the demo.
    """
    from agents import critic
    from config.settings import budget_profile
    examples = []
    for r in load_results(trip_ids):
        if "error" in r:
            continue
        t = r["trip"]
        issues = critic.rule_checks(r["itinerary"], r["num_days"], t["budget_inr"], {**t, "num_days": r["num_days"]})
        if issues or not r["critic_passed"]:
            print(f"✗ {r['id']}: not exported ({'critic not approved' if not r['critic_passed'] else issues[0][:90]})")
            continue
        examples.append({
            "id": r["id"], "trip": {k: t[k] for k in ("destination", "origin", "start_date", "end_date",
                                                      "budget_inr", "num_people", "preferences")},
            "itinerary": r["itinerary"], "num_days": r["num_days"],
            "budget_tier": budget_profile(t["budget_inr"])["tier"],
            "critic_passed": True, "revisions": r["fix_rounds"], "remaining_issues": [],
            "latency_ms": r["seconds"] * 1000,
        })
        print(f"✓ {r['id']} exported")
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
    print(f"\n{len(examples)} example(s) written to {os.path.relpath(EXAMPLES_FILE)}")


def load_results(trip_ids: list[str]) -> list[dict]:
    out = []
    for tid in trip_ids:
        path = os.path.join(RESULTS_DIR, f"{tid}.json")
        if os.path.exists(path):
            out.append(json.load(open(path)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate TripMind on sample trips.")
    ap.add_argument("--limit", type=int, help="only the first N trips")
    ap.add_argument("--only", help="comma-separated trip ids")
    ap.add_argument("--rerun", action="store_true", help="re-plan trips that already have results")
    ap.add_argument("--report", action="store_true", help="just rebuild the report from saved results")
    ap.add_argument("--no-map", action="store_true", help="skip OpenStreetMap place lookups (faster)")
    ap.add_argument("--export-examples", metavar="IDS",
                    help="comma-separated trip ids to publish as instant examples on the website")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score saved plans with the current scoring code (no AI calls)")
    args = ap.parse_args()

    if args.export_examples:
        export_examples(args.export_examples.split(","))
        return

    trips = json.load(open(TRIPS_FILE))
    if args.only:
        wanted = set(args.only.split(","))
        trips = [t for t in trips if t["id"] in wanted]
    if args.limit:
        trips = trips[:args.limit]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.rescore:
        rescore([t["id"] for t in trips], None if args.no_map else scoring.MapChecker())
    elif not args.report:
        maps = None if args.no_map else scoring.MapChecker()
        todo = [t for t in trips if args.rerun or not os.path.exists(os.path.join(RESULTS_DIR, f"{t['id']}.json"))]
        print(f"\n{len(trips)} trip(s) selected, {len(trips) - len(todo)} already done, {len(todo)} to run.\n")
        for i, trip in enumerate(todo, 1):
            print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {trip['id']} — {trip['note']}\n{'=' * 70}")
            try:
                result = evaluate_trip(trip, maps)
            except DailyLimitReached as e:
                print(f"\n⛔ Groq's daily limit was reached:\n   {str(e)[:400]}")
                print("   Finished trips are saved. Run the same command again later to continue.\n")
                break
            with open(os.path.join(RESULTS_DIR, f"{trip['id']}.json"), "w") as f:
                json.dump(result, f, indent=1, ensure_ascii=False)
            if "error" in result:
                print(f"❌ {trip['id']} failed: {result['error'][:200]}")
            else:
                fin = result["final"]
                print(f"✅ {trip['id']}: days {'ok' if fin['days_ok'] else 'WRONG'} | "
                      f"budget {'ok' if fin['within_budget'] else 'OVER/MISSING'} | "
                      f"places {fin['places_verified']}/{fin['places_total']} | "
                      f"critic {'approved' if result['critic_passed'] else 'flagged'} | {result['seconds']:.0f}s")

    results = load_results([t["id"] for t in trips])
    if not results:
        print("No results yet.")
        return
    with open(REPORT_FILE, "w") as f:
        f.write(build_report(results))
    print()
    print_summary(results)
    print(f"\nFull report: {os.path.relpath(REPORT_FILE)}")


if __name__ == "__main__":
    main()
