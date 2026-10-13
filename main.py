"""
main.py — TripMind AI launcher

Default:  starts the API server and opens the web UI in your browser.
--cli:    runs the original terminal-based planner instead.

Usage:
    python main.py            # web UI (recommended)
    python main.py --cli      # terminal mode
    python main.py --no-browser --port 8000   # server only, no auto-open
"""
import sys, os, argparse, threading, time, webbrowser, socket
sys.path.insert(0, os.path.dirname(__file__))


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def run_web(host: str, port: int, open_browser: bool):
    from config.settings import GROQ_API_KEY, TAVILY_API_KEY

    if not GROQ_API_KEY or not TAVILY_API_KEY:
        print("\n⚠  GROQ_API_KEY and/or TAVILY_API_KEY are missing.")
        print("   Copy .env.example to .env and add your keys, then re-run.\n")
        print("   Starting anyway — the UI will load, but planning a trip will fail until the keys are set.\n")

    # Only one program can use a port. If something is already there, it's
    # usually an older TripMind still running — explain instead of crashing.
    if _port_is_open("127.0.0.1", port):
        print(f"\n⚠  Port {port} is already in use — probably TripMind is already running.")
        print(f"   • To use it: open http://127.0.0.1:{port}")
        print(f"   • To restart with your latest code: press Ctrl+C in the terminal running it,")
        print(f"     or find it with:  lsof -nP -iTCP:{port} -sTCP:LISTEN   then:  kill <PID>")
        print(f"   • Or run this copy on another port:  venv/bin/python main.py --port {port + 1}\n")
        sys.exit(1)

    import uvicorn

    if open_browser:
        def _open_when_ready():
            url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
            for _ in range(100):  # ~20s max
                if _port_is_open("127.0.0.1", port):
                    webbrowser.open(url)
                    print(f"\n✈  TripMind AI is running — {url}\n")
                    return
                time.sleep(0.2)
        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run("api:app", host=host, port=port, reload=False)


def run_cli():
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from datetime import date, timedelta

    from agents.graph import travel_graph
    from config.settings import budget_profile
    from utils.llmops_logger import trace_config, tracing_status

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]✈  TripMind AI — Smart Travel Planner (CLI mode)[/bold cyan]\n"
        "[dim]Real flights • Hotels • Trains • Weather • ATMs • Petrol Pumps[/dim]",
        border_style="cyan",
    ))

    destination = Prompt.ask("[green]Destination[/green]", default="Manali, Himachal Pradesh")
    origin      = Prompt.ask("[green]Departing from[/green]", default="Delhi")
    # Default to a trip 30 days from today, so the weather forecast is real.
    default_start = date.today() + timedelta(days=30)
    start_date  = Prompt.ask("[green]Start date (YYYY-MM-DD)[/green]", default=default_start.isoformat())
    end_date    = Prompt.ask("[green]End date (YYYY-MM-DD)[/green]", default=(default_start + timedelta(days=5)).isoformat())
    budget_str  = Prompt.ask("[green]Total budget (INR)[/green]", default="80000")
    budget_inr  = float(budget_str.replace(",", "").replace("₹", "").strip())
    num_people  = int(Prompt.ask("[green]Number of people[/green]", default="2"))
    pref_str    = Prompt.ask("[green]Interests (comma separated)[/green]", default="trekking, snow, local food, adventure")
    preferences = [p.strip() for p in pref_str.split(",")]

    profile = budget_profile(budget_inr)
    console.print(f"\n[yellow]Tier:[/yellow] {profile['tier'].upper()} | [yellow]Hotel:[/yellow] ₹{profile['hotel_min']:,}–₹{profile['hotel_max']:,}/night\n")

    initial_state = {
        "destination": destination, "origin": origin,
        "start_date": start_date, "end_date": end_date,
        "budget_inr": budget_inr, "num_people": num_people,
        "preferences": preferences,
        "budget_tier": "", "num_days": 0,
        "flight_data": "", "train_data": "", "bus_data": "",
        "hotel_data": "", "weather_data": "", "attraction_data": "",
        "food_data": "", "wiki_data": "",
        "petrol_data": "", "atm_data": "", "hospital_data": "",
        "bus_station_data": "", "train_station_data": "",
        "itinerary": "", "messages": [],
    }

    console.print(f"[dim]{tracing_status()}[/dim]")
    console.print("\n[bold]Running parallel research pipeline...[/bold]\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Planning your trip...", total=None)
        final = travel_graph.invoke(initial_state, config=trace_config(initial_state, source="cli"))
        p.update(task, description="[green]Done![/green]")

    # No surrounding Panel: the border steals width and squashes the text.
    # Cap the width so lines stay readable on very wide terminals.
    console.print()
    console.rule(f"[bold cyan]{destination} — {final['num_days']} Day Itinerary[/bold cyan]", style="green")
    console.print(Markdown(final["itinerary"]), width=min(console.width, 100))
    console.rule(style="green")

    # Critic agent verdict
    fixes = final.get("revision_count", 0)
    if final.get("critic_passed"):
        console.print(f"\n[bold green]✅ Checked by the critic agent[/bold green] — approved after {fixes} fix round(s).")
    else:
        console.print(f"\n[bold yellow]⚠ Critic still flags {len(final.get('critique', []))} issue(s) after {fixes} fix round(s):[/bold yellow]")
        for issue in final.get("critique", []):
            console.print(f"  • {issue}", width=min(console.width, 100))

    os.makedirs("output", exist_ok=True)
    fname = f"output/{destination.replace(', ', '_').replace(' ', '_')}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# {destination} Travel Itinerary\n\n")
        f.write(f"**Dates:** {start_date} → {end_date} | **Budget:** ₹{budget_inr:,.0f} | **People:** {num_people}\n\n")
        f.write(final["itinerary"])

    console.print(f"\n[dim]Saved to:[/dim] [cyan]{fname}[/cyan]")


def main():
    parser = argparse.ArgumentParser(description="TripMind AI launcher")
    parser.add_argument("--cli", action="store_true", help="run the terminal-based planner instead of the web UI")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind the server to (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to bind the server to (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_web(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
