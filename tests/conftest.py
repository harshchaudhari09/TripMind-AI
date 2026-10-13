"""
Shared test setup. Every test runs offline: no Groq, Tavily or map calls, so the
suite is free, fast and works without API keys (e.g. on GitHub Actions).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fake_llm():
    """Returns a factory: fake_llm(["reply 1", "reply 2"]) → a stand-in for get_llm()."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    def factory(responses):
        # One shared queue: the app creates a new LLM object for every call, and each
        # call should get the NEXT reply (draft, then fix, ...), not start over.
        queue = list(responses)

        def get_llm(**kwargs):
            reply = queue.pop(0) if len(queue) > 1 else queue[0]
            return FakeListChatModel(responses=[reply])
        return get_llm
    return factory


@pytest.fixture
def no_research():
    """Research nodes that do nothing — the graph runs without any web calls."""
    import agents.graph as G
    return {name: (lambda state: {}) for name in G.RESEARCH_NODES}


def make_plan(days, budget_lines=(("Stay", 10000), ("Food", 5000)), total=None, extra_day_lines=None):
    """Build a small itinerary in TripMind's format for tests."""
    out = ["# Test Trip", "> **A → B** · Estimated total ₹1", "## ✨ At a glance",
           "- **Total cost:** ₹1 of ₹1 budget", "## 🗓️ Day-by-day plan"]
    for d in range(1, days + 1):
        out.append(f"### Day {d} — Mon, 0{d} Nov · Theme")
        out += (extra_day_lines or {}).get(d, ["- **9:00 AM** — Breakfast", "- **1:00 PM** — Lunch at **Cafe X**"])
    out += ["## 💰 Budget breakdown", "| Item | Cost |", "|---|---|"]
    out += [f"| {name} | ₹{amount:,} |" for name, amount in budget_lines]
    out.append(f"| **Total** | **₹{(total if total is not None else sum(a for _, a in budget_lines)):,}** |")
    return "\n".join(out) + "\n"
