"""The writer ⇄ critic loop, run end to end with fake AI models (no API calls)."""
import agents.graph as G
from agents import critic
from tests.conftest import make_plan

TRIP = {"destination": "Manali, Himachal Pradesh", "origin": "Delhi", "start_date": "2026-11-09",
        "end_date": "2026-11-10", "budget_inr": 30000.0, "num_people": 1, "preferences": []}
APPROVE = '{"approved": true, "issues": []}'


def run(monkeypatch, fake_llm, no_research, writer_replies, critic_replies=(APPROVE,) * 5):
    monkeypatch.setattr(G, "get_llm", fake_llm(writer_replies))
    monkeypatch.setattr(critic, "get_llm", fake_llm(critic_replies))
    return G.build_graph(no_research).invoke(dict(TRIP))


def test_good_first_draft_is_approved_without_fixes(monkeypatch, fake_llm, no_research):
    final = run(monkeypatch, fake_llm, no_research, [make_plan(2)])
    assert final["critic_passed"] and final["revision_count"] == 0


def test_bad_draft_is_sent_back_and_fixed(monkeypatch, fake_llm, no_research):
    bad = make_plan(2, extra_day_lines={1: ["- **9:00 AM** — Drive to **Rohtang Pass**"]})
    good = make_plan(2)
    final = run(monkeypatch, fake_llm, no_research, [bad, good])
    assert final["critic_passed"] and final["revision_count"] == 1
    assert "Rohtang" not in final["itinerary"]
    assert "Rohtang" in final["first_draft"]          # kept for evaluation


def test_loop_stops_early_when_the_same_problem_repeats(monkeypatch, fake_llm, no_research):
    bad = make_plan(2, extra_day_lines={1: ["- **9:00 AM** — Drive to **Rohtang Pass**"]})
    final = run(monkeypatch, fake_llm, no_research, [bad, bad, bad])
    assert not final["critic_passed"]
    assert final["revision_count"] == 1               # stopped after one useless round, not two
    assert final["critique"]                          # the remaining problem is reported


def test_reworded_complaint_counts_as_stuck():
    a = ("Day 3: The itinerary has travellers driving to Colva, Palolem and Cabo de Rama and "
         "returning to the Calangute hotel by 7:30 PM, which is impossible; stay overnight in South Goa.")
    b = ("Day 3: The itinerary expects to visit Colva, Palolem, and Cabo de Rama and return to the "
         "Calangute hotel by 7:30 PM, which is impossible; stay overnight in South Goa.")
    h = lambda *i: {"rule_issues": [], "llm_issues": list(i)}
    assert G._is_stuck({"critic_history": [h(a), h(b)]})
    assert not G._is_stuck({"critic_history": [h(a), h("Day 3: The fort closes at 5 PM; go earlier.")]})


def test_requests_fit_groqs_8000_token_limit():
    from langchain_core.messages import HumanMessage
    huge = [HumanMessage(content="word " * 5000)]
    small = [HumanMessage(content="hello")]
    assert G._fit_max_tokens(small, 4000) == 4000
    assert G._fit_max_tokens(huge, 4000) < 4000
