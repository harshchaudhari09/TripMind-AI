"""Chat edits go through the same critic as new plans."""
import agents.chat_agent as C
import agents.graph as G
from agents import critic
from tests.conftest import make_plan

TRIP = {"destination": "Manali, Himachal Pradesh", "origin": "Delhi", "start_date": "2026-11-09",
        "end_date": "2026-11-10", "budget_inr": 30000, "num_people": 1, "preferences": []}
APPROVE = '{"approved": true, "issues": []}'


def test_plain_answer_is_not_treated_as_a_new_plan():
    ack, plan = C._split_reply("Rohtang is closed in November, so I'd skip it.")
    assert plan is None


def test_rewritten_plan_is_split_from_the_acknowledgement():
    ack, plan = C._split_reply("Sure, made it cheaper!\n" + make_plan(2))
    assert ack == "Sure, made it cheaper!" and plan.startswith("# Test Trip")


def test_chat_edit_that_breaks_a_rule_is_fixed(monkeypatch, fake_llm):
    bad = make_plan(2, extra_day_lines={1: ["- **9:00 AM** — Drive to **Rohtang Pass**"]})
    monkeypatch.setattr(C, "get_llm", fake_llm(["Added a snow trip!\n" + bad]))
    monkeypatch.setattr(G, "get_llm", fake_llm([make_plan(2)]))        # the fix round
    monkeypatch.setattr(critic, "get_llm", fake_llm([APPROVE] * 4))
    reply, history, review = C.refine_itinerary("add a snow trip", make_plan(2), TRIP, [])
    assert review["critic_passed"] and review["revisions"] == 1
    assert "Rohtang" not in review["itinerary"]
    assert "critic found a problem" in reply


def test_long_chat_history_is_trimmed(monkeypatch, fake_llm):
    sent = {}

    def capture(**kwargs):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        class Spy(FakeListChatModel):
            def invoke(self, messages, config=None, **kw):
                sent["messages"] = messages
                return super().invoke(messages, config=config, **kw)
        return Spy(responses=["Just a tip: carry warm clothes."])
    monkeypatch.setattr(C, "get_llm", capture)
    old = [{"role": "assistant", "content": "X" * 20000}] * 20
    C.refine_itinerary("any tips?", make_plan(2), TRIP, old)
    history_part = sent["messages"][1:-1]
    assert len(history_part) <= C.MAX_HISTORY_MESSAGES
    assert all(len(m.content) <= 600 for m in history_part)


# ── Merging partial edits (the "says updated but doesn't" bug) ────────────────
def test_only_the_changed_day_and_budget_are_replaced():
    current = make_plan(3)
    reply = ("**Got it – I've trimmed Day 2.**\n\n---\n\n## 🗓️ Day‑by‑day plan (updated)\n\n"
             "### Day 2 — Tue, 02 Nov · Cheaper day\n- **9:00 AM** — Free walking tour\n\n"
             "## 💰 Budget breakdown\n| Item | Cost |\n|---|---|\n| Stay | ₹8,000 |\n| **Total** | **₹8,000** |\n")
    ack, parts = C._split_reply(reply)
    merged, replaced = C.merge_edit(current, parts)
    assert ack == "Got it – I've trimmed Day 2."
    assert replaced == ["Day 2", "budget breakdown"]
    assert "Free walking tour" in merged and "Cheaper day" in merged
    assert merged.count("### Day") == 3                      # days 1 and 3 are still there
    assert "| Stay | ₹8,000 |" in merged and "| Food | ₹5,000 |" not in merged


def test_edit_with_unknown_headings_changes_nothing():
    current = make_plan(2)
    merged, replaced = C.merge_edit(current, "## Something unrelated\nSome text\n")
    assert replaced == [] and merged == current


def test_partial_chat_reply_keeps_the_full_plan(monkeypatch, fake_llm):
    partial = "Made Day 2 cheaper.\n### Day 2 — Tue, 02 Nov · Budget day\n- **9:00 AM** — Free museum\n"
    monkeypatch.setattr(C, "get_llm", fake_llm([partial]))
    monkeypatch.setattr(critic, "get_llm", fake_llm([APPROVE] * 2))
    reply, _, review = C.refine_itinerary("make day 2 cheaper", make_plan(2), TRIP, [])
    assert reply == "Made Day 2 cheaper."
    assert review["critic_passed"] and review["revisions"] == 0
    assert "### Day 1" in review["itinerary"] and "Free museum" in review["itinerary"]
    assert review["changed_sections"] == ["Day 2"]           # the page highlights these


# ── A new budget changes the trip (found in a real LangSmith trace) ───────────
def test_new_budget_is_understood():
    assert C.new_budget_in("my budget is 60000") == 60000
    assert C.new_budget_in("can you make it fit a 45k budget") == 45000
    assert C.new_budget_in("change the total budget to ₹1,20,000 and adjust") == 120000
    assert C.new_budget_in("budget 1.5 lakh") == 150000
    assert C.new_budget_in("make day 2 cheaper") is None
    assert C.new_budget_in("my budget is 5") is None                 # not a real budget


def test_new_budget_is_used_by_the_critic(monkeypatch, fake_llm):
    # Plan costs ₹15,000. Lowering the budget to ₹10,000 must make the critic flag it.
    monkeypatch.setattr(C, "get_llm", fake_llm(["Adjusted for the new budget.\n" + make_plan(2)]))
    monkeypatch.setattr(G, "get_llm", fake_llm([make_plan(2)]))
    monkeypatch.setattr(critic, "get_llm", fake_llm([APPROVE] * 4))
    reply, _, review = C.refine_itinerary("my budget is 10000", make_plan(2), TRIP, [])
    assert review["trip_details"]["budget_inr"] == 10000
    assert any("Over budget" in i for i in review["remaining_issues"])
