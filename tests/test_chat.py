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
