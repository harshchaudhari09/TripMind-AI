"""
agents/chat_agent.py

Chatbot that lets users refine their generated itinerary
through natural conversation.

Examples:
- "Make Day 2 more budget friendly"
- "Add a waterfall visit on Day 3"
- "We have a child, make it family friendly"
- "Replace the hotel with a homestay"
- "Reduce the budget by 20000"
"""
from __future__ import annotations
import re
from utils.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

CHAT_SYSTEM_PROMPT = """You are a friendly and expert travel planner assistant.
The user has already received a travel itinerary and now wants to make changes to it.

Your job is to:
1. Understand what change the user wants
2. Modify ONLY the relevant parts of the itinerary
3. Keep everything else exactly the same
4. Return ONLY the parts you changed (code merges them into the full plan)

Rules:
- If user says "make it cheaper" → suggest lower priced hotels/transport, remove expensive activities
- If user says "add activity" → add it on the most suitable day with timing and cost
- If user says "family friendly" → add kid-friendly activities, remove adult venues, add safety tips
- If user says "more luxury" → upgrade hotel and food recommendations with specific names
- If user asks a question → answer it directly without rewriting the whole itinerary
- Always be conversational and friendly
- If the change affects the budget, recalculate and show updated cost breakdown
- Keep hotel names, flight names and specific details — only change what the user asked for
- Keep the same layout: the same "## " section headings, one "### Day N — ..." heading per day,
  bold times, and the "| Item | Cost |" budget table (its Total is recalculated by code,
  so just keep each line item accurate)

Always respond in this format:
- First: ONE short plain sentence saying what you changed (no heading, no bold).
- Then: ONLY the changed parts, each under its ORIGINAL heading, written in full:
  - a changed day → the whole "### Day N — ..." section (all its bullets)
  - a changed section → the whole "## ..." section, e.g. "## 🏨 Where to stay"
  - if any cost changed → also the whole "## 💰 Budget breakdown" table with updated line items
- Do NOT repeat parts that didn't change.
- If the user only asked a question, just answer it in a few sentences — no headings.
"""


class ChatAgent:
    """
    Maintains conversation history and refines itinerary
    based on user messages.
    """

    def __init__(self, original_itinerary: str, trip_details: dict):
        self.llm = get_llm(temperature=0.5)
        self.original_itinerary = original_itinerary
        self.trip_details = trip_details
        self.conversation_history = []
        self.current_itinerary = original_itinerary

    def chat(self, user_message: str) -> str:
        """
        Takes a user message, updates the itinerary accordingly,
        returns the assistant response.
        """
        # Build context message with current itinerary
        context = f"""
## Current Trip Details
- Destination: {self.trip_details.get('destination', '')}
- Dates: {self.trip_details.get('start_date', '')} to {self.trip_details.get('end_date', '')}
- Budget: ₹{self.trip_details.get('budget_inr', 0):,.0f}
- People: {self.trip_details.get('num_people', 1)}

## Current Itinerary
{self.current_itinerary}

## User Request
{user_message}
"""

        # Build messages list with full history
        messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

        # Add conversation history
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Add current message with context
        messages.append(HumanMessage(content=context))

        # Get response
        response = self.llm.invoke(messages, config={"run_name": "chat: refine itinerary"})
        assistant_reply = response.content if hasattr(response, "content") else str(response)

        # Update conversation history
        self.conversation_history.append({"role": "user",      "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Update current itinerary if response contains a full itinerary
        if "## Day" in assistant_reply or "**Day" in assistant_reply:
            self.current_itinerary = assistant_reply

        return assistant_reply

    def reset(self):
        """Reset conversation but keep original itinerary."""
        self.conversation_history = []
        self.current_itinerary = self.original_itinerary


# ── Simple function version for API use ──────────────────────────────────────
_DAY_HEADING = re.compile(r"^#{2,4}\s*Day\s*\d+", re.MULTILINE)
MAX_HISTORY_MESSAGES = 6


def _split_reply(reply: str) -> tuple[str, str | None]:
    """'Made Day 2 cheaper.\n### Day 2 — ...' → (acknowledgement, edited parts or None)."""
    lines = reply.splitlines()
    first_heading = next((i for i, l in enumerate(lines) if l.lstrip().startswith("#")), None)
    if first_heading is None:
        return reply.strip(), None                     # a plain answer, not an edit
    ack_lines = [l.strip().strip("*").strip() for l in lines[:first_heading]]
    ack = " ".join(l for l in ack_lines if l and not re.fullmatch(r"[-*_]{3,}", l))
    return ack, "\n".join(lines[first_heading:]).strip() + "\n"


def _section_key(heading: str) -> str:
    """'## 🗓️ Day‑by‑day plan (updated)' → 'day by day plan' (emoji, notes and dashes ignored)."""
    text = re.sub(r"\(.*?\)", "", heading.lstrip("#")).lower()
    words = [w for w in re.findall(r"[a-z]+", text) if w not in ("updated", "revised", "new")]
    return " ".join(words)


def _blocks(md: str) -> tuple[list[tuple[str, str, str]], str]:
    """
    Split an itinerary into ordered blocks. Returns ([(kind, key, text)], preamble)
    where kind is "h2" (a '## ' section without its day sections) or "day"
    (a '### Day N' section, key = N). Everything before the first '## ' is the preamble.
    """
    lines = md.splitlines(keepends=True)
    blocks, preamble, cur = [], [], None
    for line in lines:
        day = re.match(r"^#{3,4}\s*Day\W*(\d+)", line)
        if day:
            cur = ["day", day.group(1), [line]]
            blocks.append(cur)
        elif re.match(r"^##\s", line):
            cur = ["h2", _section_key(line), [line]]
            blocks.append(cur)
        elif cur is None:
            preamble.append(line)
        else:
            cur[2].append(line)
    return [(k, key, "".join(t)) for k, key, t in blocks], "".join(preamble)


def merge_edit(current: str, edit: str) -> tuple[str, list[str]]:
    """
    Put the changed sections from `edit` into `current`. Returns (merged plan, names of
    what was replaced). A changed '### Day N' replaces that day; a changed '## Section'
    replaces the section with the same name. Unchanged parts stay exactly as written.
    """
    cur_blocks, preamble = _blocks(current)
    new_blocks, _ = _blocks(edit)
    new_days = {key: text for kind, key, text in new_blocks if kind == "day"}
    # A '## ' block that only wraps day sections (e.g. "Day-by-day plan (updated)") is just a
    # container — the days themselves are merged above.
    new_sections = {key: text for kind, key, text in new_blocks
                    if kind == "h2" and len(text.strip().splitlines()) > 1}
    replaced, merged = [], []
    for kind, key, text in cur_blocks:
        if kind == "day" and key in new_days:
            new = new_days[key].rstrip("\n") + "\n\n"
            merged.append(new); replaced.append(f"Day {key}")
        elif kind == "h2" and key in new_sections:
            new = new_sections[key].rstrip("\n") + "\n\n"
            merged.append(new); replaced.append(key)
        else:
            merged.append(text)
    return preamble + "".join(merged), replaced


def review_edit(itinerary: str, trip_details: dict) -> dict:
    """
    Run a chat-edited itinerary through the SAME checks as a new plan: code recalculates
    the budget totals, the critic reviews it, and if it finds problems the writer gets
    one fix round (chat edits are cheaper to keep at one round). The better of the two
    versions is returned with the critic's verdict.
    """
    from agents import critic
    from agents.budget import recompute_totals
    from agents.graph import node_init, _revise_itinerary

    trip = {
        "destination": str(trip_details["destination"]), "origin": str(trip_details["origin"]),
        "start_date": str(trip_details["start_date"]), "end_date": str(trip_details["end_date"]),
        "budget_inr": float(trip_details["budget_inr"]), "num_people": int(trip_details.get("num_people", 1)),
        "preferences": list(trip_details.get("preferences") or []),
    }
    state = {**trip, **node_init(trip)}
    itinerary, _ = recompute_totals(itinerary, trip["budget_inr"])
    result = critic.review(itinerary, state)
    issues = result["rule_issues"] + result["llm_issues"]
    revisions = 0
    if issues:
        fixed = _revise_itinerary({**state, "itinerary": itinerary, "critique": issues,
                                   "issue_log": issues, "revision_count": 0})
        result2 = critic.review(fixed["itinerary"], state, earlier_issues=issues)
        issues2 = result2["rule_issues"] + result2["llm_issues"]
        revisions = 1
        if len(issues2) <= len(issues):               # keep the fix unless it made things worse
            itinerary, issues = fixed["itinerary"], issues2
    return {"itinerary": itinerary, "critic_passed": not issues,
            "revisions": revisions, "remaining_issues": issues}


def refine_itinerary(
    user_message:        str,
    current_itinerary:   str,
    trip_details:        dict,
    conversation_history: list,
) -> tuple[str, list, dict | None]:
    """
    Stateless version for API use.

    Returns (reply, updated_history, review). `review` is None when the reply is a plain
    answer; when the plan was rewritten it holds the checked itinerary and critic verdict.

    History only keeps short messages: the full current itinerary is sent every time
    anyway, and resending old full itineraries would push the request past Groq's
    8,000-token limit after two or three edits.
    """
    from agents.graph import _fit_max_tokens

    context = f"""
## Trip Details
- Destination: {trip_details.get('destination', '')}
- Dates: {trip_details.get('start_date', '')} to {trip_details.get('end_date', '')}
- Budget: ₹{float(trip_details.get('budget_inr', 0)):,.0f}
- People: {trip_details.get('num_people', 1)}

## Current Itinerary
{current_itinerary}

## User Request
{user_message}
"""

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
    for msg in conversation_history[-MAX_HISTORY_MESSAGES:]:
        content = str(msg.get("content", ""))[:600]
        messages.append(HumanMessage(content=content) if msg.get("role") == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=context))

    llm = get_llm(temperature=0.5, max_tokens=_fit_max_tokens(messages, 4000), reasoning_effort="low")
    response = llm.invoke(messages, config={"run_name": "chat: refine itinerary"})
    reply    = response.content if hasattr(response, "content") else str(response)

    ack, edited_parts = _split_reply(reply)
    review = None
    if edited_parts:
        merged, replaced = merge_edit(current_itinerary, edited_parts)
        if replaced:
            review = review_edit(merged, trip_details)
            if review["revisions"]:
                ack = (ack + " " if ack else "") + "The critic found a problem with that change and fixed it."
            reply = ack or "Updated the itinerary."
        else:
            reply = ack or reply                       # headings didn't match anything: show as text

    updated_history = conversation_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply if review else reply[:600]},
    ]
    return reply, updated_history, review
