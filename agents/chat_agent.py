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
from utils.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

CHAT_SYSTEM_PROMPT = """You are a friendly and expert travel planner assistant.
The user has already received a travel itinerary and now wants to make changes to it.

Your job is to:
1. Understand what change the user wants
2. Modify ONLY the relevant parts of the itinerary
3. Keep everything else exactly the same
4. Return the COMPLETE updated itinerary in Markdown format

Rules:
- If user says "make it cheaper" → suggest lower priced hotels/transport, remove expensive activities
- If user says "add activity" → add it on the most suitable day with timing and cost
- If user says "family friendly" → add kid-friendly activities, remove adult venues, add safety tips
- If user says "more luxury" → upgrade hotel and food recommendations with specific names
- If user asks a question → answer it directly without rewriting the whole itinerary
- Always be conversational and friendly
- If the change affects the budget, recalculate and show updated cost breakdown
- Keep hotel names, flight names and specific details — only change what the user asked for

Always respond in this format:
- First: acknowledge what the user wants in 1 line
- Then: provide the updated itinerary OR answer their question
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
        response = self.llm.invoke(messages)
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
def refine_itinerary(
    user_message:        str,
    current_itinerary:   str,
    trip_details:        dict,
    conversation_history: list,
) -> tuple[str, list]:
    """
    Stateless version for API use.
    Takes conversation history as input, returns response + updated history.

    Returns:
        (assistant_response, updated_conversation_history)
    """
    llm = get_llm(temperature=0.5)

    context = f"""
## Trip Details
- Destination: {trip_details.get('destination', '')}
- Dates: {trip_details.get('start_date', '')} to {trip_details.get('end_date', '')}
- Budget: ₹{trip_details.get('budget_inr', 0):,.0f}
- People: {trip_details.get('num_people', 1)}

## Current Itinerary
{current_itinerary}

## User Request
{user_message}
"""

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=context))

    response     = llm.invoke(messages)
    reply        = response.content if hasattr(response, "content") else str(response)

    updated_history = conversation_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    return reply, updated_history
