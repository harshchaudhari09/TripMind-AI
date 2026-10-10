"""
agents/budget.py

LLMs are bad at adding up numbers: in the evaluation, budget tables whose items didn't
match their Total were the most common problem, and fixing them cost extra LLM rounds.

So the AI only writes the line items, and this code does the arithmetic: it recomputes
the Total row and the two places that repeat it (the header line and "At a glance").
"""
from __future__ import annotations
import re

from agents.critic import _normalise, parse_budget_table


def _inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


def recompute_totals(itinerary: str, budget_inr: float) -> tuple[str, float | None]:
    """Return (itinerary with corrected totals, correct total). Unchanged if there's no table."""
    text = _normalise(itinerary)
    items, _ = parse_budget_table(text)
    if not items:
        return itinerary, None
    total = sum(amount for _, amount in items)

    # Detect lines on the cleaned-up copy, but edit the original so the rest of the
    # itinerary keeps its exact characters (the clean-up swaps dashes and spaces 1-for-1).
    lines = itinerary.splitlines()
    for i, (line, clean) in enumerate(zip(lines, text.splitlines())):
        stripped = clean.strip()
        # The table's Total row: "| **Total** | **₹44,800** |"
        if stripped.startswith("|") and "total" in stripped.split("|")[1].lower():
            lines[i] = f"| **Total** | **{_inr(total)}** |"
        # "> **Mumbai → Leh** · ... · Estimated total ₹44,800"
        elif re.search(r"estimated total", clean, re.I):
            lines[i] = re.sub(r"(estimated total[^₹\d]*)₹?\s*[\d,]+", rf"\g<1>{_inr(total)}", line, flags=re.I)
        # "- **Total cost:** ₹44,800 of ₹50,000 budget (₹5,200 left over)"
        elif re.match(r"\s*[-*]\s*\*\*total cost:?\*\*", clean, re.I):
            gap = budget_inr - total
            rest = f"{_inr(gap)} left over" if gap >= 0 else f"{_inr(-gap)} over budget"
            lines[i] = f"- **Total cost:** {_inr(total)} of {_inr(budget_inr)} budget ({rest})"
    return "\n".join(lines) + ("\n" if itinerary.endswith("\n") else ""), total
