"""Code recalculates budget totals so the AI's arithmetic mistakes can't reach users."""
from agents.budget import recompute_totals
from tests.conftest import make_plan


def test_wrong_total_is_replaced_everywhere():
    plan = make_plan(1, budget_lines=(("Stay", 20000), ("Travel", 16950)), total=28850)
    fixed, total = recompute_totals(plan, 30000)
    assert total == 36950
    assert "| **Total** | **₹36,950** |" in fixed
    assert "Estimated total ₹36,950" in fixed
    assert "₹6,950 over budget" in fixed


def test_under_budget_shows_amount_left():
    fixed, _ = recompute_totals(make_plan(1), 20000)
    assert "₹5,000 left over" in fixed


def test_rest_of_the_plan_is_unchanged():
    plan = make_plan(1).replace("Theme", "Arrival – 5–8 Dec stay")
    fixed, _ = recompute_totals(plan, 20000)
    assert "Arrival – 5–8 Dec stay" in fixed       # dashes and special spaces kept as written


def test_plan_without_budget_table_is_left_alone():
    assert recompute_totals("### Day 1\n- **9:00 AM** — x", 1000) == ("### Day 1\n- **9:00 AM** — x", None)
