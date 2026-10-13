"""The critic's rule checks — plain Python, so they must always give the same answer."""
from agents import critic
from tests.conftest import make_plan

NOV_MANALI = {"destination": "Manali, Himachal Pradesh", "start_date": "2026-11-09", "end_date": "2026-11-14"}


def rules(text, **kw):
    return critic.rule_checks(text, kw.get("days", 2), kw.get("budget", 20000), kw.get("trip"))


# ── Days ──────────────────────────────────────────────────────────────────────
def test_all_days_present_passes():
    assert rules(make_plan(2)) == []


def test_missing_day_is_flagged():
    issues = critic.check_days(make_plan(2), 3)
    assert any("Missing day" in i and "Day 3" in i for i in issues)


def test_repeated_day_is_flagged():
    text = make_plan(2).replace("### Day 2", "### Day 1")
    assert any("more than once" in i for i in critic.check_days(text, 2))


# ── Budget ────────────────────────────────────────────────────────────────────
def test_budget_total_that_does_not_add_up_is_flagged():
    issues = critic.check_budget(make_plan(1, total=9000), 20000)
    assert any("doesn't add up" in i for i in issues)


def test_over_budget_is_flagged_even_if_total_row_hides_it():
    # The real Goa case: Total row said ₹28,850 but the items added up to ₹36,950.
    text = make_plan(1, budget_lines=(("Stay", 20000), ("Travel", 16950)), total=28850)
    assert any("Over budget" in i and "36,950" in i for i in critic.check_budget(text, 30000))


def test_missing_budget_table_is_flagged():
    assert critic.check_budget("### Day 1\n- **9:00 AM** — x", 5000)


def test_indian_number_format_is_parsed():
    items, total = critic.parse_budget_table("| Stay | ₹1,20,000 |\n| **Total** | **₹1,20,000** |")
    assert items == [("Stay", 120000.0)] and total == 120000.0


# ── Time order ────────────────────────────────────────────────────────────────
def day(*lines):
    return "### Day 1 — x\n" + "\n".join(lines)


def test_times_going_backwards_are_flagged():
    text = day("- **9:00 AM** — a", "- **2:00 PM** — Lunch", "- **11:00 AM** — c")
    assert any("backwards" in i for i in critic.check_time_order(text))


def test_overnight_journey_ending_the_day_is_fine():
    text = day("- **5:00 PM** — Dinner", "- **7:00 PM** — Board overnight bus", "- **6:30 AM** — Arrive")
    assert critic.check_time_order(text) == []


def test_next_day_plans_under_the_travel_day_are_flagged():
    # The real Munnar case: Friday's sightseeing was filed under Thursday.
    text = day("- **4:30 PM** — Board Volvo bus", "- **1:35 AM** — Arrive", "- **8:00 AM** — Breakfast")
    issues = critic.check_time_order(text)
    assert any("overnight journey" in i and "Day 2" in i for i in issues)


def test_activities_after_midnight_are_flagged():
    # The real Varanasi case: Ganga Aarti planned for 1:30 AM.
    text = day("- **11:30 PM** — Arrive at hotel", "- **1:30 AM** — Ganga Aarti")
    assert any("past midnight" in i for i in critic.check_time_order(text))


# ── Time before flights / trains ──────────────────────────────────────────────
def test_too_little_time_before_a_flight_is_flagged():
    text = day("- **6:15 PM** — Leave hotel for the airport", "- **7:00 PM** — Flight departs")
    assert any("45 min" in i for i in critic.check_departure_buffers(text))


def test_plenty_of_time_before_a_flight_is_fine():
    # The AI reviewer once complained about this 6-hour gap; the code check doesn't.
    text = day("- **1:00 PM** — Leave hotel for the airport", "- **7:00 PM** — Flight departs")
    assert critic.check_departure_buffers(text) == []


def test_too_little_time_before_a_train_is_flagged():
    text = day("- **6:40 AM** — Auto to the station", "- **7:00 AM** — Train 12015 departs")
    assert any("train" in i for i in critic.check_departure_buffers(text))


# ── Seasons (verified list, not AI guesses) ───────────────────────────────────
def test_closed_pass_in_season_is_flagged():
    text = day("- **9:00 AM** — Drive to **Rohtang Pass** for snow")
    assert any("Rohtang" in i for i in critic.check_season(text, NOV_MANALI))


def test_mentioning_a_closure_is_not_flagged():
    text = day("- **9:00 AM** — Rohtang Pass is closed, so we visit Sissu instead")
    assert critic.check_season(text, NOV_MANALI) == []


def test_same_pass_in_summer_is_fine():
    june = {**NOV_MANALI, "start_date": "2026-06-10", "end_date": "2026-06-12"}
    assert critic.check_season(day("- **9:00 AM** — **Rohtang Pass**"), june) == []


def test_open_all_year_place_is_never_flagged():
    # The AI reviewer once claimed Khardung La closes in November — it doesn't.
    leh = {"destination": "Leh, Ladakh", "start_date": "2026-11-10", "end_date": "2026-11-15"}
    assert critic.check_season(day("- **8:00 AM** — Drive over **Khardung La** to Nubra"), leh) == []


def test_trip_months_cross_the_new_year():
    assert critic.trip_months("2026-12-28", "2027-01-03") == [12, 1]
