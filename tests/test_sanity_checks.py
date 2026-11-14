"""Phase 9b: deterministic sanity checks (dates, non-negativity, fund registry)."""
import datetime as dt
import json
from pathlib import Path

import pytest

from evals.scoring import load_truth
from generator.common import FUNDS
from validation import rules
from validation.validate import validate_record

ROOT = Path(__file__).resolve().parents[1]
FUND = FUNDS[0].name


def cas(**kw):
    f = dict(fund_name=FUND, manager="M", investor_name="I", period_start="2025-07-01", period_end="2025-09-30",
             currency="USD", beginning_nav=100.0, contributions=0.0, distributions=0.0, management_fees=1.0,
             incentive_fees=0.0, net_gain_loss=1.0, ending_nav=100.0, total_commitment=200.0,
             unfunded_commitment=50.0, confidence=0.95)
    return {**f, **kw}


def soi(**kw):
    hs = [dict(name="A", sector="S", geography="G", cost=10.0, fair_value=60.0, pct_of_nav=60.0),
          dict(name="B", sector="S", geography="G", cost=10.0, fair_value=40.0, pct_of_nav=40.0)]
    return {**dict(fund_name=FUND, as_of_date="2025-09-30", holdings=hs, total_fair_value=100.0, confidence=0.9), **kw}


# --------------------------------------------------------------------------- dates

def test_period_order():
    assert rules.cas_period_order(cas()) is None
    assert rules.cas_period_order(cas(period_end="2025-07-01")).rule == "cas_period_order"   # same day
    issue = rules.cas_period_order(cas(period_end="2024-09-30"))
    assert issue.actual == "2024-09-30" and issue.difference < 0


@pytest.mark.parametrize("start,end,ok", [
    ("2025-07-01", "2025-09-30", True),    # 92 days
    ("2025-01-01", "2025-03-31", True),    # 90 days
    ("2025-01-01", "2025-03-21", True),    # 80 days: lower edge
    ("2025-01-01", "2025-03-20", False),   # 79 days
    ("2025-07-01", "2025-10-08", True),    # 100 days: upper edge
    ("2025-07-01", "2025-10-09", False),   # 101 days
    ("2024-07-01", "2025-09-30", False),   # start a year early (a fault the injector makes)
])
def test_period_span(start, end, ok):
    assert (rules.cas_period_span(cas(period_start=start, period_end=end)) is None) is ok


def test_span_leaves_reversed_dates_to_order_check():
    assert rules.cas_period_span(cas(period_end="2024-09-30")) is None


@pytest.mark.parametrize("start,end,ok", [
    ("2025-07-01", "2025-09-30", True),
    ("2024-10-01", "2024-12-31", True),
    ("2025-07-15", "2025-10-14", False),   # about a quarter long, but not a calendar quarter
    ("2025-07-01", "2025-09-29", False),
])
def test_period_matches_a_calendar_quarter(start, end, ok):
    issue = rules.cas_period_quarter(cas(period_start=start, period_end=end))
    assert (issue is None) is ok
    if issue:
        assert issue.expected.startswith("2025-07-01 to ")


@pytest.mark.parametrize("label,ok", [("Q3 2025", True), ("Q1 2024", True), ("Q5 2025", False),
                                      ("3Q25", False), ("Q3 25", False)])
def test_letter_quarter_label(label, ok):
    assert (rules.letter_quarter_label({"quarter": label}) is None) is ok


def test_date_objects_and_strings_both_work():
    assert rules.cas_period_span(cas(period_start=dt.date(2025, 7, 1), period_end=dt.date(2025, 9, 30))) is None


# --------------------------------------------------------------------------- non-negativity

@pytest.mark.parametrize("field", ["management_fees", "incentive_fees", "total_commitment", "unfunded_commitment"])
def test_cas_non_negative(field):
    assert rules.cas_non_negative(cas(**{field: 0.0})) is None           # zero is fine
    issue = rules.cas_non_negative(cas(**{field: -0.01}))
    assert issue.rule == "cas_non_negative" and field in issue.reason


def test_cas_signed_fields_may_be_negative():
    assert rules.cas_non_negative(cas(net_gain_loss=-5.0)) is None


def test_soi_non_negative_names_the_holding():
    f = soi()
    f["holdings"][1]["cost"] = -10.0
    issue = rules.soi_non_negative(f)
    assert issue and "holdings[B].cost=-10.00" in issue.reason
    assert rules.soi_non_negative(soi(total_fair_value=-1.0)).rule == "soi_non_negative"
    assert rules.soi_non_negative(soi()) is None


# --------------------------------------------------------------------------- fund registry

def test_registry_is_the_generators_fund_list():
    reg = json.loads((ROOT / "data/funds.json").read_text())["funds"]
    assert [x["fund_name"] for x in reg] == [f.name for f in FUNDS]
    assert rules.fund_registry() == tuple(f.name for f in FUNDS)


def test_exact_match_required():
    assert rules.fund_in_registry({"fund_name": FUND}) is None
    assert rules.fund_in_registry({"fund_name": FUND.lower()}) is not None
    assert rules.fund_in_registry({"fund_name": FUND + " "}) is not None


@pytest.mark.parametrize("name", ["Granite Peak Capital Partners III",            # suffix dropped
                                  "Granite Peak Captial Partners III, L.P."])      # letters swapped
def test_near_miss_names_the_closest_fund(name):
    issue = rules.fund_in_registry({"fund_name": name})
    assert issue.reason == "fund name not in registry, closest: Granite Peak Capital Partners III, L.P."
    assert issue.actual == name


def test_unrelated_name_has_no_closest():
    assert rules.fund_in_registry({"fund_name": "Zzyzx Holdings"}).reason == "fund name not in registry"


# --------------------------------------------------------------------------- no clean doc flagged, either split

@pytest.mark.parametrize("split", ["main", "hard"])
def test_no_clean_truth_doc_flagged_by_sanity_checks(split):
    for d in load_truth(split):
        if d["doc_type"] == "UNKNOWN":
            continue
        rec = {"file": d["file"], "route": {"doc_type": d["doc_type"], "confidence": 0.95, "reason": ""},
               "extracted": {**d["fields"], "confidence": 0.95}, "error": None}
        sanity = {r.__name__ for r in rules.SANITY_RULES[d["doc_type"]]}
        flagged = [i.rule for i in validate_record(rec).issues if i.rule in sanity]
        assert flagged == [], (d["doc_id"], flagged)


def test_faults_are_still_chosen_by_arithmetic_rules_only():
    """Adding checks must not change which faults a seeded run injects."""
    from agents.faults import breaks_rule
    f = cas(fund_name="Not A Fund", period_end="2024-09-30")
    assert breaks_rule("CAS", f) == []
