"""Phase 4: every finance rule, at and just past its tolerance edge, plus the non-rule exception reasons."""
import csv
import json
from pathlib import Path

import pytest

from evals.exceptions import score
from validation import rules
from validation.rules import PCT_TOL, REL_TOL
from validation.validate import ACCEPTED, EXCEPTION, validate_record, write_exceptions_report

ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]
STEP = 0.01  # "just past the edge"


# --------------------------------------------------------------------------- fixtures

def cas(**kw):
    # rolls forward exactly: 1,000,000 + 100,000 - 50,000 - 10,000 - 20,000 - 20,000 = 1,000,000
    f = dict(fund_name="F", manager="M", investor_name="I", period_start="2025-01-01", period_end="2025-03-31",
             currency="USD", beginning_nav=1_000_000.0, contributions=100_000.0, distributions=50_000.0,
             management_fees=10_000.0, incentive_fees=20_000.0, net_gain_loss=-20_000.0, ending_nav=1_000_000.0,
             total_commitment=2_000_000.0, unfunded_commitment=500_000.0, confidence=0.95)
    return {**f, **kw}


def soi(fvs=(600_000.0, 400_000.0), pcts=(60.0, 40.0), total=None, **kw):
    hs = [dict(name=f"H{i}", sector="S", geography="G", cost=1.0, fair_value=fv, pct_of_nav=p)
          for i, (fv, p) in enumerate(zip(fvs, pcts))]
    return dict(fund_name="F", as_of_date="2025-03-31", holdings=hs,
                total_fair_value=sum(fvs) if total is None else total, confidence=0.95, **kw)


def letter(long_=140.0, short=60.0, gross=200.0, net=80.0, **kw):
    f = dict(fund_name="F", quarter="Q1 2025", long_exposure=long_, short_exposure=short, gross_exposure=gross,
             net_exposure=net, top_sector_exposures=[], top_holdings=[], quarterly_return=1.0,
             commentary_summary="c", confidence=0.95)
    return {**f, **kw}


def rec(doc_type, extracted, route_conf=0.95, error=None, file="doc_999.pdf"):
    return {"file": file, "route": {"doc_type": doc_type, "reason": "r", "confidence": route_conf},
            "extracted": extracted, "error": error}


# --------------------------------------------------------------------------- CAS rules

def test_cas_rollforward_clean():
    assert rules.cas_rollforward(cas()) is None


@pytest.mark.parametrize("ending,ok", [
    # roll-forward = 1,000,000; tol = 0.5% of max(|beginning|, |ending|)
    (995_000.0, True),           # diff -5,000; ref 1,000,000 -> tol 5,000 (exact edge)
    (995_000.0 - STEP, False),   # just past
    (1_005_025.0, True),         # diff +5,025; ref 1,005,025 -> tol 5,025.125
    (1_005_026.0, False),        # diff +5,026; tol 5,025.13
])
def test_cas_rollforward_tolerance_edge(ending, ok):
    issue = rules.cas_rollforward(cas(ending_nav=ending))
    assert (issue is None) is ok
    if issue:
        assert issue.rule == "cas_rollforward" and issue.expected == 1_000_000 and issue.actual == ending


def test_cas_rollforward_mm_rounding_passes():
    # every component printed to the nearest 10k ("$x.xxmm"): 7 roundings of up to 5k each still pass
    assert rules.cas_rollforward(cas(beginning_nav=27_180_000, contributions=1_130_000, distributions=0,
                                     management_fees=180_000, incentive_fees=40_000, net_gain_loss=240_000,
                                     ending_nav=28_330_000 + 35_000)) is None


def test_cas_rollforward_detects_doc021_magnitude():
    # doc_021: printed ending 14,140,000 vs roll-forward 13,770,000 (diff 370,000 >> tol 70,700)
    f = cas(beginning_nav=15_090_000, contributions=0, distributions=400_000, management_fees=140_000,
            incentive_fees=0, net_gain_loss=-780_000, ending_nav=14_140_000)
    issue = rules.cas_rollforward(f)
    assert issue.expected == 13_770_000 and issue.actual == 14_140_000 and issue.difference == 370_000


def test_cas_unfunded_edge():
    tol = REL_TOL * 2_000_000  # 10,000
    assert rules.cas_unfunded_le_commitment(cas(unfunded_commitment=2_000_000.0)) is None
    assert rules.cas_unfunded_le_commitment(cas(unfunded_commitment=2_000_000.0 + tol)) is None
    issue = rules.cas_unfunded_le_commitment(cas(unfunded_commitment=2_000_000.0 + tol + STEP))
    assert issue and issue.rule == "cas_unfunded_le_commitment"


# --------------------------------------------------------------------------- SOI rules

def test_soi_fair_value_sum_edge():
    # holdings sum to 1,000,000; reference = max(total, sum)
    assert rules.soi_fair_value_sum(soi()) is None
    assert rules.soi_fair_value_sum(soi(total=995_000.0)) is None            # diff 5,000 = 0.5% of 1,000,000
    assert rules.soi_fair_value_sum(soi(total=995_000.0 - STEP)) is not None
    assert rules.soi_fair_value_sum(soi(total=1_005_025.0)) is None          # ref 1,005,025 -> tol 5,025.125
    assert rules.soi_fair_value_sum(soi(total=1_005_026.0)) is not None


@pytest.mark.parametrize("pcts,ok", [
    ((60.0, 40.0), True),
    ((60.0, 40.5), True), ((60.0, 39.5), True),        # exactly +/-0.5
    ((60.0, 40.51), False), ((60.0, 39.49), False),    # just past
    ((33.33, 33.33, 33.33), True),                     # rounding noise, 99.99
])
def test_soi_pct_sum_edge(pcts, ok):
    f = soi(fvs=tuple(1.0 for _ in pcts), pcts=pcts)
    assert (rules.soi_pct_sum(f) is None) is ok


# --------------------------------------------------------------------------- LETTER rules

@pytest.mark.parametrize("gross,ok", [(200.0, True), (200.5, True), (199.5, True), (200.51, False), (199.49, False)])
def test_letter_gross_edge(gross, ok):
    assert (rules.letter_gross(letter(gross=gross)) is None) is ok


@pytest.mark.parametrize("net,ok", [(80.0, True), (80.5, True), (79.5, True), (80.51, False), (79.49, False)])
def test_letter_net_edge(net, ok):
    assert (rules.letter_net(letter(net=net)) is None) is ok


def test_letter_one_decimal_rounding_passes():
    # long 119.2 + short 37.3 printed as gross 156.5: float sum is 156.50000000000003
    assert rules.letter_gross(letter(long_=119.2, short=37.3, gross=156.5, net=81.9)) is None
    assert rules.letter_net(letter(long_=119.2, short=37.3, gross=156.5, net=81.9)) is None


# --------------------------------------------------------------------------- non-rule exception reasons

def _rules(res):
    return [i.rule for i in res.issues]


def test_clean_docs_accepted():
    for t, f in (("CAS", cas()), ("SOI", soi()), ("LETTER", letter())):
        assert validate_record(rec(t, f)).status == ACCEPTED, t


def test_unknown_goes_to_exceptions():
    res = validate_record(rec("UNKNOWN", None))
    assert res.status == EXCEPTION and _rules(res) == ["unknown_doc_type"]


def test_extraction_error_goes_to_exceptions():
    res = validate_record(rec("CAS", None, error="cas extractor: no structured output"))
    assert _rules(res) == ["extraction_error"]
    res = validate_record({"file": "doc_9.pdf", "route": None, "extracted": None, "error": "router: boom"})
    assert _rules(res) == ["extraction_error"] and res.doc_type is None


@pytest.mark.parametrize("conf,flagged", [(0.7, False), (0.69, True), (0.95, False)])
def test_low_confidence_threshold(conf, flagged):
    res = validate_record(rec("CAS", cas(confidence=conf)))
    assert ("low_confidence" in _rules(res)) is flagged


def test_low_router_confidence():
    assert _rules(validate_record(rec("CAS", cas(), route_conf=0.5))) == ["low_router_confidence"]


def test_schema_invalid_goes_to_exceptions():
    bad = cas()
    del bad["ending_nav"]
    assert _rules(validate_record(rec("CAS", bad))) == ["schema_invalid"]


def test_rule_failure_flagged_despite_high_confidence():
    res = validate_record(rec("CAS", cas(ending_nav=1_100_000.0, confidence=0.95)))
    assert res.status == EXCEPTION and _rules(res) == ["cas_rollforward"]


def test_multiple_reasons_each_get_a_row(tmp_path):
    res = validate_record(rec("CAS", cas(ending_nav=1_100_000.0, unfunded_commitment=3_000_000.0, confidence=0.5)))
    assert _rules(res) == ["cas_rollforward", "cas_unfunded_le_commitment", "low_confidence"]
    out = tmp_path / "x.csv"
    assert write_exceptions_report([res, validate_record(rec("CAS", cas()))], out) == 3
    rows = list(csv.DictReader(out.open()))
    assert [r["rule"] for r in rows] == _rules(res)
    r0 = rows[0]
    assert r0["doc_id"] == "doc_999" and r0["doc_type"] == "CAS" and r0["confidence"] == "0.5"
    assert float(r0["expected"]) == 1_000_000 and float(r0["actual"]) == 1_100_000
    assert float(r0["difference"]) == 100_000 and r0["tolerance"]


# --------------------------------------------------------------------------- ground truth sanity

def test_rules_pass_every_clean_truth_doc_and_fail_every_broken_one():
    """Feeding ground truth straight through validation must reproduce expected_exception exactly."""
    recs = [rec(d["doc_type"], {**d["fields"], "confidence": 0.95} if d["fields"] else None, file=d["file"])
            for d in TRUTH]
    results = [validate_record(r) for r in recs]
    s = score(results, TRUTH)
    assert s["fp"] == [] and s["fn"] == [] and s["precision"] == s["recall"] == 1.0
    assert sorted(s["unknown_flagged"]) == sorted(s["unknown"])
