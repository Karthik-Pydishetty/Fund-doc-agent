"""Phase 4: every finance rule, at and just past its tolerance edge, plus the non-rule exception reasons."""
import csv
import json
from pathlib import Path

import pytest

from evals.exceptions import score
from validation import rules
from validation.validate import ACCEPTED, EXCEPTION, validate_record, write_exceptions_report

ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]


# --------------------------------------------------------------------------- fixtures

def cas(**kw):
    """Full-unit CAS (1,000,003 pins the printed step to 1). Rolls forward exactly:
    1,000,003 + 100,000 - 50,000 - 10,000 - 20,000 - 20,000 = 1,000,003."""
    f = dict(fund_name="F", manager="M", investor_name="I", period_start="2025-01-01", period_end="2025-03-31",
             currency="USD", beginning_nav=1_000_003.0, contributions=100_000.0, distributions=50_000.0,
             management_fees=10_000.0, incentive_fees=20_000.0, net_gain_loss=-20_000.0, ending_nav=1_000_003.0,
             total_commitment=2_000_000.0, unfunded_commitment=500_000.0, confidence=0.95)
    return {**f, **kw}


def cas_scaled(step):
    """Same statement printed in k (step 1,000) or mm (step 10,000): every amount a multiple of step."""
    f = cas(beginning_nav=27_180_000.0, contributions=1_130_000.0, distributions=0.0, management_fees=180_000.0,
            incentive_fees=40_000.0, net_gain_loss=240_000.0, ending_nav=28_330_000.0,
            total_commitment=38_000_000.0, unfunded_commitment=13_020_000.0)
    if step == 1_000:
        f.update(contributions=1_131_000.0, ending_nav=28_331_000.0)  # 1,131k pins the step to 1,000
    return f


def soi(fvs=(600_001.0, 400_000.0), pcts=(60.0, 40.0), total=None, **kw):
    hs = [dict(name=f"H{i}", sector="S", geography="G", cost=fv, fair_value=fv, pct_of_nav=p)
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


# --------------------------------------------------------------------------- printed precision

@pytest.mark.parametrize("values,step", [
    ((1_234_567, 45_000), 1.0),                 # "$1,234,567"
    ((1_235_000, 45_000, 0), 1_000.0),          # "$1,235k" / in thousands
    ((27_180_000, 1_130_000, 180_000), 10_000.0),  # "$27.18mm"
    ((38_000_000, 2_000_000), 10_000.0),        # all round: capped at MAX_STEP
    ((1_234_567.25, 45_000), 0.01),             # cents
    ((0, 0), 1.0),
])
def test_printed_step(values, step):
    assert rules.printed_step(values) == step


# --------------------------------------------------------------------------- CAS rules

def test_cas_rollforward_clean():
    assert rules.cas_rollforward(cas()) is None


@pytest.mark.parametrize("delta,ok", [(7, True), (-7, True), (8, False), (-8, False)])
def test_cas_rollforward_full_unit_edge(delta, ok):
    # step 1 x 7 terms = 7 units
    issue = rules.cas_rollforward(cas(ending_nav=1_000_003.0 + delta))
    assert (issue is None) is ok
    if issue:
        assert issue.expected == 1_000_003 and issue.difference == delta and issue.tolerance == 7


def test_cas_rollforward_60k_error_on_full_unit_doc_is_caught():
    """A 60,000 error on a ~$15mm full-unit statement passed the old 0.5% tolerance (75k); now it fails."""
    f = cas(beginning_nav=15_090_123.0, contributions=0.0, distributions=400_000.0, management_fees=140_000.0,
            incentive_fees=0.0, net_gain_loss=-780_000.0, ending_nav=13_770_123.0,
            total_commitment=29_250_000.0, unfunded_commitment=15_380_000.0)
    assert rules.cas_rollforward(f) is None
    issue = rules.cas_rollforward({**f, "ending_nav": 13_770_123.0 + 60_000})
    assert issue is not None
    assert issue.difference == 60_000 and issue.tolerance == 7


@pytest.mark.parametrize("step", [1_000, 10_000])
def test_cas_rollforward_scaled_edge(step):
    # k / mm statements: tolerance = step x 7 terms; edges stay on the printed grid so the step is unchanged
    f = cas_scaled(step)
    assert rules.cas_step(f) == step
    assert rules.cas_rollforward(f) is None
    assert rules.cas_rollforward({**f, "ending_nav": f["ending_nav"] + 7 * step}) is None
    assert rules.cas_rollforward({**f, "ending_nav": f["ending_nav"] - 7 * step}) is None
    issue = rules.cas_rollforward({**f, "ending_nav": f["ending_nav"] + 8 * step})
    assert issue and issue.tolerance == 7 * step


def test_cas_rollforward_mm_rounding_passes():
    # each of 7 terms printed to the nearest 10k is off by <= 5k: worst case 35k < 70k tolerance
    f = cas_scaled(10_000)
    assert rules.cas_rollforward({**f, "ending_nav": f["ending_nav"] + 30_000}) is None


def test_cas_rollforward_detects_doc021_magnitude():
    # doc_021 (printed "€x.xx million"): ending 14,140,000 vs roll-forward 13,770,000; tol 70,000
    f = cas(beginning_nav=15_090_000, contributions=0, distributions=400_000, management_fees=140_000,
            incentive_fees=0, net_gain_loss=-780_000, ending_nav=14_140_000,
            total_commitment=29_250_000, unfunded_commitment=15_380_000)
    issue = rules.cas_rollforward(f)
    assert issue.expected == 13_770_000 and issue.actual == 14_140_000 and issue.difference == 370_000
    assert issue.tolerance == 70_000


@pytest.mark.parametrize("excess,ok", [(-1.0, True), (0.0, True), (1.0, True), (1.01, False), (10_000.0, False)])
def test_cas_unfunded_strict_edge(excess, ok):
    # $1 tolerance regardless of print precision, even on an mm statement
    for f in (cas(), cas_scaled(10_000)):
        issue = rules.cas_unfunded_le_commitment({**f, "unfunded_commitment": f["total_commitment"] + excess})
        assert (issue is None) is ok
        if issue:
            assert issue.tolerance == 1.0


# --------------------------------------------------------------------------- SOI rules

@pytest.mark.parametrize("delta,ok", [(0, True), (3, True), (-3, True), (4, False), (-4, False)])
def test_soi_fair_value_sum_full_unit_edge(delta, ok):
    # holdings sum 1,000,001 at step 1; 2 holdings + total = 3 terms -> tol 3
    assert (rules.soi_fair_value_sum(soi(total=1_000_001.0 + delta)) is None) is ok


@pytest.mark.parametrize("delta,ok", [(3_000, True), (-3_000, True), (4_000, False), (-4_000, False)])
def test_soi_fair_value_sum_thousands_edge(delta, ok):
    # "(in thousands)": step 1,000 x 3 terms = 3,000
    f = soi(fvs=(36_957_000.0, 22_835_000.0), total=59_792_000.0 + delta)
    assert rules.soi_step(f) == 1_000
    assert (rules.soi_fair_value_sum(f) is None) is ok


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
    res = validate_record(rec("CAS", cas(ending_nav=1_100_003.0, confidence=0.95)))
    assert res.status == EXCEPTION and _rules(res) == ["cas_rollforward"]


def test_multiple_reasons_each_get_a_row(tmp_path):
    res = validate_record(rec("CAS", cas(ending_nav=1_100_003.0, unfunded_commitment=3_000_000.0, confidence=0.5)))
    assert _rules(res) == ["cas_rollforward", "cas_unfunded_le_commitment", "low_confidence"]
    out = tmp_path / "x.csv"
    assert write_exceptions_report([res, validate_record(rec("CAS", cas()))], out) == 3
    rows = list(csv.DictReader(out.open()))
    assert [r["rule"] for r in rows] == _rules(res)
    r0 = rows[0]
    assert r0["doc_id"] == "doc_999" and r0["doc_type"] == "CAS" and r0["confidence"] == "0.5"
    assert float(r0["expected"]) == 1_000_003 and float(r0["actual"]) == 1_100_003
    assert float(r0["difference"]) == 100_000 and float(r0["tolerance"]) == 7


# --------------------------------------------------------------------------- ground truth sanity

def test_rules_pass_every_clean_truth_doc_and_fail_every_broken_one():
    """Feeding ground truth straight through validation must reproduce expected_exception exactly."""
    recs = [rec(d["doc_type"], {**d["fields"], "confidence": 0.95} if d["fields"] else None, file=d["file"])
            for d in TRUTH]
    results = [validate_record(r) for r in recs]
    s = score(results, TRUTH)
    assert s["fp"] == [] and s["fn"] == [] and s["precision"] == s["recall"] == 1.0
    assert sorted(s["unknown_flagged"]) == sorted(s["unknown"])


def test_inferred_step_matches_each_truth_docs_print_format():
    """The step validation infers from values alone agrees with how the generator printed each doc."""
    printed = {"full": 1, "plain": 1, "decimals": 1, "k": 1_000, "mm": 10_000, "words": 10_000, "thousands": 1_000}
    for d in TRUTH:
        if d["doc_type"] == "CAS":
            assert rules.cas_step(d["fields"]) == printed[d["number_format"]], d["doc_id"]
        elif d["doc_type"] == "SOI":
            assert rules.soi_step(d["fields"]) == printed[d["number_format"]], d["doc_id"]
