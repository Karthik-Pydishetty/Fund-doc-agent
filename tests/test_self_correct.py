"""Phase 9: self-correction loop and fault injection, with a fake LLM client (no API calls)."""
import copy
import json
import random
from pathlib import Path

import pytest

from agents import faults
from agents.llm_client import ExtractionError, LLMClient
from agents.pdf_text import pdf_to_text
from agents.self_correct import (
    MAX_RETRIES, check_quote, flatten, review_schema, self_correct,
)
from evals import regression
from schemas.cas import CAS
from schemas.router import RouteDecision
from validation.validate import validate_record

ROOT = Path(__file__).resolve().parents[1]
TRUTH = {d["doc_id"]: d for d in json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]}
RETRY_RULES_TEXT = "Never change a value to make the numbers add up"


def text_of(doc_id):
    return pdf_to_text(ROOT / "data" / TRUTH[doc_id]["file"])


def first_pass(doc_id, **overrides):
    d = TRUTH[doc_id]
    rec = {"file": d["file"], "route": {"doc_type": d["doc_type"], "reason": "", "confidence": 0.95},
           "extracted": {**copy.deepcopy(d["fields"]), "confidence": 0.95, **overrides}, "error": None, "calls": {}}
    v = validate_record(rec)
    rec["validation"] = {"status": v.status, "issues": [i.to_dict() for i in v.issues]}
    return rec


def line_with(text, value):
    return next(line for line in text.splitlines() if f"{int(value):,}" in line)


class ReviewClient(LLMClient):
    """Returns a scripted review: review_fn(first_pass_fields) -> (extraction dict, [(field, quote)])."""

    provider = "fake"

    def __init__(self, review_fn=None, fail=False):
        super().__init__("fake-model")
        self.review_fn, self.fail, self.calls = review_fn, fail, []

    def extract(self, system_prompt, document_text, pydantic_model):
        self.calls.append((system_prompt, document_text, pydantic_model))
        if self.fail:
            raise ExtractionError("retry blew up")
        extraction, changes = self.review_fn()
        return pydantic_model.model_validate(
            {"extraction": extraction, "changes": [{"field": f, "quote": q} for f, q in changes]})


# --------------------------------------------------------------------------- trigger

def test_no_retry_without_a_finance_rule_failure():
    client = ReviewClient(lambda: pytest.fail("should not be called"))
    for rec in (first_pass("doc_008"), first_pass("doc_008", confidence=0.3)):   # clean, then low confidence only
        self_correct(rec, text_of("doc_008"), client, "cas_v1")
        assert "self_correction" not in rec
    assert client.calls == []


# --------------------------------------------------------------------------- accepted correction

def test_misread_corrected_with_verbatim_quote_is_accepted():
    truth = TRUTH["doc_008"]["fields"]
    rec = first_pass("doc_008", ending_nav=1_130_325.0)  # dropped digit -> roll-forward fails
    assert rec["validation"]["status"] == "EXCEPTION"
    text = text_of("doc_008")
    fixed = {**rec["extracted"], "ending_nav": float(truth["ending_nav"])}
    client = ReviewClient(lambda: (fixed, [("ending_nav", line_with(text, truth["ending_nav"]))]))
    self_correct(rec, text, client, "cas_v1")
    sc = rec["self_correction"]
    assert rec["extracted"]["ending_nav"] == truth["ending_nav"]
    assert rec["validation"]["status"] == "ACCEPTED" and sc["status_after"] == "ACCEPTED"
    assert sc["changes"] == [{"field": "ending_nav", "old": 1_130_325.0, "new": float(truth["ending_nav"]),
                              "quote": line_with(text, truth["ending_nav"]), "accepted": True,
                              "reason": "quote shows value"}]
    assert len(client.calls) == MAX_RETRIES == 1
    assert sc["triggered_by"] == ["cas_rollforward"] and sc["attempts"] == 1 and sc["status_before"] == "EXCEPTION"


# --------------------------------------------------------------------------- fudges rejected

def test_fudge_on_genuinely_broken_doc_is_rejected_and_doc_stays_exception():
    """doc_021 prints 14.14 million; the roll-forward gives 13.77 million. 'Fixing' it needs a quote
    for a number the document doesn't contain, so the change is reverted."""
    rec = first_pass("doc_021")
    assert [i["rule"] for i in rec["validation"]["issues"]] == ["cas_rollforward"]
    text = text_of("doc_021")
    fudged = {**rec["extracted"], "ending_nav": 13_770_000.0}
    ending_line = next(line for line in text.splitlines() if "14.14" in line)
    client = ReviewClient(lambda: (fudged, [("ending_nav", ending_line)]))
    self_correct(rec, text, client, "cas_v1")
    ch = rec["self_correction"]["changes"][0]
    assert ch["accepted"] is False and ch["reason"] == "quote does not show the new value"
    assert rec["extracted"]["ending_nav"] == 14_140_000
    assert rec["validation"]["status"] == "EXCEPTION"


def test_non_verbatim_quote_is_rejected():
    truth = TRUTH["doc_008"]["fields"]
    rec = first_pass("doc_008", ending_nav=1_130_325.0)
    invented = f"Ending capital was ${int(truth['ending_nav']):,} for the quarter"
    client = ReviewClient(lambda: ({**rec["extracted"], "ending_nav": float(truth["ending_nav"])},
                                   [("ending_nav", invented)]))
    self_correct(rec, text_of("doc_008"), client, "cas_v1")
    assert rec["self_correction"]["changes"][0]["reason"] == "quote not found verbatim in document"
    assert rec["extracted"]["ending_nav"] == 1_130_325.0


def test_change_without_a_changes_entry_is_rejected():
    rec = first_pass("doc_021")
    client = ReviewClient(lambda: ({**rec["extracted"], "management_fees": 510_000.0}, []))
    self_correct(rec, text_of("doc_021"), client, "cas_v1")
    assert rec["self_correction"]["changes"][0]["reason"] == "no quote"
    assert rec["extracted"]["management_fees"] == 140_000


def test_structural_change_is_rejected():
    rec = first_pass("doc_014")  # broken SOI (fair values don't sum to total)
    shorter = copy.deepcopy(rec["extracted"])
    dropped = shorter["holdings"].pop()
    client = ReviewClient(lambda: (shorter, []))
    self_correct(rec, pdf_to_text(ROOT / "data/doc_014.pdf"), client, "soi_v1")
    assert any(dropped["name"] in p for p in rec["self_correction"]["structural_change_rejected"])
    assert len(rec["extracted"]["holdings"]) == len(TRUTH["doc_014"]["fields"]["holdings"])


# --------------------------------------------------------------------------- bounds and logging

def test_retry_error_keeps_first_pass_and_is_logged():
    rec = first_pass("doc_021")
    before = copy.deepcopy(rec["extracted"])
    self_correct(rec, text_of("doc_021"), ReviewClient(fail=True), "cas_v1")
    assert rec["self_correction"]["error"] == "retry blew up"
    assert rec["extracted"] == before and rec["validation"]["status"] == "EXCEPTION"


def test_retry_prompt_carries_findings_rules_and_paths():
    rec = first_pass("doc_014")
    client = ReviewClient(lambda: (rec["extracted"], []))
    self_correct(rec, pdf_to_text(ROOT / "data/doc_014.pdf"), client, "soi_v1")
    system, document, model = client.calls[0]
    assert RETRY_RULES_TEXT in system and "soi_fair_value_sum" in system
    assert "holdings[" in system and "].fair_value:" in system
    assert model.__name__ == "SOIReview" and "Schedule of Investments" in document


def test_review_schema_wraps_extraction():
    s = review_schema(CAS)
    assert set(s.model_fields) == {"extraction", "changes"}


@pytest.mark.parametrize("quote,value,ok", [
    ("Ending capital $17, 176,7O9", 17_176_709, True),       # OCR: O for 0, stray space
    ("Ending capital $17, 176,7O9", 1_716_709, False),
    ("Opening capital 6,592", 6_592_000, True),              # in $000s
    ("€14.14 million", 14_140_000, True),
    ("€14.14 million", 14_000_000, False),                   # half the printed precision is 5,000
    ("Short (77.9)%", 77.9, True),
    ("07/O1/2025", "2025-07-01", True),
    ("07/O1/2025", "2024-07-01", False),
])
def test_quote_must_show_the_new_value(quote, value, ok):
    doc = "Ending capital $17, 176,7O9\nOpening capital 6,592\n€14.14 million\nShort (77.9)%\nPeriod: 07/O1/2025"
    assert check_quote(quote, value, doc)[0] is ok


# --------------------------------------------------------------------------- fault injection

def _clean_recs():
    return [first_pass(i) for i in ("doc_002", "doc_008", "doc_012", "doc_003", "doc_001", "doc_007", "doc_021")]


def test_injection_is_seeded_mixed_and_recorded():
    a, b = _clean_recs(), _clean_recs()
    sa, sb = faults.inject_faults(a, 6, seed=3), faults.inject_faults(b, 6, seed=3)
    assert sa == sb and [r.get("fault_injection") for r in a] == [r.get("fault_injection") for r in b]
    injected = [r for r in a if r.get("fault_injection")]
    assert len(injected) == 6
    assert sa["by_category"] == {"covered": 3, "uncovered_text": 2, "uncovered_numeric": 1}
    assert all(r["file"] != "doc_021.pdf" for r in injected)  # broken doc is never "clean"
    for r in injected:
        f = r["fault_injection"]["injected"]
        assert flatten(r["extracted"])[f["field"]] == f["corrupted"]
        assert flatten(r["first_pass_extracted"])[f["field"]] == f["original"]
        failing = [i["rule"] for i in r["validation"]["issues"]]
        if f["category"] == "covered":
            assert f["rules_broken"] and set(f["rules_broken"]) <= set(failing)
        else:
            assert f["rules_broken"] == [] and r["validation"]["status"] == "ACCEPTED"
            assert isinstance(f["original"], str) is (f["category"] == "uncovered_text")


def test_corruption_helpers():
    rng = random.Random(0)
    assert faults.corrupt_number(1_183_581, "sign_flip", rng) == -1_183_581
    assert faults.corrupt_number(120.9, "scale_10x", rng) in (1209.0, 12.09)
    d = faults.corrupt_number(17_176_709, "dropped_digit", rng)
    assert d is not None and len(str(int(d))) == 7
    assert faults.corrupt_date("2025-09-30") == "2024-09-30"
    assert faults.corrupt_quarter("Q1 2025") == "Q4 2024"
    assert faults.corrupt_name("Granite Peak Capital Partners III, L.P.", random.Random(1)) != \
        "Granite Peak Capital Partners III, L.P."


# --------------------------------------------------------------------------- end to end through run_pipeline

class PipelineClient(LLMClient):
    """Routes and extracts from ground truth; reviews return the extraction unchanged."""

    provider = "fake"

    def __init__(self, docs):
        super().__init__("fake-model")
        self.by_text = {pdf_to_text(ROOT / "data" / TRUTH[d]["file"]): TRUTH[d] for d in docs}
        self.reviews = 0

    def extract(self, system_prompt, document_text, pydantic_model):
        d = self.by_text[document_text]
        if pydantic_model is RouteDecision:
            return RouteDecision(doc_type=d["doc_type"], reason="r", confidence=0.95)
        if pydantic_model.__name__.endswith("Review"):
            self.reviews += 1
            return pydantic_model.model_validate({"extraction": {**d["fields"], "confidence": 0.9}, "changes": []})
        return pydantic_model.model_validate({**d["fields"], "confidence": 0.95})


def test_run_pipeline_logs_faults_and_retries(tmp_path):
    from main import run_pipeline
    from agents.pipeline import resolve_prompts

    docs = ["doc_021", "doc_008", "doc_012", "doc_002"]
    client = PipelineClient(docs)
    pdfs = [ROOT / "data" / TRUTH[d]["file"] for d in docs]
    log, header, records = run_pipeline(pdfs, client, resolve_prompts(), workers=2, runs_dir=tmp_path,
                                        report=tmp_path / "x.csv", self_correct=True, inject_faults=2, fault_seed=1)
    assert header["self_correct"] is True and header["inject_faults"] == 2
    assert header["fault_summary"]["injected"] == 2
    by = {r["file"]: r for r in records}
    assert "self_correction" in by["doc_021.pdf"] and by["doc_021.pdf"]["validation"]["status"] == "EXCEPTION"
    faulted = [r for r in records if r.get("fault_injection")]
    covered = [r for r in faulted if r["fault_injection"]["injected"]["category"] == "covered"]
    # the fake review returns the true values but lists no `changes`, so every correction lacks a quote and is reverted
    for r in covered:
        assert r["self_correction"]["changes"] and not any(c["accepted"] for c in r["self_correction"]["changes"])
    assert client.reviews == 1 + len(covered)  # one retry per rule-failing doc, never more
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert logged[0]["fault_summary"] == header["fault_summary"]
    assert regression.latest_run_log(tmp_path, "main") is None            # gate ignores fault runs
    assert regression.latest_run_log(tmp_path, "main", include_fault_injected=True) == log


def test_free_text_summary_rewrite_is_not_a_change():
    rec = first_pass("doc_005")  # broken LETTER (gross != long + short)
    reworded = {**rec["extracted"], "commentary_summary": "Something else entirely."}
    self_correct(rec, pdf_to_text(ROOT / "data/doc_005.pdf"), ReviewClient(lambda: (reworded, [])), "letter_v1")
    assert rec["self_correction"]["changes"] == []
    assert rec["extracted"]["commentary_summary"] == TRUTH["doc_005"]["fields"]["commentary_summary"]
