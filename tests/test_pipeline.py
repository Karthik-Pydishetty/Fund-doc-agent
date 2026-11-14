"""Phase 3 offline tests: router + pipeline wiring with a fake client, and the all-docs scorer."""
import json
from pathlib import Path

import pytest

from agents import pipeline
from agents.llm_client import ExtractionError, LLMClient
from evals import compare
from schemas.letter import Letter
from schemas.router import RouteDecision
from schemas.soi import SOI

ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]
BY_TYPE = {t: next(d for d in TRUTH if d["doc_type"] == t) for t in ("CAS", "SOI", "LETTER", "UNKNOWN")}


class ScriptedClient(LLMClient):
    """Routes to a fixed type and returns that type's ground-truth fields (or raises)."""

    provider = "fake"

    def __init__(self, route: str, fields: dict | None = None, fail: str | None = None):
        super().__init__("fake-model")
        self.route, self.fields, self.fail, self.prompts = route, fields or {}, fail, []

    def extract(self, system_prompt, document_text, pydantic_model):
        self.prompts.append(system_prompt)
        if pydantic_model is RouteDecision:
            if self.fail == "router":
                raise ExtractionError("boom")
            return RouteDecision(doc_type=self.route, reason="r", confidence=0.9)
        if self.fail == "extractor":
            raise ExtractionError("bad output")
        return pydantic_model.model_validate({**self.fields, "confidence": 0.9})


@pytest.mark.parametrize("doc_type", ["CAS", "SOI", "LETTER"])
def test_pipeline_routes_to_matching_extractor(doc_type):
    d = BY_TYPE[doc_type]
    client = ScriptedClient(doc_type, d["fields"])
    rec = pipeline.process(ROOT / "data" / d["file"], client)
    assert rec["route"]["doc_type"] == doc_type and rec["error"] is None
    assert rec["extracted"]["fund_name"] == d["fields"]["fund_name"]
    version = pipeline.EXTRACTORS[doc_type].prompt_version
    assert client.prompts[1] == (ROOT / f"agents/prompts/{version}.md").read_text()


def test_unknown_is_not_extracted():
    rec = pipeline.process(ROOT / "data" / BY_TYPE["UNKNOWN"]["file"], ScriptedClient("UNKNOWN"))
    assert rec["route"]["doc_type"] == "UNKNOWN" and rec["extracted"] is None and rec["error"] is None


@pytest.mark.parametrize("stage", ["router", "extractor"])
def test_model_failures_are_recorded_not_raised(stage):
    d = BY_TYPE["CAS"]
    rec = pipeline.process(ROOT / "data" / d["file"], ScriptedClient("CAS", d["fields"], fail=stage))
    assert rec["error"].startswith("router" if stage == "router" else "cas extractor")
    assert rec["extracted"] is None


def test_schemas_accept_ground_truth_shapes():
    SOI.model_validate({**BY_TYPE["SOI"]["fields"], "confidence": 1})
    Letter.model_validate({**BY_TYPE["LETTER"]["fields"], "confidence": 1})


def _rec(d, extracted, route=None):
    return {"file": d["file"], "route": {"doc_type": route or d["doc_type"], "reason": ""}, "extracted": extracted}


def test_scorer_matches_lists_by_key_and_flags_errors():
    d = BY_TYPE["SOI"]
    f = json.loads(json.dumps(d["fields"]))
    perfect = list(compare.score_doc(d, _rec(d, f)))
    assert all(ok for *_, ok in perfect)

    f["holdings"] = list(reversed(f["holdings"]))           # order must not matter
    assert all(ok for *_, ok in compare.score_doc(d, _rec(d, f)))

    f["holdings"][0]["pct_of_nav"] += 0.1                     # outside PCT_TOL
    dropped = f["holdings"].pop()                             # missing holding
    f["holdings"].append({**dropped, "name": "Invented Co"})  # extra holding
    bad = {(g, ok) for _, g, *_, ok in compare.score_doc(d, _rec(d, f))}
    assert ("holdings.pct_of_nav", False) in bad
    assert ("holdings[present]", False) in bad
    assert ("extra item", False) in bad


def test_scorer_fails_every_field_when_misrouted():
    d = BY_TYPE["LETTER"]
    rows = list(compare.score_doc(d, _rec(d, d["fields"], route="CAS")))
    assert rows and not any(ok for *_, ok in rows)


def test_commentary_summary_not_scored():
    d = BY_TYPE["LETTER"]
    assert not any("commentary" in k for k, *_ in compare.score_doc(d, _rec(d, d["fields"])))
