"""Phase 2 unit tests. No network: the LLM client is faked."""
import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.extractors import cas as cas_extractor
from agents.llm_client import LLMClient
from schemas.cas import CAS

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = dict(
    fund_name="F", manager="M", investor_name="I", period_start="2025-01-01", period_end="2025-03-31",
    currency="USD", beginning_nav=100.0, contributions=10.0, distributions=5.0, management_fees=1.0,
    incentive_fees=0.0, net_gain_loss=-4.0, ending_nav=100.0, total_commitment=200.0,
    unfunded_commitment=50.0, confidence=0.9,
)


class FakeClient(LLMClient):
    provider = "fake"

    def __init__(self):
        super().__init__("fake-model")
        self.calls = []

    def extract(self, system_prompt, document_text, pydantic_model):
        self.calls.append((system_prompt, document_text, pydantic_model))
        return pydantic_model.model_validate(SAMPLE)


def test_extractor_sends_versioned_prompt_and_pdf_text():
    client = FakeClient()
    pdf = next(p for p in sorted((ROOT / "data").glob("doc_*.pdf")))
    out = cas_extractor.extract_cas(pdf, client)
    prompt, text, model = client.calls[0]
    assert isinstance(out, CAS) and model is CAS
    assert prompt == (ROOT / "agents/prompts/cas_v1.md").read_text()
    assert "--- page 1 ---" in text and len(text) > 200


def test_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        CAS.model_validate({**SAMPLE, "confidence": 1.5})
    with pytest.raises(ValidationError):
        CAS.model_validate({**SAMPLE, "extra_field": 1})
    with pytest.raises(ValidationError):
        CAS.model_validate({k: v for k, v in SAMPLE.items() if k != "ending_nav"})


def test_schema_documents_unit_conventions():
    desc = {k: f.description for k, f in CAS.model_fields.items()}
    assert "positive magnitude" in desc["management_fees"]
    assert "negative" in desc["net_gain_loss"]
    assert "15090000" in desc["ending_nav"]


def _imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_provider_sdk_only_in_llm_client():
    offenders = [p for p in ROOT.rglob("*.py")
                 if ".venv" not in p.parts and p.name != "llm_client.py" and "anthropic" in _imports(p)]
    assert offenders == []


def test_extractors_never_touch_ground_truth():
    for p in (ROOT / "agents").rglob("*.py"):
        assert "ground_truth" not in p.read_text(), p
