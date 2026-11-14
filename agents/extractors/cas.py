"""CAS extractor: PDF -> text -> LLM structured output -> validated CAS model.

Sees only the PDF; the eval answer key is off limits to everything under agents/.
"""
from __future__ import annotations

from pathlib import Path

from agents.llm_client import LLMClient
from agents.pdf_text import pdf_to_text
from schemas.cas import CAS

PROMPT_VERSION = "cas_v1"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / f"{PROMPT_VERSION}.md"


def load_prompt() -> str:
    return PROMPT_PATH.read_text()


def extract_cas(pdf_path: str | Path, client: LLMClient) -> CAS:
    return client.extract(load_prompt(), pdf_to_text(pdf_path), CAS)
