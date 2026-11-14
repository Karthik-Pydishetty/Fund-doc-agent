"""CAS extractor: PDF -> text -> LLM structured output -> validated CAS model."""
from __future__ import annotations

from pathlib import Path

from agents.extractors.base import Extractor, load_prompt as _load
from agents.llm_client import LLMClient
from schemas.cas import CAS

PROMPT_VERSION = "cas_v1"
EXTRACTOR = Extractor("CAS", PROMPT_VERSION, CAS)


def load_prompt() -> str:
    return _load(PROMPT_VERSION)


def extract_cas(pdf_path: str | Path, client: LLMClient) -> CAS:
    return EXTRACTOR.extract(pdf_path, client)
