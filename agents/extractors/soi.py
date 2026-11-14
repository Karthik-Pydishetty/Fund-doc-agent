"""SOI extractor: PDF -> text -> LLM structured output -> validated SOI model."""
from __future__ import annotations

from pathlib import Path

from agents.extractors.base import Extractor
from agents.llm_client import LLMClient
from schemas.soi import SOI

PROMPT_VERSION = "soi_v1"
EXTRACTOR = Extractor("SOI", PROMPT_VERSION, SOI)


def extract_soi(pdf_path: str | Path, client: LLMClient) -> SOI:
    return EXTRACTOR.extract(pdf_path, client)
