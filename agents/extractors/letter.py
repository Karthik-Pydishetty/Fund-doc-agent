"""LETTER extractor: PDF -> text -> LLM structured output -> validated Letter model."""
from __future__ import annotations

from pathlib import Path

from agents.extractors.base import Extractor
from agents.llm_client import LLMClient
from schemas.letter import Letter

PROMPT_VERSION = "letter_v1"
EXTRACTOR = Extractor("LETTER", PROMPT_VERSION, Letter)


def extract_letter(pdf_path: str | Path, client: LLMClient) -> Letter:
    return EXTRACTOR.extract(pdf_path, client)
