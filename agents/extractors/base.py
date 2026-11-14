"""Shared extractor plumbing: a versioned prompt + a pydantic schema + the LLM client.

Extractors see only document text; the eval answer key is off limits to everything under agents/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from agents.llm_client import LLMClient
from agents.pdf_text import pdf_to_text

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
T = TypeVar("T", bound=BaseModel)


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text()


@dataclass(frozen=True)
class Extractor(Generic[T]):
    doc_type: str
    prompt_version: str
    schema: type[T]

    def extract_text(self, text: str, client: LLMClient) -> T:
        return client.extract(load_prompt(self.prompt_version), text, self.schema)

    def extract(self, pdf_path: str | Path, client: LLMClient) -> T:
        return self.extract_text(pdf_to_text(pdf_path), client)
