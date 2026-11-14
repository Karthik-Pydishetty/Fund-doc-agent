"""Router: classify a document as CAS | SOI | LETTER | UNKNOWN."""
from __future__ import annotations

from agents.extractors.base import load_prompt
from agents.llm_client import LLMClient
from schemas.router import RouteDecision

PROMPT_VERSION = "router_v1"


def route_text(text: str, client: LLMClient, prompt_version: str = PROMPT_VERSION) -> RouteDecision:
    return client.extract(load_prompt(prompt_version), text, RouteDecision)
