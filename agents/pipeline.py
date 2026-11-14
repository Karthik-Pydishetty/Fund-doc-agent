"""PDF -> text -> router -> specialist extractor. Validation (Phase 4) plugs in after extraction."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from agents import router
from agents.extractors import cas, letter, soi
from agents.extractors.base import Extractor
from agents.llm_client import ExtractionError, LLMClient
from agents.pdf_text import pdf_to_text

EXTRACTORS: dict[str, Extractor] = {e.doc_type: e for e in (cas.EXTRACTOR, soi.EXTRACTOR, letter.EXTRACTOR)}


def prompt_versions() -> dict[str, str]:
    return {"router": router.PROMPT_VERSION, **{t.lower(): e.prompt_version for t, e in EXTRACTORS.items()}}


def process(pdf_path: str | Path, client: LLMClient) -> dict:
    """Route and extract one document. Never raises on model failures; they land in `error`."""
    pdf_path = Path(pdf_path)
    rec: dict = {"file": pdf_path.name, "route": None, "extracted": None, "error": None, "calls": {}}
    text = pdf_to_text(pdf_path)

    try:
        decision = router.route_text(text, client)
    except ExtractionError as e:
        rec["error"] = f"router: {e}"
        return rec
    finally:
        if client.last_call:
            rec["calls"]["router"] = asdict(client.last_call)
    rec["route"] = decision.model_dump(mode="json")

    extractor = EXTRACTORS.get(decision.doc_type)
    if extractor is None:  # UNKNOWN: nothing to extract
        return rec
    try:
        rec["extracted"] = extractor.extract_text(text, client).model_dump(mode="json")
    except ExtractionError as e:
        rec["error"] = f"{decision.doc_type.lower()} extractor: {e}"
    finally:
        if client.last_call:
            rec["calls"]["extractor"] = asdict(client.last_call)
    return rec
