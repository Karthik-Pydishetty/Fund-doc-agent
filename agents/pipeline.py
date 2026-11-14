"""PDF -> text -> router -> specialist extractor. Validation runs on the records afterwards.

Prompt versions are data, not code: pass `prompts` (from resolve_prompts) to choose e.g. cas_v2
without editing anything. Defaults are each module's PROMPT_VERSION.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from agents import router
from agents.extractors import cas, letter, soi
from agents.extractors.base import PROMPTS_DIR, Extractor
from agents.llm_client import ExtractionError, LLMClient
from agents.pdf_text import pdf_to_text

EXTRACTORS: dict[str, Extractor] = {e.doc_type: e for e in (cas.EXTRACTOR, soi.EXTRACTOR, letter.EXTRACTOR)}
DEFAULT_PROMPTS: dict[str, str] = {"router": router.PROMPT_VERSION,
                                   **{t.lower(): e.prompt_version for t, e in EXTRACTORS.items()}}


def prompt_versions() -> dict[str, str]:
    return dict(DEFAULT_PROMPTS)


def resolve_prompts(overrides: list[str] | dict[str, str] | None = None) -> dict[str, str]:
    """Defaults plus overrides like ["cas=v2"] or {"cas": "cas_v2"}. Fails fast on unknown keys/files."""
    prompts = prompt_versions()
    items = overrides.items() if isinstance(overrides, dict) else (
        tuple(o.split("=", 1)) if "=" in o else (o, None) for o in (overrides or []))
    for key, value in items:
        key = (key or "").strip().lower()
        if key not in prompts or not value:
            raise ValueError(f"bad prompt override {key}={value!r}; use <stage>=<version> with stage in "
                             f"{sorted(prompts)}, e.g. cas=v2")
        version = value.strip() if value.strip().startswith(f"{key}_") else f"{key}_{value.strip()}"
        if not (PROMPTS_DIR / f"{version}.md").exists():
            available = sorted(p.stem for p in PROMPTS_DIR.glob(f"{key}_*.md"))
            raise ValueError(f"no prompt file agents/prompts/{version}.md; available for {key}: {available}")
        prompts[key] = version
    return prompts


def process(pdf_path: str | Path, client: LLMClient, prompts: dict[str, str] | None = None) -> dict:
    """Route and extract one document. Never raises on model failures; they land in `error`."""
    prompts = prompts or prompt_versions()
    pdf_path = Path(pdf_path)
    rec: dict = {"file": pdf_path.name, "route": None, "extracted": None, "error": None, "calls": {}}
    text = pdf_to_text(pdf_path)

    try:
        decision = router.route_text(text, client, prompts["router"])
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
    extractor = replace(extractor, prompt_version=prompts[decision.doc_type.lower()])
    try:
        rec["extracted"] = extractor.extract_text(text, client).model_dump(mode="json")
    except ExtractionError as e:
        rec["error"] = f"{decision.doc_type.lower()} extractor: {e}"
    finally:
        if client.last_call:
            rec["calls"]["extractor"] = asdict(client.last_call)
    return rec
