"""One uploaded PDF through the existing pipeline: route -> extract -> validate -> self-correct."""
from __future__ import annotations

from pathlib import Path

from agents.llm_client import LLMClient
from agents.pdf_text import pdf_to_text
from agents.pipeline import process
from agents.self_correct import finance_issues, self_correct
from validation.validate import validate_record


def process_document(pdf_path: Path, client: LLMClient, prompts: dict[str, str]) -> dict:
    rec = process(pdf_path, client, prompts)
    v = validate_record(rec)
    rec["validation"] = {"status": v.status, "issues": [i.to_dict() for i in v.issues]}
    if finance_issues(rec):  # same rule as main.py --self-correct: one retry, quotes checked in code
        self_correct(rec, pdf_to_text(pdf_path), client, prompts[rec["route"]["doc_type"].lower()])
    return rec
