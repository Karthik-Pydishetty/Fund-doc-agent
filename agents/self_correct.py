"""Self-correction: one bounded retry when a finance rule fails, with code-checked evidence.

Flow for one record (after first-pass extraction and validation):
  1. Trigger only on finance-rule failures (validation.rules). Low confidence, UNKNOWN, schema and
     extraction errors are not retried: there is no specific finding to give the model.
  2. Re-ask the same extractor once (MAX_RETRIES = 1) with the validator's findings and the
     first-pass values. The retry prompt (prompts/retry_v1.md) forbids changing values to satisfy
     the math and requires a verbatim quote for every changed value.
  3. Code, not the model, decides what is accepted. Every field whose value differs from the
     first pass must have a `changes` entry whose quote (a) appears verbatim in the document text
     (whitespace-normalised) and (b) shows the new value (some number in the quote equals it at
     scale 1, 1e3 or 1e6, after OCR clean-up of O-for-0 and stray spaces). Anything else is reverted.
     Structural edits (added/removed list items) are rejected.
  4. The record is re-validated; the retry, every proposed change and its verdict are logged.
A genuinely inconsistent document therefore stays an EXCEPTION: its true values are printed, so
"fixing" it would need a quote for a number the document doesn't contain.
"""
from __future__ import annotations

import copy
import datetime as dt
import re
from dataclasses import asdict

from pydantic import BaseModel, ConfigDict, Field, create_model

from agents.extractors.base import load_prompt
from agents.llm_client import ExtractionError, LLMClient
from validation.rules import RULES
from validation.validate import SCHEMAS, validate_record

MAX_RETRIES = 1
RETRY_PROMPT_VERSION = "retry_v1"
FINANCE_RULES = {r.__name__ for rules in RULES.values() for r in rules}
LIST_KEYS = {"holdings": "name", "top_holdings": "name", "top_sector_exposures": "sector"}
SCALES = (1, 1_000, 1_000_000)
FREE_TEXT = {"commentary_summary"}  # model-written prose, not a value read from the page: first pass kept


# --------------------------------------------------------------------------- field paths

def flatten(fields: dict) -> dict[str, object]:
    """{"holdings[Acme].cost": 1.0, "ending_nav": 2.0, ...}; list items keyed by name/sector."""
    flat = {}
    for k, v in fields.items():
        if k == "confidence":
            continue
        if k in LIST_KEYS and isinstance(v, list):
            for item in v:
                ident = item[LIST_KEYS[k]]
                for sk, sv in item.items():
                    if sk != LIST_KEYS[k]:
                        flat[f"{k}[{ident}].{sk}"] = sv
        else:
            flat[k] = v
    return flat


_PATH = re.compile(r"^(?P<list>\w+)\[(?P<ident>.*)\]\.(?P<field>\w+)$")


def set_path(fields: dict, path: str, value) -> None:
    m = _PATH.match(path)
    if not m:
        fields[path] = value
        return
    key = LIST_KEYS[m["list"]]
    item = next(i for i in fields[m["list"]] if i[key] == m["ident"])
    item[m["field"]] = value


def get_path(fields: dict, path: str):
    return flatten(fields).get(path)


# --------------------------------------------------------------------------- quote check

def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _ocr_clean(s: str) -> str:
    prev = None
    while prev != s:  # O between digits / commas / points is a zero
        prev = s
        s = re.sub(r"(?<=[\d,.])O|O(?=[\d,.])", "0", s)
    return re.sub(r"(?<=\d),\s+(?=\d)", ",", s)


def numbers_in(quote: str) -> list[tuple[float, int]]:
    """(value, decimals printed) for every number in the quote, after OCR clean-up."""
    return [(float(x.replace(",", "")), len(x.split(".")[1]) if "." in x else 0)
            for x in re.findall(r"\d[\d,]*(?:\.\d+)?", _ocr_clean(quote))]


def shows_value(quote: str, value: float) -> bool:
    """True if some number in the quote equals |value| at scale 1, 1e3 or 1e6, within half its printed precision."""
    target = abs(float(value))
    return any(abs(n * s - target) <= 0.5 * s * 10 ** -d + 1e-9 for n, d in numbers_in(quote) for s in SCALES)


def check_quote(quote: str, new_value, document_text: str) -> tuple[bool, str]:
    """Code-side evidence check for one changed value. Returns (accepted, reason)."""
    if not quote or not quote.strip():
        return False, "no quote"
    if _ws(quote) not in _ws(document_text):
        return False, "quote not found verbatim in document"
    if isinstance(new_value, bool) or new_value is None:
        return False, "unsupported value type"
    if isinstance(new_value, (int, float)):
        if shows_value(quote, new_value):
            return True, "quote shows value"
        return False, "quote does not show the new value"
    if any(form.casefold() in _ws(_ocr_clean(quote)).casefold() for form in printed_forms(str(new_value))):
        return True, "quote shows value"
    return False, "quote does not show the new value"


def printed_forms(value: str) -> list[str]:
    """Ways a string value can appear in a document: itself, or an ISO date in common printed formats."""
    forms = [value.strip()]
    try:
        d = dt.date.fromisoformat(value.strip())
    except ValueError:
        return forms
    return forms + [f"{d:%B} {d.day}, {d.year}", f"{d:%m/%d/%Y}", f"{d:%d-%b-%Y}", f"{d.day} {d:%B} {d.year}"]


# --------------------------------------------------------------------------- retry

class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(description="Field path exactly as listed under 'First-pass values'.")
    quote: str = Field(description="Verbatim text copied from the document that shows the corrected value.")


def review_schema(schema: type[BaseModel]) -> type[BaseModel]:
    return create_model(
        f"{schema.__name__}Review",
        __config__=ConfigDict(extra="forbid"),
        extraction=(schema, Field(description="The full extraction after review.")),
        changes=(list[Change], Field(description="One entry per value you changed; empty if none.")),
    )


def finance_issues(rec: dict) -> list[dict]:
    issues = (rec.get("validation") or {}).get("issues", [])
    return [i for i in issues if i["rule"] in FINANCE_RULES]


def build_review_prompt(extractor_prompt: str, extracted: dict, issues: list[dict]) -> str:
    values = "\n".join(f"- {k}: {v}" for k, v in flatten(extracted).items())
    findings = "\n".join(f"- {i['rule']}: {i['reason']} (expected {i['expected']}, found {i['actual']}, "
                         f"difference {i['difference']})" for i in issues)
    return (f"{extractor_prompt}\n\n{load_prompt(RETRY_PROMPT_VERSION)}\n\n"
            f"### First-pass values\n{values}\n\n### Validator findings\n{findings}\n")


def self_correct(rec: dict, document_text: str, client: LLMClient, extractor_prompt_version: str) -> dict:
    """Run at most one retry on a record that failed a finance rule. Mutates and returns rec."""
    issues = finance_issues(rec)
    doc_type = (rec.get("route") or {}).get("doc_type")
    if not issues or doc_type not in SCHEMAS or not rec.get("extracted"):
        return rec
    before = copy.deepcopy(rec["extracted"])
    log: dict = {"triggered_by": [i["rule"] for i in issues], "prompt_version": RETRY_PROMPT_VERSION,
                 "attempts": 1, "changes": [], "error": None, "status_before": rec["validation"]["status"]}
    rec["self_correction"] = log
    try:
        review = client.extract(build_review_prompt(load_prompt(extractor_prompt_version), before, issues),
                                document_text, review_schema(SCHEMAS[doc_type]))
    except ExtractionError as e:
        log["error"] = str(e)
        return rec
    finally:
        if client.last_call:
            rec.setdefault("calls", {})["retry"] = asdict(client.last_call)

    proposed = review.extraction.model_dump(mode="json")
    old_flat, new_flat = flatten(before), flatten(proposed)
    quotes = {c.field: c.quote for c in review.changes}
    final = copy.deepcopy(before)
    if set(old_flat) != set(new_flat):
        log["structural_change_rejected"] = sorted(set(old_flat) ^ set(new_flat))
    for path, old in old_flat.items():
        if path in FREE_TEXT:
            continue
        new = new_flat.get(path, old)
        if new == old or (isinstance(old, (int, float)) and isinstance(new, (int, float)) and abs(new - old) < 1e-9):
            continue
        ok, reason = check_quote(quotes.get(path, ""), new, document_text) if path in quotes else (False, "no quote")
        log["changes"].append({"field": path, "old": old, "new": new, "quote": quotes.get(path),
                               "accepted": ok, "reason": reason})
        if ok:
            set_path(final, path, new)
    final["confidence"] = proposed.get("confidence", before.get("confidence"))
    rec["extracted"] = final
    v = validate_record(rec)
    rec["validation"] = {"status": v.status, "issues": [i.to_dict() for i in v.issues]}
    log["status_after"] = v.status
    return rec
