"""Turn one pipeline record into ACCEPTED or EXCEPTION, and write exceptions_report.csv.

A document is an EXCEPTION if any of these hold (all reasons are recorded, one CSV row each):
  extraction_error       the router or extractor failed (no usable output)
  unknown_doc_type       routed to UNKNOWN; no extractor applies, needs a human
  schema_invalid         extracted output does not validate against the pydantic schema
  <finance rule>         a rule in validation.rules fails beyond tolerance
  low_confidence         extractor confidence < MIN_CONFIDENCE
  low_router_confidence  router confidence < MIN_CONFIDENCE

Uses only the pipeline record; never ground truth.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from schemas.cas import CAS
from schemas.letter import Letter
from schemas.soi import SOI
from validation.rules import RULES, Issue

MIN_CONFIDENCE = 0.7
SCHEMAS = {"CAS": CAS, "SOI": SOI, "LETTER": Letter}
ACCEPTED, EXCEPTION = "ACCEPTED", "EXCEPTION"
CSV_COLUMNS = ["doc_id", "file", "doc_type", "rule", "reason", "expected", "actual", "difference",
               "tolerance", "confidence"]


@dataclass
class ValidationResult:
    doc_id: str
    file: str
    doc_type: str | None
    confidence: float | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def status(self) -> str:
        return EXCEPTION if self.issues else ACCEPTED


def validate_record(rec: dict) -> ValidationResult:
    route = rec.get("route") or {}
    doc_type = route.get("doc_type")
    extracted = rec.get("extracted")
    confidence = (extracted or {}).get("confidence", route.get("confidence"))
    res = ValidationResult(Path(rec["file"]).stem, rec["file"], doc_type, confidence)

    if rec.get("error"):
        res.issues.append(Issue("extraction_error", rec["error"]))
    if doc_type == "UNKNOWN":
        res.issues.append(Issue("unknown_doc_type", f"router could not assign a known type: {route.get('reason')}"))
    rc = route.get("confidence")
    if rc is not None and rc < MIN_CONFIDENCE:
        res.issues.append(Issue("low_router_confidence", "router confidence below threshold",
                                f">= {MIN_CONFIDENCE}", rc, round(rc - MIN_CONFIDENCE, 4)))
    if doc_type not in SCHEMAS or not extracted:
        return res

    try:
        fields = SCHEMAS[doc_type].model_validate(extracted).model_dump()
    except ValidationError as e:
        res.issues.append(Issue("schema_invalid", f"{e.error_count()} schema error(s): {e.errors()[0]['msg']}"))
        return res

    for rule in RULES[doc_type]:
        issue = rule(fields)
        if issue:
            res.issues.append(issue)
    ec = fields["confidence"]
    if ec < MIN_CONFIDENCE:
        res.issues.append(Issue("low_confidence", "extractor confidence below threshold",
                                f">= {MIN_CONFIDENCE}", ec, round(ec - MIN_CONFIDENCE, 4)))
    return res


def validate_records(records: list[dict]) -> list[ValidationResult]:
    return sorted((validate_record(r) for r in records), key=lambda r: r.doc_id)


def _report_rows(results: list[ValidationResult]):
    for r in results:
        for i in r.issues:
            yield {"doc_id": r.doc_id, "file": r.file, "doc_type": r.doc_type or "", "rule": i.rule,
                   "reason": i.reason, "expected": "" if i.expected is None else i.expected,
                   "actual": "" if i.actual is None else i.actual,
                   "difference": "" if i.difference is None else i.difference,
                   "tolerance": "" if i.tolerance is None else i.tolerance,
                   "confidence": "" if r.confidence is None else r.confidence}


def write_exceptions_report(results: list[ValidationResult], path: str | Path) -> int:
    """Rewrite the report with one row per failed check (what a pipeline run does)."""
    rows = list(_report_rows(results))
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def append_exceptions_report(results: list[ValidationResult], path: str | Path) -> int:
    """Append rows for these results (what the webhook service does); writes the header if the file is new."""
    path = Path(path)
    rows = list(_report_rows(results))
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)
