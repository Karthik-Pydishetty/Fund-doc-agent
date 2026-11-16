"""Webhook document state, persisted in the pipeline's own run logs and exceptions report.

Each server session writes one run log, runs/<timestamp>_pipeline.jsonl, with header
"split": "webhook", and appends one record per finished document (same record format as main.py,
plus a "webhook" block). Exception rows are appended to exceptions_report.csv. On start-up every
earlier webhook log is read back, so a document already processed is never processed again, even
after a restart. The document id is the SHA-256 of the file's bytes.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

from validation.validate import append_exceptions_report, validate_record

PROCESSING, FAILED = "processing", "FAILED"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class WebhookStore:
    def __init__(self, runs_dir: Path, report: Path, header: dict):
        self.runs_dir, self.report, self.header = Path(runs_dir), Path(report), header
        self.lock = threading.Lock()
        self.log_path: Path | None = None
        self.docs: dict[str, dict] = {}      # doc_id -> {"status", "record", "filename", "received_at", "run_log"}
        self._load_existing()

    # ------------------------------------------------------------------ persistence

    def _load_existing(self) -> None:
        for log in sorted(self.runs_dir.glob("*_pipeline.jsonl")):
            lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            if not lines or lines[0].get("split") != "webhook":
                continue
            for rec in lines[1:]:
                wh = rec.get("webhook")
                if wh:
                    self.docs[wh["document_id"]] = {"status": wh["status"], "record": rec,
                                                    "filename": wh.get("filename"),
                                                    "received_at": wh.get("received_at"), "run_log": log.name}

    def _append_record(self, rec: dict) -> None:
        if self.log_path is None:  # one log per server session, created on the first finished document
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            started = dt.datetime.now(dt.timezone.utc)
            self.log_path = self.runs_dir / f"{started:%Y%m%dT%H%M%SZ}_pipeline.jsonl"
            while self.log_path.exists():  # two sessions in the same second
                started += dt.timedelta(seconds=1)
                self.log_path = self.runs_dir / f"{started:%Y%m%dT%H%M%SZ}_pipeline.jsonl"
            self.log_path.write_text(json.dumps({**self.header, "timestamp": started.isoformat()}) + "\n")
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ state

    def claim(self, doc_id: str, filename: str | None) -> dict | None:
        """Register a new document as processing. Returns the existing entry if the id is already known."""
        with self.lock:
            if doc_id in self.docs:
                return self.docs[doc_id]
            self.docs[doc_id] = {"status": PROCESSING, "record": None, "filename": filename,
                                 "received_at": now(), "run_log": None}
            return None

    def get(self, doc_id: str) -> dict | None:
        return self.docs.get(doc_id)

    def finish(self, doc_id: str, rec: dict, status: str) -> None:
        entry = self.docs[doc_id]
        rec["webhook"] = {"document_id": doc_id, "filename": entry["filename"], "received_at": entry["received_at"],
                          "completed_at": now(), "status": status}
        result = validate_record(rec)
        rec["validation"] = rec.get("validation") or {"status": result.status,
                                                      "issues": [i.to_dict() for i in result.issues]}
        with self.lock:
            self._append_record(rec)
            if result.issues:
                append_exceptions_report([result], self.report)
            entry.update(status=status, record=rec, run_log=self.log_path.name)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.docs.values():
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out
