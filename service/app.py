"""Webhook service: signed PDF uploads in, routed / extracted / validated results out.

    python -m service                 # http://127.0.0.1:8000, secret WEBHOOK_SECRET from .env

POST /documents        raw PDF bytes as the body (Content-Type: application/pdf), optional X-Filename.
                       X-Signature: sha256=<hex HMAC-SHA256 of the body with WEBHOOK_SECRET>.
                       401 unsigned or badly signed, 413 over 10 MB, 400 not a PDF.
                       202 + {"document_id"} for a new file; processing continues in the background
                       (route -> extract -> validate -> one self-correction retry if a rule fails).
                       The id is the file's SHA-256: re-sending a file returns the existing result
                       (200 when finished, 202 while still processing) and is never reprocessed.
GET  /documents/{id}   status (processing / ACCEPTED / EXCEPTION / FAILED), extracted fields,
                       exception reasons, retry history.
GET  /health
Results go to the pipeline's run logs (split "webhook") and exceptions_report.csv.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

import config  # noqa: F401  (imported for its side effect: loads .env)
from agents.llm_client import LLMClient, get_client
from agents.pipeline import resolve_prompts
from service.processor import process_document
from service.store import FAILED, PROCESSING, WebhookStore

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORT = ROOT / "exceptions_report.csv"
UPLOADS = ROOT / "service_data" / "uploads"
MAX_BYTES = 10 * 1024 * 1024
ID_RE = re.compile(r"^[0-9a-f]{64}$")


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def signature_ok(secret: str, body: bytes, header: str | None) -> bool:
    if not header:
        return False
    given = header.strip()
    given = given if given.startswith("sha256=") else "sha256=" + given
    return hmac.compare_digest(given, sign(secret, body))


def view(doc_id: str, entry: dict) -> dict:
    rec = entry.get("record") or {}
    sc = rec.get("self_correction")
    return {
        "document_id": doc_id,
        "status": entry["status"],
        "filename": entry.get("filename"),
        "received_at": entry.get("received_at"),
        "completed_at": (rec.get("webhook") or {}).get("completed_at"),
        "doc_type": (rec.get("route") or {}).get("doc_type"),
        "router_reason": (rec.get("route") or {}).get("reason"),
        "extracted": rec.get("extracted"),
        "error": rec.get("error"),
        "exception_reasons": [{k: i.get(k) for k in ("rule", "reason", "expected", "actual", "difference", "tolerance")}
                              for i in (rec.get("validation") or {}).get("issues", [])],
        "retry_history": None if not sc else {
            "triggered_by": sc.get("triggered_by"), "status_before": sc.get("status_before"),
            "status_after": sc.get("status_after"), "error": sc.get("error"),
            "changes": [{k: c.get(k) for k in ("field", "old", "new", "quote", "accepted", "reason")}
                        for c in sc.get("changes", [])]},
        "run_log": f"runs/{entry['run_log']}" if entry.get("run_log") else None,
    }


def create_app(client: LLMClient | None = None, secret: str | None = None, runs_dir: Path = RUNS,
               report: Path = REPORT, uploads: Path = UPLOADS, prompt_overrides: list[str] | None = None) -> FastAPI:
    secret = secret or os.environ.get("WEBHOOK_SECRET")
    if not secret:
        raise RuntimeError("WEBHOOK_SECRET is not set (add it to .env)")
    client = client or get_client()
    prompts = resolve_prompts(prompt_overrides)
    store = WebhookStore(runs_dir, report, header={
        "kind": "pipeline", "split": "webhook", "source": "webhook service", "provider": client.provider,
        "model": client.model, "prompt_versions": prompts, "self_correct": True, "inject_faults": 0})
    uploads = Path(uploads)

    app = FastAPI(title="Fund document webhook", version="1.0")
    app.state.store = store

    def run(doc_id: str, path: Path) -> None:
        try:
            rec = process_document(path, client, prompts)
            status = rec["validation"]["status"]
        except Exception as e:  # unreadable PDF or an unexpected crash: record it, never lose the document
            rec = {"file": path.name, "route": None, "extracted": None, "calls": {},
                   "error": f"processing failed: {type(e).__name__}: {e}"}
            status = FAILED
        store.finish(doc_id, rec, status)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": client.model, "documents": store.counts(),
                "run_log": f"runs/{store.log_path.name}" if store.log_path else None}

    @app.post("/documents", status_code=202)
    async def upload(request: Request, background: BackgroundTasks,
                     x_signature: str | None = Header(default=None),
                     x_filename: str | None = Header(default=None)):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BYTES:
            raise HTTPException(413, f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BYTES:
                raise HTTPException(413, f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
        body = bytes(body)
        if not signature_ok(secret, body, x_signature):
            raise HTTPException(401, "missing or invalid X-Signature")
        if not body.startswith(b"%PDF-"):
            raise HTTPException(400, "body is not a PDF")

        doc_id = hashlib.sha256(body).hexdigest()
        existing = store.claim(doc_id, x_filename)
        if existing is not None:
            code = 202 if existing["status"] == PROCESSING else 200
            return JSONResponse({**view(doc_id, existing), "duplicate": True}, status_code=code)

        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{doc_id}.pdf"
        path.write_bytes(body)
        background.add_task(run, doc_id, path)
        return {"document_id": doc_id, "status": PROCESSING, "duplicate": False,
                "status_url": f"/documents/{doc_id}"}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        entry = store.get(doc_id) if ID_RE.match(doc_id) else None
        if entry is None:
            raise HTTPException(404, f"no document {doc_id}")
        return view(doc_id, entry)

    return app

