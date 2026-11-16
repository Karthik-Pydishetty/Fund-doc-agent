"""Webhook service: FastAPI TestClient + a fake LLM client (no API calls)."""
import csv
import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.llm_client import LLMClient
from agents.pdf_text import pdf_to_text
from mcp_server import store as mcp_store
from schemas.router import RouteDecision
from service.app import MAX_BYTES, create_app

ROOT = Path(__file__).resolve().parents[1]
TRUTH = {d["file"]: d for d in json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]}
SECRET = "test-secret"


class FakeClient(LLMClient):
    """Answers from the answer key by recognising the document text; reviews change nothing."""

    provider = "fake"

    def __init__(self):
        super().__init__("fake-model")
        self.by_text = {pdf_to_text(ROOT / "data" / f): d for f, d in TRUTH.items() if f in ("doc_002.pdf", "doc_021.pdf")}
        self.calls = []

    def extract(self, system_prompt, document_text, pydantic_model):
        d = self.by_text[document_text]
        self.calls.append((d["file"], pydantic_model.__name__))
        if pydantic_model is RouteDecision:
            return RouteDecision(doc_type=d["doc_type"], reason="fake", confidence=0.95)
        if pydantic_model.__name__.endswith("Review"):
            return pydantic_model.model_validate({"extraction": {**d["fields"], "confidence": 0.9}, "changes": []})
        return pydantic_model.model_validate({**d["fields"], "confidence": 0.95})


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def pdf(name: str) -> bytes:
    return (ROOT / "data" / name).read_bytes()


@pytest.fixture
def env(tmp_path):
    fake = FakeClient()
    paths = dict(runs_dir=tmp_path / "runs", report=tmp_path / "exceptions_report.csv", uploads=tmp_path / "up")
    app = create_app(client=fake, secret=SECRET, **paths)
    return TestClient(app), fake, paths


def post(client, body: bytes, signature: str | None = "auto", name="x.pdf"):
    headers = {"Content-Type": "application/pdf", "X-Filename": name}
    if signature == "auto":
        signature = sign(body)
    if signature is not None:
        headers["X-Signature"] = signature
    return client.post("/documents", content=body, headers=headers)


def log_records(runs_dir: Path) -> list[dict]:
    out = []
    for log in sorted(runs_dir.glob("*_pipeline.jsonl")):
        lines = [json.loads(x) for x in log.read_text().splitlines()]
        assert lines[0]["split"] == "webhook" and lines[0]["self_correct"] is True
        out += lines[1:]
    return out


# --------------------------------------------------------------------------- tests

def test_health(env):
    client, _, _ = env
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_valid_signature_clean_doc_is_accepted(env):
    client, fake, paths = env
    body = pdf("doc_002.pdf")
    r = post(client, body, name="doc_002.pdf")
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert doc_id == hashlib.sha256(body).hexdigest() and r.json()["status"] == "processing"

    got = client.get(f"/documents/{doc_id}").json()        # TestClient runs the background task first
    assert got["status"] == "ACCEPTED" and got["doc_type"] == "CAS" and got["filename"] == "doc_002.pdf"
    assert got["extracted"]["ending_nav"] == TRUTH["doc_002.pdf"]["fields"]["ending_nav"]
    assert got["exception_reasons"] == [] and got["retry_history"] is None
    recs = log_records(paths["runs_dir"])
    assert len(recs) == 1 and recs[0]["webhook"]["document_id"] == doc_id and recs[0]["file"] == f"{doc_id}.pdf"
    assert not paths["report"].exists()                    # nothing to report for a clean doc


def test_broken_doc_is_an_exception_with_retry_history(env):
    client, fake, paths = env
    doc_id = post(client, pdf("doc_021.pdf"), name="doc_021.pdf").json()["document_id"]
    got = client.get(f"/documents/{doc_id}").json()
    assert got["status"] == "EXCEPTION"
    reason = got["exception_reasons"][0]
    assert (reason["rule"], reason["expected"], reason["actual"]) == ("cas_rollforward", 13_770_000, 14_140_000)
    assert got["retry_history"]["triggered_by"] == ["cas_rollforward"]
    assert got["retry_history"]["status_after"] == "EXCEPTION"
    assert ("doc_021.pdf", "CASReview") in fake.calls       # self-correct on: one retry was made
    rows = list(csv.DictReader(paths["report"].open()))
    assert [(r["doc_id"], r["rule"]) for r in rows] == [(doc_id, "cas_rollforward")]


@pytest.mark.parametrize("signature", [None, "", "sha256=" + "0" * 64, sign(b"other body"), "not-a-signature"])
def test_unsigned_or_badly_signed_is_401(env, signature):
    client, fake, paths = env
    r = post(client, pdf("doc_002.pdf"), signature=signature)
    assert r.status_code == 401 and fake.calls == [] and not paths["runs_dir"].exists()


def test_signed_with_the_wrong_secret_is_401(env):
    client, fake, _ = env
    body = pdf("doc_002.pdf")
    assert post(client, body, signature=sign(body, "wrong-secret")).status_code == 401 and fake.calls == []


def test_bare_hex_signature_is_accepted(env):
    client, _, _ = env
    body = pdf("doc_002.pdf")
    assert post(client, body, signature=sign(body).removeprefix("sha256=")).status_code == 202


def test_non_pdf_is_400(env):
    client, fake, _ = env
    r = post(client, b"just some text, signed correctly")
    assert r.status_code == 400 and fake.calls == []


def test_file_over_10_mb_is_413(env):
    client, fake, _ = env
    r = post(client, b"%PDF-" + b"0" * MAX_BYTES)
    assert r.status_code == 413 and fake.calls == []


def test_duplicate_upload_returns_same_id_without_reprocessing(env):
    client, fake, paths = env
    body = pdf("doc_021.pdf")
    first = post(client, body).json()
    n_calls = len(fake.calls)
    again = post(client, body)
    assert again.status_code == 200 and again.json()["duplicate"] is True
    assert again.json()["document_id"] == first["document_id"] and again.json()["status"] == "EXCEPTION"
    assert len(fake.calls) == n_calls                       # no model call the second time
    assert len(log_records(paths["runs_dir"])) == 1
    assert len(list(csv.DictReader(paths["report"].open()))) == 1


def test_duplicate_is_recognised_after_a_restart(env):
    client, fake, paths = env
    body = pdf("doc_002.pdf")
    doc_id = post(client, body).json()["document_id"]
    fresh_fake = FakeClient()
    restarted = TestClient(create_app(client=fresh_fake, secret=SECRET, **paths))
    assert restarted.get(f"/documents/{doc_id}").json()["status"] == "ACCEPTED"
    r = post(restarted, body)
    assert r.status_code == 200 and r.json()["duplicate"] is True and fresh_fake.calls == []


def test_unreadable_pdf_is_failed_not_lost(env):
    client, _, paths = env
    doc_id = post(client, b"%PDF-1.4 this is not really a pdf").json()["document_id"]
    got = client.get(f"/documents/{doc_id}").json()
    assert got["status"] == "FAILED" and got["error"].startswith("processing failed")
    assert log_records(paths["runs_dir"])[0]["webhook"]["status"] == "FAILED"


def test_unknown_document_is_404(env):
    client, _, _ = env
    assert client.get("/documents/" + "a" * 64).status_code == 404
    assert client.get("/documents/not-an-id").status_code == 404


def test_mcp_server_sees_webhook_documents(env, tmp_path):
    client, _, paths = env
    doc_id = post(client, pdf("doc_021.pdf"), name="doc_021.pdf").json()["document_id"]
    main_log = paths["runs_dir"] / "20200101T000000Z_pipeline.jsonl"   # a main-split run beside the webhook log
    main_log.write_text(json.dumps({"timestamp": "2020-01-01T00:00:00+00:00", "split": "main", "model": "m"}) + "\n")
    run = mcp_store.load(main_log, report=paths["report"])
    assert run.report_used                                  # the appended CSV rows match the webhook records
    exc = mcp_store.list_exceptions(run)["exceptions"]
    assert [e["doc_id"] for e in exc] == [doc_id] and exc[0]["received_via_webhook"] == "doc_021.pdf"
    assert mcp_store.get_document(run, doc_id[:12])["doc_id"] == doc_id   # unique prefix works


def test_report_includes_webhook_documents(env):
    from evals import report
    client, _, paths = env
    post(client, pdf("doc_021.pdf"), name="doc_021.pdf")
    runs = report.load_runs(paths["runs_dir"])
    assert runs[0]["split"] == "webhook" and runs[0]["metrics"] is None and not runs[0]["superseded_by"]
    html = report.sec_exceptions(runs)
    assert "webhook uploads · 1 of 1 documents flagged" in html and "cas_rollforward" in html
