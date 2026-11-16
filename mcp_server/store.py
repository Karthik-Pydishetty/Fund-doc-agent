"""Read-only view of one pipeline run, for the MCP tools. Plain functions, so tests can call them directly.

Sources, and nothing else:
  - one pipeline run log in runs/ (default: the newest run of the main split that is not a
    fault-injection experiment; override with FUND_DOCS_LOG=<path> or FUND_DOCS_SPLIT=hard)
  - documents received by the webhook service (service/): every run log with "split": "webhook" in the
    same folder, merged in by document id (the file's SHA-256)
  - exceptions_report.csv, used only when it belongs to that run: the pipeline rewrites the file
    on every run and records no run id in it, so its (doc_id, rule) rows must match the log's
    validation results; otherwise the log's own validation results are used and the reply says so.
Never reads ground truth, never calls a model, never writes a file.
"""
from __future__ import annotations

import csv
import difflib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORT = ROOT / "exceptions_report.csv"
FUZZY_CUTOFF = 0.5


# --------------------------------------------------------------------------- loading

def _header(path: Path) -> dict:
    with path.open() as fh:
        return json.loads(fh.readline())


def choose_log(runs_dir: Path = RUNS, split: str | None = None) -> Path:
    env = os.environ.get("FUND_DOCS_LOG")
    if env:
        return Path(env).expanduser().resolve()
    split = split or os.environ.get("FUND_DOCS_SPLIT", "main")
    for log in sorted(runs_dir.glob("*_pipeline.jsonl"), reverse=True):
        h = _header(log)
        if h.get("split", "main") == split and not h.get("inject_faults"):
            return log
    raise FileNotFoundError(f"no {split}-split pipeline run in {runs_dir} (fault-injection runs are skipped)")


@dataclass
class Run:
    log: Path
    header: dict
    records: dict[str, dict]                      # doc_id -> record
    webhook_logs: list[Path] = field(default_factory=list)
    report_rows: list[dict] = field(default_factory=list)
    report_used: bool = False
    note: str | None = None

    def source(self) -> dict:
        h = self.header
        return {"run_log": _rel(self.log), "run_started": h.get("timestamp"), "split": h.get("split", "main"),
                "model": h.get("model"), "prompt_versions": h.get("prompt_versions"),
                "self_correct": bool(h.get("self_correct")),
                "webhook_logs": [_rel(p) for p in self.webhook_logs],
                "webhook_documents": sum(1 for r in self.records.values() if r.get("webhook")),
                "exceptions_source": "exceptions_report.csv" if self.report_used else "run log validation",
                **({"note": self.note} if self.note else {})}


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _validation(rec: dict) -> dict:
    if rec.get("validation"):
        return rec["validation"]
    from validation.validate import validate_record  # logs written before Phase 4 carry no validation
    v = validate_record(rec)
    return {"status": v.status, "issues": [i.to_dict() for i in v.issues]}


def load(log: Path | None = None, report: Path = REPORT) -> Run:
    log = Path(log) if log else choose_log()
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    records = {Path(r["file"]).stem: r for r in lines[1:] if "file" in r}
    for r in records.values():
        r["validation"] = _validation(r)
    run = Run(log, lines[0], records)
    for wl in sorted(log.parent.glob("*_pipeline.jsonl")):  # webhook documents, newest record wins
        wlines = [json.loads(x) for x in wl.read_text().splitlines() if x.strip()]
        if wl == log or not wlines or wlines[0].get("split") != "webhook":
            continue
        run.webhook_logs.append(wl)
        for r in wlines[1:]:
            if "file" in r:
                r["validation"] = _validation(r)
                records[Path(r["file"]).stem] = r

    log_pairs = {(d, i["rule"]) for d, r in records.items() for i in r["validation"]["issues"]}
    if report.exists():
        with report.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if {(r["doc_id"], r["rule"]) for r in rows} == log_pairs:
            run.report_rows, run.report_used = rows, True
        else:
            run.note = (f"{report.name} was written by a different run, so exceptions come from the log's own "
                        f"validation results. Regenerate it offline with: python -m validation.run --log {_rel(log)}")
    else:
        run.note = f"{report.name} not found; exceptions come from the log's own validation results"
    return run


# --------------------------------------------------------------------------- helpers

def _num(v):
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v if v != "" else None
    return v


def _issues(run: Run, doc_id: str) -> list[dict]:
    keys = ("rule", "reason", "expected", "actual", "difference", "tolerance")
    if run.report_used:
        return [{k: _num(r[k]) for k in keys} for r in run.report_rows if r["doc_id"] == doc_id]
    return [{k: i.get(k) for k in keys} for i in run.records[doc_id]["validation"]["issues"]]


def _retry(rec: dict) -> dict | None:
    sc = rec.get("self_correction")
    if not sc:
        return None
    return {"triggered_by": sc.get("triggered_by"), "attempts": sc.get("attempts"),
            "status_before": sc.get("status_before"), "status_after": sc.get("status_after"),
            "error": sc.get("error"),
            "changes": [{k: c.get(k) for k in ("field", "old", "new", "quote", "accepted", "reason")}
                        for c in sc.get("changes", [])]}


def _doc_type(rec: dict) -> str | None:
    return (rec.get("route") or {}).get("doc_type")


def _confidence(rec: dict):
    return (rec.get("extracted") or {}).get("confidence", (rec.get("route") or {}).get("confidence"))


def resolve_doc_id(run: Run, doc_id: str) -> str | None:
    """Exact id ('doc_021', 'doc_021.pdf'), or a unique prefix of 8+ characters (webhook SHA-256 ids)."""
    key = Path(str(doc_id).strip()).stem
    if key in run.records:
        return key
    hits = [k for k in run.records if len(key) >= 8 and k.startswith(key)]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- tools

def list_exceptions(run: Run) -> dict:
    items = []
    for doc_id, rec in sorted(run.records.items()):
        if rec["validation"]["status"] != "EXCEPTION":
            continue
        items.append({"doc_id": doc_id, "doc_type": _doc_type(rec), "confidence": _confidence(rec),
                      **({"received_via_webhook": rec["webhook"].get("filename")} if rec.get("webhook") else {}),
                      "failed_checks": _issues(run, doc_id), "retry_history": _retry(rec)})
    return {"source": run.source(), "count": len(items), "documents_in_run": len(run.records), "exceptions": items}


def get_document(run: Run, doc_id: str) -> dict:
    key = resolve_doc_id(run, doc_id)
    if key is None:
        return {"source": run.source(), "error": f"no document {doc_id!r} in this run",
                "available_doc_ids": sorted(run.records)}
    rec = run.records[key]
    out = {"source": run.source(), "doc_id": key, "file": rec["file"],
           **({"webhook": rec["webhook"]} if rec.get("webhook") else {}),
           "route": rec.get("route"), "extracted": rec.get("extracted"), "extraction_error": rec.get("error"),
           "validation": {"status": rec["validation"]["status"], "failed_checks": _issues(run, key)},
           "retry_history": _retry(rec)}
    if rec.get("fault_injection"):
        out["fault_injection"] = rec["fault_injection"]
    return out


def _quarter_key(q: str) -> tuple[int, int]:
    try:
        n, y = str(q).strip().split(" ")
        return int(y), int(n.lstrip("Qq"))
    except ValueError:
        return (0, 0)


def match_fund(names: list[str], query: str) -> tuple[str | None, str, list[str]]:
    """(matched name, how it matched, candidates). Exact (ignoring case) > contained > fuzzy."""
    q = query.strip().casefold()
    exact = [n for n in names if n.casefold() == q]
    if exact:
        return exact[0], "exact", exact
    contains = [n for n in names if q and q in n.casefold()]
    if len(contains) == 1:
        return contains[0], "contains", contains
    if len(contains) > 1:
        return None, "ambiguous", sorted(contains)
    folded = {n.casefold(): n for n in names}
    close = difflib.get_close_matches(q, list(folded), n=3, cutoff=FUZZY_CUTOFF)
    if close:
        return folded[close[0]], "fuzzy", [folded[c] for c in close]
    return None, "none", []


def get_fund_summary(run: Run, fund_name: str) -> dict:
    by_fund: dict[str, list[tuple[str, dict]]] = {}
    for doc_id, rec in run.records.items():
        name = (rec.get("extracted") or {}).get("fund_name")
        if name:
            by_fund.setdefault(name, []).append((doc_id, rec))
    matched, how, candidates = match_fund(sorted(by_fund), fund_name)
    base = {"source": run.source(), "query": fund_name, "match": how}
    if matched is None:
        return {**base, "error": f"no fund matching {fund_name!r} in this run",
                "candidates": candidates or sorted(by_fund)}

    docs = sorted(by_fund[matched])
    status = {d: r["validation"]["status"] for d, r in docs}
    checks = {d: [i["rule"] for i in _issues(run, d)] for d, _ in docs}

    def tag(doc_id: str) -> dict:
        return {"doc_id": doc_id, "status": status[doc_id], "failed_checks": checks[doc_id]}

    cas = [(d, r["extracted"]) for d, r in docs if _doc_type(r) == "CAS"]
    soi = [(d, r["extracted"]) for d, r in docs if _doc_type(r) == "SOI"]
    letters = [(d, r["extracted"]) for d, r in docs if _doc_type(r) == "LETTER"]

    latest_by_investor: dict[str, tuple[str, dict]] = {}
    for d, f in cas:
        cur = latest_by_investor.get(f["investor_name"])
        if cur is None or str(f["period_end"]) > str(cur[1]["period_end"]):
            latest_by_investor[f["investor_name"]] = (d, f)
    capital_accounts = [{**tag(d), "investor": f["investor_name"], "period_start": f["period_start"],
                         "period_end": f["period_end"], "currency": f["currency"],
                         **{k: f[k] for k in ("beginning_nav", "contributions", "distributions", "management_fees",
                                              "incentive_fees", "net_gain_loss", "ending_nav", "total_commitment",
                                              "unfunded_commitment")}}
                        for d, f in sorted(latest_by_investor.values(), key=lambda x: x[1]["investor_name"])]

    schedule = None
    if soi:
        d, f = max(soi, key=lambda x: str(x[1]["as_of_date"]))
        hs = sorted(f["holdings"], key=lambda h: -h["fair_value"])
        schedule = {**tag(d), "as_of_date": f["as_of_date"], "total_fair_value": f["total_fair_value"],
                    "total_cost": sum(h["cost"] for h in hs), "holdings_count": len(hs),
                    "largest_holdings": [{k: h[k] for k in ("name", "sector", "geography", "fair_value", "pct_of_nav")}
                                         for h in hs[:5]]}

    letter = None
    if letters:
        d, f = max(letters, key=lambda x: _quarter_key(x[1]["quarter"]))
        letter = {**tag(d), **{k: f[k] for k in ("quarter", "quarterly_return", "long_exposure", "short_exposure",
                                                  "gross_exposure", "net_exposure", "top_sector_exposures",
                                                  "top_holdings", "commentary_summary")}}

    return {**base, "fund_name": matched, "candidates": candidates if how == "fuzzy" else None,
            "manager": next((f["manager"] for _, f in cas), None),
            "documents": [{**tag(d), "doc_type": _doc_type(r)} for d, r in docs],
            "documents_needing_review": [d for d, _ in docs if status[d] == "EXCEPTION"],
            "latest_capital_accounts": capital_accounts, "latest_schedule_of_investments": schedule,
            "latest_investor_letter": letter,
            "note_on_figures": "Figures are as extracted from each document. Any document listed under "
                               "documents_needing_review failed a check; treat its figures with care."}
