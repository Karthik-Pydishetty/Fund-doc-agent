"""Phase 10: the read-only MCP server. Tools are called directly, through the server, and over stdio."""
import ast
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import pytest

from mcp_server import store
from mcp_server.server import server

ROOT = Path(__file__).resolve().parents[1]
MAIN_LOG = ROOT / "runs/20261114T013741Z_pipeline.jsonl"   # committed main-split run (Sonnet, v1)
FUND_A = "Granite Peak Capital Partners III, L.P."
FUND_B = "Granite Peak Growth Fund"


def cas(fund, investor, start, end, ending, **kw):
    return {"fund_name": fund, "manager": "GP LLC", "investor_name": investor, "period_start": start,
            "period_end": end, "currency": "USD", "beginning_nav": 100.0, "contributions": 0.0,
            "distributions": 0.0, "management_fees": 1.0, "incentive_fees": 0.0, "net_gain_loss": 1.0,
            "ending_nav": ending, "total_commitment": 500.0, "unfunded_commitment": 50.0, "confidence": 0.9, **kw}


def rec(file, doc_type, extracted, status="ACCEPTED", issues=(), **kw):
    return {"file": file, "route": {"doc_type": doc_type, "reason": "r", "confidence": 0.97},
            "extracted": extracted, "error": None, "calls": {},
            "validation": {"status": status, "issues": list(issues)}, **kw}


ROLL = {"rule": "cas_rollforward", "reason": "ending_nav != roll-forward", "expected": 100.0, "actual": 120.0,
        "difference": 20.0, "tolerance": 7.0}
RETRY = {"triggered_by": ["cas_rollforward"], "attempts": 1, "status_before": "EXCEPTION",
         "status_after": "EXCEPTION", "error": None, "prompt_version": "retry_v1",
         "changes": [{"field": "ending_nav", "old": 120.0, "new": 100.0, "quote": "Ending balance $120",
                      "accepted": False, "reason": "quote does not show the new value"}]}


def fixture_records():
    soi = {"fund_name": FUND_A, "as_of_date": "2025-09-30", "total_fair_value": 100.0, "confidence": 0.9,
           "holdings": [{"name": n, "sector": "S", "geography": "G", "cost": c, "fair_value": fv, "pct_of_nav": fv}
                        for n, c, fv in (("Small Co", 5.0, 10.0), ("Big Co", 30.0, 60.0), ("Mid Co", 20.0, 30.0))]}
    letter = lambda q, ret: {"fund_name": FUND_B, "quarter": q, "long_exposure": 120.0, "short_exposure": 40.0,  # noqa: E731
                             "gross_exposure": 160.0, "net_exposure": 80.0, "top_sector_exposures": [],
                             "top_holdings": [], "quarterly_return": ret, "commentary_summary": "c", "confidence": 0.9}
    return [
        rec("doc_001.pdf", "CAS", cas(FUND_A, "Pension Plan", "2025-04-01", "2025-06-30", 90.0)),
        rec("doc_002.pdf", "CAS", cas(FUND_A, "Pension Plan", "2025-07-01", "2025-09-30", 100.0)),
        rec("doc_003.pdf", "CAS", cas(FUND_A, "Endowment", "2025-07-01", "2025-09-30", 120.0),
            "EXCEPTION", [ROLL], self_correction=RETRY),
        rec("doc_004.pdf", "SOI", soi),
        rec("doc_005.pdf", "LETTER", letter("Q4 2024", -1.5)),
        rec("doc_006.pdf", "LETTER", letter("Q2 2025", 3.2)),
        {"file": "doc_007.pdf", "route": {"doc_type": "UNKNOWN", "reason": "a notice", "confidence": 0.96},
         "extracted": None, "error": None, "calls": {},
         "validation": {"status": "EXCEPTION", "issues": [{"rule": "unknown_doc_type", "reason": "router could not "
                        "assign a known type", "expected": None, "actual": None, "difference": None, "tolerance": None}]}},
    ]


def write_log(path: Path, records, **header) -> Path:
    h = {"timestamp": "2026-01-01T00:00:00+00:00", "kind": "pipeline", "split": "main", "model": "m",
         "prompt_versions": {"cas": "cas_v1"}, **header}
    path.write_text("\n".join(json.dumps(x) for x in [h, *records]) + "\n")
    return path


def write_report(path: Path, rows) -> Path:
    cols = ["doc_id", "file", "doc_type", "rule", "reason", "expected", "actual", "difference", "tolerance", "confidence"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


@pytest.fixture
def run(tmp_path):
    log = write_log(tmp_path / "20260101T000000Z_pipeline.jsonl", fixture_records())
    return store.load(log, report=tmp_path / "missing.csv")


# --------------------------------------------------------------------------- choosing the run

def test_choose_log_skips_fault_runs_and_other_splits(tmp_path, monkeypatch):
    monkeypatch.delenv("FUND_DOCS_LOG", raising=False)
    monkeypatch.delenv("FUND_DOCS_SPLIT", raising=False)
    main = write_log(tmp_path / "20260101T000000Z_pipeline.jsonl", [])
    write_log(tmp_path / "20260102T000000Z_pipeline.jsonl", [], split="hard")
    write_log(tmp_path / "20260103T000000Z_pipeline.jsonl", [], inject_faults=10)
    assert store.choose_log(tmp_path) == main
    assert store.choose_log(tmp_path, split="hard").name.startswith("20260102")
    monkeypatch.setenv("FUND_DOCS_LOG", str(main))
    assert store.choose_log(tmp_path, split="hard") == main


def test_default_run_is_newest_main_non_fault_run(monkeypatch):
    monkeypatch.delenv("FUND_DOCS_LOG", raising=False)
    monkeypatch.delenv("FUND_DOCS_SPLIT", raising=False)
    h = json.loads(store.choose_log().open().readline())
    assert h.get("split", "main") == "main" and not h.get("inject_faults")


# --------------------------------------------------------------------------- list_exceptions

def test_list_exceptions(run):
    out = store.list_exceptions(run)
    assert out["count"] == 2 and out["documents_in_run"] == 7
    by = {e["doc_id"]: e for e in out["exceptions"]}
    e = by["doc_003"]
    assert e["doc_type"] == "CAS" and e["confidence"] == 0.9
    assert e["failed_checks"] == [{k: ROLL[k] for k in ("rule", "reason", "expected", "actual", "difference", "tolerance")}]
    assert e["retry_history"]["changes"][0]["accepted"] is False
    assert e["retry_history"]["changes"][0]["quote"] == "Ending balance $120"
    assert by["doc_007"]["doc_type"] == "UNKNOWN" and by["doc_007"]["retry_history"] is None
    assert out["source"]["exceptions_source"] == "run log validation" and "not found" in out["source"]["note"]


def test_report_csv_used_only_when_it_matches_the_run(tmp_path):
    log = write_log(tmp_path / "20260101T000000Z_pipeline.jsonl", fixture_records())
    rows = [{"doc_id": "doc_003", "doc_type": "CAS", **{k: ROLL[k] for k in ROLL}, "confidence": 0.9},
            {"doc_id": "doc_007", "doc_type": "UNKNOWN", "rule": "unknown_doc_type", "reason": "from the csv"}]
    good = store.load(log, report=write_report(tmp_path / "good.csv", rows))
    assert good.report_used and good.source()["exceptions_source"] == "exceptions_report.csv"
    checks = {e["doc_id"]: e["failed_checks"] for e in store.list_exceptions(good)["exceptions"]}
    assert checks["doc_007"][0]["reason"] == "from the csv"
    assert checks["doc_003"][0]["expected"] == 100 and checks["doc_003"][0]["tolerance"] == 7  # numbers, not strings

    stale = store.load(log, report=write_report(tmp_path / "stale.csv", rows[:1]))
    assert not stale.report_used and "different run" in stale.source()["note"]


# --------------------------------------------------------------------------- get_document

@pytest.mark.parametrize("doc_id", ["doc_003", "doc_003.pdf", " doc_003 "])
def test_get_document(run, doc_id):
    out = store.get_document(run, doc_id)
    assert out["doc_id"] == "doc_003" and out["extracted"]["ending_nav"] == 120.0
    assert out["validation"]["status"] == "EXCEPTION"
    assert out["validation"]["failed_checks"][0]["rule"] == "cas_rollforward"
    assert out["retry_history"]["triggered_by"] == ["cas_rollforward"]
    assert out["route"]["doc_type"] == "CAS"


def test_get_document_unknown_id(run):
    out = store.get_document(run, "doc_999")
    assert "no document" in out["error"] and "doc_001" in out["available_doc_ids"]


# --------------------------------------------------------------------------- get_fund_summary

@pytest.mark.parametrize("query,how", [
    (FUND_A, "exact"), (FUND_A.upper(), "exact"),
    ("Capital Partners", "contains"),
    ("Granit Peak Capitol Partners III LP", "fuzzy"),
])
def test_fund_name_matching(run, query, how):
    out = store.get_fund_summary(run, query)
    assert out["fund_name"] == FUND_A and out["match"] == how


def test_ambiguous_and_missing_fund_names(run):
    amb = store.get_fund_summary(run, "granite peak")
    assert amb["match"] == "ambiguous" and set(amb["candidates"]) == {FUND_A, FUND_B} and "error" in amb
    none = store.get_fund_summary(run, "Zzyzx Holdings")
    assert none["match"] == "none" and FUND_A in none["candidates"]


def test_fund_summary_takes_latest_figures(run):
    out = store.get_fund_summary(run, FUND_A)
    accounts = {a["investor"]: a for a in out["latest_capital_accounts"]}
    assert accounts["Pension Plan"]["doc_id"] == "doc_002" and accounts["Pension Plan"]["ending_nav"] == 100.0
    assert accounts["Endowment"]["status"] == "EXCEPTION" and accounts["Endowment"]["failed_checks"] == ["cas_rollforward"]
    assert out["documents_needing_review"] == ["doc_003"]
    soi = out["latest_schedule_of_investments"]
    assert soi["holdings_count"] == 3 and soi["total_cost"] == 55.0
    assert [h["name"] for h in soi["largest_holdings"]] == ["Big Co", "Mid Co", "Small Co"]
    assert out["latest_investor_letter"] is None and out["manager"] == "GP LLC"

    letter = store.get_fund_summary(run, FUND_B)["latest_investor_letter"]
    assert letter["quarter"] == "Q2 2025" and letter["quarterly_return"] == 3.2   # Q2 2025 is after Q4 2024


# --------------------------------------------------------------------------- the real logged run, through the server

def call(name, args=None):
    result = asyncio.run(server.call_tool(name, args or {}))
    assert not result.is_error
    return result.structured_content


@pytest.fixture
def real_log(monkeypatch):
    monkeypatch.setenv("FUND_DOCS_LOG", str(MAIN_LOG))


def test_real_run_exception_queue(real_log):
    out = call("list_exceptions")
    assert out["source"]["run_log"] == "runs/20261114T013741Z_pipeline.jsonl"
    by = {e["doc_id"]: e for e in out["exceptions"]}
    assert set(by) == {"doc_005", "doc_014", "doc_017", "doc_021", "doc_028", "doc_032"}
    roll = by["doc_021"]["failed_checks"][0]
    assert (roll["rule"], roll["expected"], roll["actual"]) == ("cas_rollforward", 13_770_000, 14_140_000)


def test_real_run_document_and_fund(real_log):
    doc = call("get_document", {"doc_id": "doc_021"})
    assert doc["extracted"]["ending_nav"] == 14_140_000 and doc["validation"]["status"] == "EXCEPTION"
    fund = call("get_fund_summary", {"fund_name": "kestrel growth"})
    assert fund["fund_name"] == "Kestrel Growth Equity Fund IV, SCSp" and fund["match"] == "contains"
    assert "doc_021" in fund["documents_needing_review"]
    assert fund["latest_schedule_of_investments"]["as_of_date"]


def test_tools_are_declared_read_only():
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"list_exceptions", "get_document", "get_fund_summary"}
    for t in tools:
        a = t.annotations
        assert a.read_only_hint is True and a.destructive_hint is False and a.open_world_hint is False


def test_stdio_round_trip():
    """Launch the server exactly as Claude Desktop would, and call a tool over stdio."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], cwd=str(ROOT),
                                   env={**os.environ, "PYTHONPATH": str(ROOT), "FUND_DOCS_LOG": str(MAIN_LOG)})

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                result = await session.call_tool("get_document", {"doc_id": "doc_032"})
                return names, result

    names, result = asyncio.run(go())
    assert names == {"list_exceptions", "get_document", "get_fund_summary"}
    assert not result.is_error
    assert result.structured_content["validation"]["failed_checks"][0]["rule"] == "cas_unfunded_le_commitment"


# --------------------------------------------------------------------------- read-only, no model, no ground truth

def _snapshot():
    skip = {".venv", ".git", "__pycache__", ".pytest_cache"}
    return {p: p.stat().st_mtime_ns for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts)}


def test_tools_write_nothing(real_log):
    before = _snapshot()
    call("list_exceptions")
    call("get_document", {"doc_id": "doc_021"})
    call("get_fund_summary", {"fund_name": "Bluewatr Infrastructure"})
    assert _snapshot() == before


def test_server_never_touches_ground_truth_or_a_model():
    for path in (ROOT / "mcp_server").glob("*.py"):
        text = path.read_text()
        assert "ground_truth" not in text, path
        mods = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        assert not {m for m in mods if m.split(".")[0] in ("anthropic", "evals") or m == "agents.llm_client"}, path
