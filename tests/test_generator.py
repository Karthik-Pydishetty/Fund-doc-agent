"""Phase 1: the generator is deterministic, internally consistent, and its PDFs extract cleanly.

These tests generate into a temp dir; they never touch data/ground_truth.json.
"""
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pdfplumber
import pytest

from generator.generate import generate

TOL = 1e-6
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    return out, generate(out, seed=42)


def _cas_ok(f):
    roll = (f["beginning_nav"] + f["contributions"] - f["distributions"] - f["management_fees"]
            - f["incentive_fees"] + f["net_gain_loss"])
    return abs(roll - f["ending_nav"]) < TOL and f["unfunded_commitment"] <= f["total_commitment"]


def _soi_ok(f):
    hs = f["holdings"]
    return (abs(sum(h["fair_value"] for h in hs) - f["total_fair_value"]) < TOL
            and abs(sum(h["pct_of_nav"] for h in hs) - 100) <= 0.5)


def _letter_ok(f):
    return (abs(f["long_exposure"] + f["short_exposure"] - f["gross_exposure"]) < 0.05
            and abs(f["long_exposure"] - f["short_exposure"] - f["net_exposure"]) < 0.05)


RULES = {"CAS": _cas_ok, "SOI": _soi_ok, "LETTER": _letter_ok}


def test_counts_and_layouts(run):
    out, truth = run
    docs = truth["documents"]
    assert Counter(d["doc_type"] for d in docs) == {"CAS": 10, "SOI": 10, "LETTER": 10, "UNKNOWN": 2}
    for t in ("CAS", "SOI", "LETTER"):
        assert len({d["layout"] for d in docs if d["doc_type"] == t}) == 3
    assert all((out / d["file"]).exists() for d in docs)
    assert sum(d["expected_exception"] for d in docs) == 4


def test_rules_hold_exactly_when_not_broken(run):
    for d in run[1]["documents"]:
        if d["doc_type"] == "UNKNOWN":
            continue
        assert RULES[d["doc_type"]](d["fields"]) is (not d["expected_exception"]), d["doc_id"]


def test_deterministic(tmp_path, run):
    out1, truth1 = run
    truth2 = generate(tmp_path, seed=42)
    assert truth1 == truth2
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    for d in truth1["documents"]:
        assert digest(out1 / d["file"]) == digest(tmp_path / d["file"]), d["file"]


def test_deterministic_across_processes(tmp_path):
    """Separate interpreters with different hash seeds must produce identical bytes."""
    digests = []
    for hashseed in ("1", "2"):
        out = tmp_path / hashseed
        subprocess.run([sys.executable, "-m", "generator.generate", "--out", str(out)], check=True,
                       cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": hashseed}, capture_output=True)
        digests.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    assert digests[0] == digests[1]


def test_pdf_text_extracts_cleanly(run):
    out, truth = run
    for d in truth["documents"]:
        with pdfplumber.open(out / d["file"]) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "(cid:" not in text, d["file"]
        if d["doc_type"] != "UNKNOWN":
            assert d["fields"]["fund_name"] in text, d["file"]
