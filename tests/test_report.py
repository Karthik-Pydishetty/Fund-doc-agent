"""evals.report builds reports/results.html from the committed run logs, offline."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evals import report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def page():
    return report.build()


@pytest.fixture(scope="module")
def runs():
    return report.load_runs()


def section(html: str, n: int) -> str:
    return re.search(rf'<section id="s{n}">(.*?)</section>', html, re.S).group(1)


def test_all_seven_sections_present_and_each_names_its_runs(page):
    titles = re.findall(r"<h2>(\d)\. ([^<]+)</h2>", page)
    assert [t for _, t in titles] == ["Headline", "Regression gate", "Per-field accuracy", "Sonnet vs. Haiku",
                                      "Self-correction", "Exceptions queue", "Run history"]
    for n in range(1, 8):
        assert "runs/2026" in section(page, n), f"section {n} does not name a run log"


def test_self_contained(page):
    assert not re.search(r'(src|href)="https?://', page)
    assert "<script" not in page and "@import" not in page


def test_headline_numbers_come_from_the_gate_runs(page, runs):
    gate = report.gate_runs(runs)
    assert gate["main"]["path"].name.startswith("20261114T013741Z")
    s1 = section(page, 1)
    for split, r in gate.items():
        assert f"<code>{r['name']}</code>" in s1
    assert "$0.38" in s1 and "100.0%" in s1


def test_gate_passes_on_committed_logs(page):
    s2 = section(page, 2)
    assert s2.count("&#10003; PASS") == 2 and "FAIL" not in s2


def test_run_labels(runs):
    labels = {r["path"].name[:16]: report.label(r) for r in runs}
    assert labels["20261114T011048Z"].startswith("superseded by 20261114T013741Z")
    assert labels["20261114T191242Z"] == "fault-injection (10 faults), superseded by 20261114T191454Z"
    assert labels["20261114T194504Z"] == "fault-injection (10 faults)"   # Phase 9b "after": not superseded
    assert labels["20261114T013741Z"] == ""


def test_self_correction_uses_before_and_after_runs(page, runs):
    pairs = report.fault_pairs(runs)
    assert pairs["main"][0]["path"].name.startswith("20261114T191454Z")
    assert pairs["main"][1]["path"].name.startswith("20261114T194504Z")
    s5 = section(page, 5)
    assert "5/10" in s5 and "9/10" in s5 and "1/10" in s5      # recovered before/after, blind spot after
    assert s5.count("<b>closed</b>") == 6                       # 4 main + 2 hard faults closed by Phase 9b


def test_models_and_exceptions(page):
    s4 = section(page, 4)
    assert "claude-sonnet-5-5" in s4 and "claude-haiku-5-5" in s4 and "$0.0087" in s4
    s6 = section(page, 6)
    assert "cas_rollforward" in s6 and "doc_021" in s6 and "soi_pct_sum" in s6


def test_history_chart_has_a_point_per_run_per_panel(page, runs):
    s7 = section(page, 7)
    assert s7.count('<g class="pt-mark') == sum(3 if r["metrics"] else 2 for r in runs)  # webhook: no accuracy
    assert s7.count('<g class="pt-mark faded">') == 3 * sum(bool(r["superseded_by"]) for r in runs)
    assert "Table view of the chart" in s7


def test_cli_writes_only_the_report(tmp_path):
    out = tmp_path / "r.html"
    subprocess.run([sys.executable, "-m", "evals.report", "--out", str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    assert out.read_text().startswith("<!doctype html>")
