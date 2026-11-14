"""Regression gate. Offline: reads run logs, never calls the API.

test_no_regression_vs_baseline is the gate, once per split: the newest run of that split by the
baseline's model must not, vs evals/baseline.json (main) or evals/baseline_hard.json (hard), drop
any field's accuracy > 5 points, drop router accuracy, or drop exception recall. Point it at a
specific log with EVAL_LOG=path/to/run.jsonl (its split comes from the log header). The other
tests prove the gate fails on degraded logs.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from evals import regression
from evals.scoring import compute_metrics, load_log, split_of

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("split", sorted(regression.BASELINES))
def test_no_regression_vs_baseline(split):
    env = os.environ.get("EVAL_LOG")
    if env:
        log = Path(env).resolve()
        if split_of(load_log(log)[0]) != split:
            pytest.skip(f"EVAL_LOG is not a {split}-split log")
    else:
        log = regression.gate_log(split)
    if log is None:
        pytest.skip(f"no {split} baseline or no {split}-split run by the baseline's model yet")
    _, problems = regression.check_log(log)
    base = regression.BASELINES[split].relative_to(ROOT)
    assert not problems, f"Regression in {log.name} vs {base}:\n  - " + "\n  - ".join(problems)


# --------------------------------------------------------------------------- the gate actually bites

@pytest.fixture(scope="module")
def baseline_records():
    """Records from the run the baseline was built from."""
    src = ROOT / regression.load_baseline()["source_log"]
    return load_log(src)


def _write(tmp_path, header, records, name="20990101T000000Z_pipeline.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in [header, *records]) + "\n")
    return p


def _problems(tmp_path, header, records):
    return regression.check_log(_write(tmp_path, header, records))[1]


def _doc(records, doc_file):
    return next(r for r in records if r["file"] == doc_file)


def test_baseline_log_itself_passes(tmp_path, baseline_records):
    header, records, _ = baseline_records
    assert _problems(tmp_path, header, records) == []


def test_field_drop_over_5_points_fails_and_names_field(tmp_path, baseline_records):
    header, records, _ = baseline_records
    recs = copy.deepcopy(records)
    cas_files = [r["file"] for r in recs if (r.get("route") or {}).get("doc_type") == "CAS"][:1]
    for f in cas_files:  # 1 of 10 CAS docs wrong -> CAS.ending_nav 100% -> 90% (10-point drop)
        _doc(recs, f)["extracted"]["ending_nav"] += 1_000_000
    problems = _problems(tmp_path, header, recs)
    assert any(p.startswith("field CAS.ending_nav accuracy dropped 10.0 points") and cas_files[0][:-4] in p
               for p in problems), problems


def test_small_drop_within_5_points_passes(tmp_path, baseline_records):
    header, records, _ = baseline_records
    recs = copy.deepcopy(records)
    soi = next(r for r in recs if (r.get("route") or {}).get("doc_type") == "SOI")
    soi["extracted"]["holdings"][0]["cost"] += 1_000_000  # 1 of 86 holdings -> ~1.2-point drop
    assert not any("SOI.holdings.cost" in p for p in _problems(tmp_path, header, recs))


def test_router_drop_fails(tmp_path, baseline_records):
    header, records, _ = baseline_records
    recs = copy.deepcopy(records)
    unk = next(r for r in recs if (r.get("route") or {}).get("doc_type") == "UNKNOWN")
    unk["route"]["doc_type"] = "CAS"  # misroute an UNKNOWN doc: no field drops, router drops
    problems = _problems(tmp_path, header, recs)
    assert any(p.startswith("router accuracy dropped") for p in problems), problems


def test_exception_recall_drop_fails(tmp_path, baseline_records):
    header, records, _ = baseline_records
    recs = copy.deepcopy(records)
    # doc_005 (broken LETTER: gross 234 vs long+short 219). Pretend extraction "fixed" the gross.
    letter = _doc(recs, "doc_005.pdf")
    letter["extracted"]["gross_exposure"] = letter["extracted"]["long_exposure"] + letter["extracted"]["short_exposure"]
    problems = _problems(tmp_path, header, recs)
    assert any(p.startswith("exception recall dropped: 1.00 -> 0.75") and "doc_005" in p for p in problems), problems


def test_missing_field_counts_as_regression(tmp_path, baseline_records):
    header, records, _ = baseline_records
    recs = [r for r in copy.deepcopy(records) if (r.get("route") or {}).get("doc_type") != "LETTER"]
    problems = _problems(tmp_path, header, recs)
    assert any(p.startswith("field LETTER.gross_exposure: missing from this run") for p in problems), problems


def test_metrics_recomputed_not_trusted_from_log(tmp_path, baseline_records):
    """An appended eval line claiming 100% can't hide a degraded record."""
    header, records, _ = baseline_records
    recs = copy.deepcopy(records)
    _doc(recs, "doc_002.pdf")["extracted"]["investor_name"] = "Someone Else"
    good = {"kind": "eval", "metrics": compute_metrics(records)}
    problems = regression.check_log(_write(tmp_path, header, recs + [good]))[1]
    assert any("CAS.investor_name" in p for p in problems), problems
