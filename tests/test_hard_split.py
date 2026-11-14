"""Phase 7: the hard split is reproducible, consistent with the validator, and kept apart from data/."""
import hashlib
import json
from pathlib import Path

import pytest

from evals import regression
from evals.compare_models import newest_per_model
from evals.scoring import load_truth
from generator import hard
from validation import rules
from validation.validate import EXCEPTION, validate_record

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "data" / "hard"
TRUTH = load_truth("hard")


def _digests(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def test_regenerating_reproduces_committed_hard_split(tmp_path):
    hard.generate(tmp_path)
    assert _digests(tmp_path) == _digests(HARD)


def test_refuses_to_write_into_main_split():
    with pytest.raises(SystemExit, match="frozen"):
        hard.generate(ROOT / "data")


def test_composition():
    layouts = {d["layout"] for d in TRUTH}
    assert layouts == set(hard._CHALLENGES) and len(TRUTH) == 10
    assert [d["layout"] for d in TRUTH if d["expected_exception"]] == ["soi_pct_broken"]
    assert [d["layout"] for d in TRUTH if d["doc_type"] == "UNKNOWN"] == ["k1_cover"]
    assert all(d["challenge"] for d in TRUTH)


def test_truth_through_validator_matches_expected_exception():
    for d in TRUTH:
        rec = {"file": d["file"], "route": {"doc_type": d["doc_type"], "confidence": 0.95, "reason": ""},
               "extracted": {**d["fields"], "confidence": 0.95} if d["fields"] else None, "error": None}
        status = validate_record(rec).status
        assert (status == EXCEPTION) == (d["expected_exception"] or d["doc_type"] == "UNKNOWN"), d["doc_id"]


def test_footer_units_doc_infers_thousands_step():
    d = next(x for x in TRUTH if x["layout"] == "cas_footer_units")
    assert rules.cas_step(d["fields"]) == 1_000


def test_computed_letter_exposures_are_consistent():
    f = next(x for x in TRUTH if x["layout"] == "letter_pct_of_gross")["fields"]
    assert abs(f["long_exposure"] + f["short_exposure"] - f["gross_exposure"]) < 1e-9
    assert abs(f["long_exposure"] - f["short_exposure"] - f["net_exposure"]) < 1e-9


def _write_log(dirpath: Path, ts: str, split: str | None, model: str) -> Path:
    header = {"timestamp": ts, "model": model, "prompt_versions": {}}
    if split:
        header["split"] = split
    p = dirpath / f"{ts}_pipeline.jsonl"
    p.write_text(json.dumps(header) + "\n")
    return p


def test_latest_run_log_filters_by_split_and_model(tmp_path):
    old_main = _write_log(tmp_path, "20260101T000000Z", None, "m1")      # pre-split logs count as main
    hard_m1 = _write_log(tmp_path, "20260102T000000Z", "hard", "m1")
    hard_m2 = _write_log(tmp_path, "20260103T000000Z", "hard", "m2")
    assert regression.latest_run_log(tmp_path, "main") == old_main
    assert regression.latest_run_log(tmp_path, "hard") == hard_m2
    assert regression.latest_run_log(tmp_path, "hard", model="m1") == hard_m1


def test_compare_models_picks_newest_run_of_each_model():
    paths = newest_per_model("hard")
    models = [json.loads(p.open().readline())["model"] for p in paths]
    assert len(models) == len(set(models))
