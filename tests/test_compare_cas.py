"""compare_cas --from-log: rebuilds the table offline and still flags mismatches."""
import json
import subprocess
import sys
from pathlib import Path

from evals import compare_cas

ROOT = Path(__file__).resolve().parents[1]


def _write_log(path: Path, tamper: dict[str, dict] | None = None) -> None:
    truth = compare_cas.cas_truth()
    rows = []
    for d in truth:
        out = {**d["fields"], "confidence": 0.9, **(tamper or {}).get(d["doc_id"], {})}
        rows.append({"doc_id": d["doc_id"], "extracted": out, "error": None, "call": None})
    header = {"timestamp": "2026-01-01T00:00:00+00:00", "model": "m", "prompt_versions": {"cas": "cas_vX"}}
    path.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")


def test_from_log_flags_mismatch_without_api(tmp_path):
    first = compare_cas.cas_truth()[0]
    log = tmp_path / "20260101T000000Z_compare_cas.jsonl"
    _write_log(log, {first["doc_id"]: {"ending_nav": 1.0, "currency": "XXX"}})
    # Fresh interpreter with the anthropic SDK made unimportable: --from-log must not need it
    code = ("import sys; sys.modules['anthropic'] = None; sys.argv = ['x', '--from-log', %r, '--no-color'];"
            "from evals.compare_cas import main; main()") % str(log)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.count("✗ MISMATCH") == 2
    assert "ending_nav           1 " in out
    assert "Overall: 148/150 fields" in out and "9/10 docs fully correct" in out
    assert "no API calls made" in out


def test_latest_log_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_cas, "RUNS", tmp_path)
    for ts in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
        _write_log(tmp_path / f"{ts}_compare_cas.jsonl")
    assert compare_cas.latest_log().name.startswith("20260301")
