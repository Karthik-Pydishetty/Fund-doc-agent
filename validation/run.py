"""Validate a logged pipeline run offline (no API calls) and write exceptions_report.csv.

    python -m validation.run [--log runs/<ts>_pipeline.jsonl] [--out exceptions_report.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.validate import EXCEPTION, validate_records, write_exceptions_report

ROOT = Path(__file__).resolve().parents[1]


def latest_pipeline_log() -> Path:
    logs = sorted((ROOT / "runs").glob("*_pipeline.jsonl"))
    if not logs:
        sys.exit("no runs/*_pipeline.jsonl; run `python main.py` first")
    return logs[-1]


def load_records(path: Path) -> tuple[dict, list[dict]]:
    header, *rest = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return header, [r for r in rest if "file" in r]  # skip appended eval records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="pipeline log (default: newest runs/*_pipeline.jsonl)")
    ap.add_argument("--out", default=str(ROOT / "exceptions_report.csv"))
    args = ap.parse_args()

    path = Path(args.log).resolve() if args.log else latest_pipeline_log()
    _, recs = load_records(path)
    results = validate_records(recs)
    n_rows = write_exceptions_report(results, args.out)

    exc = [r for r in results if r.status == EXCEPTION]
    print(f"Validated {len(results)} docs from {path.name}: {len(results) - len(exc)} ACCEPTED, {len(exc)} EXCEPTION")
    for r in exc:
        print(f"  {r.doc_id} {r.doc_type or '-':<8} conf={r.confidence}  " + "; ".join(i.rule for i in r.issues))
    print(f"{n_rows} row(s) written to {args.out}")


if __name__ == "__main__":
    main()
