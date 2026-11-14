"""Run the full pipeline over a folder of PDFs: route -> extract -> validate, and log the run.

    python main.py [--data data] [--workers 5] [FILE ...]

Writes runs/<timestamp>_pipeline.jsonl: a header line (timestamp, provider, model, prompt
versions, counts) and one line per document (route, extraction, validation status and issues,
errors, token usage), plus exceptions_report.csv. Reads PDFs only; never ground truth.
To re-validate an existing log without API calls: python -m validation.run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.llm_client import get_client
from agents.pipeline import process, prompt_versions
from validation.validate import EXCEPTION, validate_records, write_exceptions_report

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("files", nargs="*", help="specific PDF names inside --data (default: all)")
    args = ap.parse_args()

    data = Path(args.data)
    pdfs = [data / f for f in args.files] if args.files else sorted(data.glob("*.pdf"))
    client = get_client()
    started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda p: process(p, client), pdfs))

    results = {v.file: v for v in validate_records(records)}
    for r in records:
        v = results[r["file"]]
        r["validation"] = {"status": v.status, "issues": [i.to_dict() for i in v.issues]}
    report = ROOT / "exceptions_report.csv"
    n_rows = write_exceptions_report(sorted(results.values(), key=lambda v: v.doc_id), report)
    n_exc = sum(v.status == EXCEPTION for v in results.values())

    routes = Counter(r["route"]["doc_type"] if r["route"] else "ERROR" for r in records)
    errors = [r for r in records if r["error"]]
    tokens = [c for r in records for c in r["calls"].values()]
    header = {
        "timestamp": started.isoformat(), "kind": "pipeline", "provider": client.provider, "model": client.model,
        "prompt_versions": prompt_versions(), "n_docs": len(records), "routes": dict(routes),
        "n_errors": len(errors), "n_exceptions": n_exc,
        "usage": {"input_tokens": sum(c["input_tokens"] for c in tokens),
                  "output_tokens": sum(c["output_tokens"] for c in tokens)},
    }
    RUNS.mkdir(exist_ok=True)
    log = RUNS / f"{started:%Y%m%dT%H%M%SZ}_pipeline.jsonl"
    with log.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"Processed {len(records)} docs with {client.model}; routes: {dict(routes)}")
    for r in errors:
        print(f"  ERROR {r['file']}: {r['error']}")
    print(f"Validation: {len(records) - n_exc} ACCEPTED, {n_exc} EXCEPTION; {n_rows} row(s) in {report.name}")
    print(f"Tokens: {header['usage']['input_tokens']:,} in / {header['usage']['output_tokens']:,} out")
    print(f"Run logged to {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
