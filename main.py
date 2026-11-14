"""Run the full pipeline over a folder of PDFs: route -> extract -> validate, and log the run.

    python main.py [--data data] [--workers 5] [--prompt cas=v2 ...] [FILE ...]

Writes runs/<timestamp>_pipeline.jsonl: a header line (timestamp, provider, model, prompt
versions, counts) and one line per document (route, extraction, validation status and issues,
errors, token usage), plus exceptions_report.csv. Reads PDFs only; never ground truth.
To re-validate an existing log without API calls: python -m validation.run
To run and score in one step: python -m evals.run_eval
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.llm_client import LLMClient, get_client
from agents.pipeline import process, resolve_prompts
from validation.validate import EXCEPTION, validate_records, write_exceptions_report

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORT = ROOT / "exceptions_report.csv"


def run_pipeline(pdfs: list[Path], client: LLMClient, prompts: dict[str, str], workers: int = 5,
                 runs_dir: Path = RUNS, report: Path = REPORT) -> tuple[Path, dict, list[dict]]:
    started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(lambda p: process(p, client, prompts), pdfs))

    results = {v.file: v for v in validate_records(records)}
    for r in records:
        v = results[r["file"]]
        r["validation"] = {"status": v.status, "issues": [i.to_dict() for i in v.issues]}
    write_exceptions_report(sorted(results.values(), key=lambda v: v.doc_id), report)

    calls = [c for r in records for c in r["calls"].values()]
    header = {
        "timestamp": started.isoformat(), "kind": "pipeline", "provider": client.provider, "model": client.model,
        "prompt_versions": prompts, "n_docs": len(records),
        "routes": dict(Counter(r["route"]["doc_type"] if r["route"] else "ERROR" for r in records)),
        "n_errors": sum(bool(r["error"]) for r in records),
        "n_exceptions": sum(v.status == EXCEPTION for v in results.values()),
        "usage": {"input_tokens": sum(c["input_tokens"] for c in calls),
                  "output_tokens": sum(c["output_tokens"] for c in calls)},
    }
    runs_dir.mkdir(exist_ok=True)
    log = runs_dir / f"{started:%Y%m%dT%H%M%SZ}_pipeline.jsonl"
    with log.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return log, header, records


def add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--prompt", action="append", default=[], metavar="STAGE=VERSION",
                    help="use another prompt version, e.g. --prompt cas=v2 (repeatable; stages: "
                         "router, cas, soi, letter)")
    ap.add_argument("files", nargs="*", help="specific PDF names inside --data (default: all)")


def pdfs_from_args(args) -> list[Path]:
    data = Path(args.data)
    return [data / f for f in args.files] if args.files else sorted(data.glob("*.pdf"))


def print_summary(log: Path, header: dict, records: list[dict]) -> None:
    print(f"Processed {header['n_docs']} docs with {header['model']}; prompts {header['prompt_versions']}")
    print(f"Routes: {header['routes']}")
    for r in records:
        if r["error"]:
            print(f"  ERROR {r['file']}: {r['error']}")
    print(f"Validation: {header['n_docs'] - header['n_exceptions']} ACCEPTED, {header['n_exceptions']} EXCEPTION "
          f"-> {REPORT.name}")
    print(f"Tokens: {header['usage']['input_tokens']:,} in / {header['usage']['output_tokens']:,} out")
    print(f"Run logged to {log.relative_to(ROOT) if log.is_relative_to(ROOT) else log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_pipeline_args(ap)
    args = ap.parse_args()
    try:
        prompts = resolve_prompts(args.prompt)
    except ValueError as e:
        ap.error(str(e))
    log, header, records = run_pipeline(pdfs_from_args(args), get_client(), prompts, args.workers)
    print_summary(log, header, records)


if __name__ == "__main__":
    main()
