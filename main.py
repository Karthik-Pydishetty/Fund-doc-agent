"""Run the full pipeline over a folder of PDFs and log the run.

    python main.py [--data data] [--workers 5] [FILE ...]

Writes runs/<timestamp>_pipeline.jsonl: a header line (timestamp, provider, model, prompt
versions, counts) and one line per document (route, extraction, errors, token usage).
Reads PDFs only; never ground truth.
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

    routes = Counter(r["route"]["doc_type"] if r["route"] else "ERROR" for r in records)
    errors = [r for r in records if r["error"]]
    tokens = [c for r in records for c in r["calls"].values()]
    header = {
        "timestamp": started.isoformat(), "kind": "pipeline", "provider": client.provider, "model": client.model,
        "prompt_versions": prompt_versions(), "n_docs": len(records), "routes": dict(routes),
        "n_errors": len(errors),
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
    print(f"Tokens: {header['usage']['input_tokens']:,} in / {header['usage']['output_tokens']:,} out")
    print(f"Run logged to {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
