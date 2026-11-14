"""Score validation against ground truth's expected_exception. Offline: validates a logged pipeline run.

    python -m evals.exceptions [--log PATH]

Primary metric: precision/recall of EXCEPTION vs expected_exception over docs whose true type is
CAS/SOI/LETTER (the intentionally broken docs). UNKNOWN docs are scored separately: they should all
land in exceptions with reason unknown_doc_type, but they are not "broken data", so counting them in
the primary metric would blur what the finance rules catch. Definitions live in evals/scoring.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evals.scoring import exception_scores as score, load_log, load_truth, split_of  # `score` kept for callers
from validation.run import latest_pipeline_log
from validation.validate import EXCEPTION, validate_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    args = ap.parse_args()
    path = Path(args.log).resolve() if args.log else latest_pipeline_log()
    header, recs, _ = load_log(path)
    results = validate_records(recs)
    truth = load_truth(split_of(header))
    tmap = {d["doc_id"]: d for d in truth}
    s = score(results, truth)

    print(f"Run {path.name}: {len(results)} docs")
    print(f"{'doc':<8} {'true type':<9} {'expected':<10} {'status':<10} {'conf':<5} reasons")
    for r in results:
        d = tmap[r.doc_id]
        exp = "EXCEPTION" if d["expected_exception"] else "-"
        flag = "" if (r.status == EXCEPTION) == (d["expected_exception"] or d["doc_type"] == "UNKNOWN") else "  <-- WRONG"
        print(f"{r.doc_id:<8} {d['doc_type']:<9} {exp:<10} {r.status:<10} {str(r.confidence):<5} "
              f"{', '.join(i.rule for i in r.issues)}{flag}")

    print(f"\nBroken-data exceptions (CAS/SOI/LETTER docs): precision {s['precision']:.2f} "
          f"({len(s['tp'])}/{len(s['tp']) + len(s['fp'])}), recall {s['recall']:.2f} "
          f"({len(s['tp'])}/{len(s['tp']) + len(s['fn'])})")
    print(f"  false positives (clean docs flagged): {s['fp'] or 'none'}")
    print(f"  false negatives (broken docs missed): {s['fn'] or 'none'}")
    print(f"UNKNOWN docs sent to exceptions as unknown_doc_type: {len(s['unknown_flagged'])}/{len(s['unknown'])}")


if __name__ == "__main__":
    main()
