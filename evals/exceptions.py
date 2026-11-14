"""Score validation against ground truth's expected_exception. Offline: validates a logged pipeline run.

    python -m evals.exceptions [--log PATH]

Primary metric: precision/recall of EXCEPTION vs expected_exception over docs whose true type is
CAS/SOI/LETTER (the intentionally broken docs). UNKNOWN docs are scored separately: they should all
land in exceptions with reason unknown_doc_type, but they are not "broken data", so counting them in
the primary metric would blur what the finance rules catch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.run import latest_pipeline_log, load_records
from validation.validate import EXCEPTION, validate_records

ROOT = Path(__file__).resolve().parents[1]


def score(results, truth_docs) -> dict:
    by_id = {r.doc_id: r for r in results}
    known = [d for d in truth_docs if d["doc_type"] != "UNKNOWN" and d["doc_id"] in by_id]
    tp = [d["doc_id"] for d in known if d["expected_exception"] and by_id[d["doc_id"]].status == EXCEPTION]
    fp = [d["doc_id"] for d in known if not d["expected_exception"] and by_id[d["doc_id"]].status == EXCEPTION]
    fn = [d["doc_id"] for d in known if d["expected_exception"] and by_id[d["doc_id"]].status != EXCEPTION]
    unknown = [d["doc_id"] for d in truth_docs if d["doc_type"] == "UNKNOWN" and d["doc_id"] in by_id]
    unknown_flagged = [i for i in unknown if any(x.rule == "unknown_doc_type" for x in by_id[i].issues)]
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": len(tp) / (len(tp) + len(fp)) if tp or fp else 1.0,
        "recall": len(tp) / (len(tp) + len(fn)) if tp or fn else 1.0,
        "unknown": unknown, "unknown_flagged": unknown_flagged,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    args = ap.parse_args()
    path = Path(args.log).resolve() if args.log else latest_pipeline_log()
    _, recs = load_records(path)
    results = validate_records(recs)
    truth = json.loads((ROOT / "data/ground_truth.json").read_text())["documents"]
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
