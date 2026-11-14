"""Compare a run's metrics with evals/baseline.json. Offline.

    python -m evals.regression [--log PATH]      # exit 1 and list every regression

Fails when:
  - any field's accuracy drops more than FIELD_DROP_POINTS percentage points vs baseline
    (a baseline field missing from the run counts as a drop to 0)
  - router accuracy drops at all
  - exception recall drops at all
Metrics are recomputed from the log's records (evals/scoring.py), never trusted from the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.scoring import compute_metrics, load_log

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"
RUNS = ROOT / "runs"
FIELD_DROP_POINTS = 5.0
EPS = 1e-9


def latest_run_log(runs_dir: Path = RUNS) -> Path | None:
    logs = sorted(runs_dir.glob("*_pipeline.jsonl"))
    return logs[-1] if logs else None


def load_baseline(path: Path = BASELINE) -> dict:
    return json.loads(path.read_text())


def find_regressions(current: dict, baseline: dict) -> list[str]:
    problems = []
    b, c = baseline["metrics"], current
    if c["router_accuracy"] < b["router_accuracy"] - EPS:
        problems.append(f"router accuracy dropped: {b['router_accuracy']:.1%} -> {c['router_accuracy']:.1%}")
    br, cr = b["exceptions"]["recall"], c["exceptions"]["recall"]
    if cr < br - EPS:
        problems.append(f"exception recall dropped: {br:.2f} -> {cr:.2f} (missed: {c['exceptions']['fn']})")
    for field, b_acc in sorted(b["field_accuracy"].items()):
        c_acc = c["field_accuracy"].get(field)
        if c_acc is None:
            problems.append(f"field {field}: missing from this run (baseline {b_acc:.1%})")
            continue
        drop = (b_acc - c_acc) * 100
        if drop > FIELD_DROP_POINTS + EPS:
            bad = sorted({m["doc_id"] for m in c.get("mismatches", []) if m.get("group") == field})
            problems.append(f"field {field} accuracy dropped {drop:.1f} points: {b_acc:.1%} -> {c_acc:.1%}"
                            + (f" (failing docs: {', '.join(bad)})" if bad else ""))
    return problems


def check_log(path: Path, baseline_path: Path = BASELINE) -> tuple[dict, list[str]]:
    _, records, _ = load_log(path)
    metrics = compute_metrics(records)
    return metrics, find_regressions(metrics, load_baseline(baseline_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="pipeline log (default: newest runs/*_pipeline.jsonl)")
    args = ap.parse_args()
    path = Path(args.log).resolve() if args.log else latest_run_log()
    if path is None:
        sys.exit("no runs/*_pipeline.jsonl to check")
    _, problems = check_log(path)
    if problems:
        print(f"REGRESSION in {path.name} vs {BASELINE.relative_to(ROOT)}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: {path.name} has no regressions vs {BASELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
