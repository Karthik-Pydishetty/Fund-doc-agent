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

from evals.scoring import compute_metrics, load_log, split_of

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"
BASELINES = {"main": BASELINE, "hard": ROOT / "evals" / "baseline_hard.json"}
RUNS = ROOT / "runs"
FIELD_DROP_POINTS = 5.0
EPS = 1e-9


def latest_run_log(runs_dir: Path = RUNS, split: str = "main", model: str | None = None) -> Path | None:
    """Newest pipeline log for a split (and model, if given), read from each log's header line."""
    for log in sorted(runs_dir.glob("*_pipeline.jsonl"), reverse=True):
        with log.open() as fh:
            header = json.loads(fh.readline())
        if split_of(header) == split and model in (None, header.get("model")):
            return log
    return None


def gate_log(split: str = "main") -> Path | None:
    """The log the regression gate checks: newest run of this split by the baseline's model."""
    base = BASELINES[split]
    if not base.exists():
        return None
    return latest_run_log(split=split, model=load_baseline(base).get("model"))


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


def check_log(path: Path, baseline_path: Path | None = None) -> tuple[dict, list[str]]:
    """Score a log against the baseline for its own split (from the log header)."""
    header, records, _ = load_log(path)
    split = split_of(header)
    metrics = compute_metrics(records, split=split)
    return metrics, find_regressions(metrics, load_baseline(baseline_path or BASELINES[split]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="pipeline log (default: newest run log for --split)")
    ap.add_argument("--split", default="main", choices=sorted(BASELINES))
    args = ap.parse_args()
    path = Path(args.log).resolve() if args.log else gate_log(args.split)
    if path is None:
        sys.exit(f"no {args.split}-split runs/*_pipeline.jsonl to check")
    base = BASELINES[split_of(load_log(path)[0])]
    _, problems = check_log(path)
    if problems:
        print(f"REGRESSION in {path.name} vs {base.relative_to(ROOT)}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: {path.name} has no regressions vs {base.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
