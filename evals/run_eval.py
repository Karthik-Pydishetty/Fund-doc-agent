"""One command: run the full pipeline (API calls), score it, and append the eval record to its log.

    python -m evals.run_eval [--prompt cas=v2 ...] [--workers 5] [FILE ...]
    python -m evals.run_eval --from-log [PATH]       # score an existing log, no API calls
    python -m evals.run_eval ... --save-baseline     # also write evals/baseline.json from this run

The appended line ({"kind": "eval", ...}) holds the metrics, git commit (with a dirty flag),
model and prompt versions. Afterwards it checks for regressions against evals/baseline.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from evals.regression import BASELINE, find_regressions, latest_run_log
from evals.scoring import compute_metrics, load_log

ROOT = Path(__file__).resolve().parents[1]


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def append_eval(log: Path, header: dict, metrics: dict) -> dict:
    rec = {"kind": "eval", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "git": git_info(),
           "model": header["model"], "prompt_versions": header["prompt_versions"], "metrics": metrics}
    with log.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def save_baseline(log: Path, rec: dict, path: Path = BASELINE) -> None:
    m = {k: rec["metrics"][k] for k in ("n_docs", "router_accuracy", "field_accuracy_overall",
                                         "doc_type_accuracy", "field_accuracy", "exceptions")}
    out = {"source_log": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),
           "created": rec["timestamp"], "git": rec["git"], "model": rec["model"],
           "prompt_versions": rec["prompt_versions"], "metrics": m}
    path.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")


def print_report(rec: dict) -> None:
    m = rec["metrics"]
    print(f"\nEval  model={rec['model']}  prompts={rec['prompt_versions']}  "
          f"commit={(rec['git']['commit'] or '?')[:8]}{' (dirty)' if rec['git']['dirty'] else ''}")
    print(f"  router accuracy      {m['router_accuracy']:.1%}  ({m['n_docs']} docs)")
    print(f"  field accuracy       {m['field_accuracy_overall']:.1%}  "
          + "  ".join(f"{t} {a:.1%}" for t, a in m["doc_type_accuracy"].items()))
    e = m["exceptions"]
    print(f"  exceptions           precision {e['precision']:.2f}  recall {e['recall']:.2f}  "
          f"(fp {e['fp'] or 'none'}, fn {e['fn'] or 'none'});  UNKNOWN flagged {m['unknown_flagged']}")
    below = {k: v for k, v in m["field_accuracy"].items() if v < 1}
    print("  fields below 100%:   " + (", ".join(f"{k} {v:.0%}" for k, v in below.items()) or "none"))


def main() -> None:
    from main import add_pipeline_args  # shares --data/--workers/--prompt/files with main.py

    ap = argparse.ArgumentParser()
    add_pipeline_args(ap)
    ap.add_argument("--from-log", nargs="?", const="latest", metavar="PATH",
                    help="score an existing pipeline log instead of running the pipeline")
    ap.add_argument("--save-baseline", action="store_true", help=f"write {BASELINE.relative_to(ROOT)}")
    args = ap.parse_args()

    if args.from_log:
        log = latest_run_log() if args.from_log == "latest" else Path(args.from_log).resolve()
        if log is None:
            sys.exit("no runs/*_pipeline.jsonl")
        header, records, _ = load_log(log)
    else:
        from agents.llm_client import get_client
        from agents.pipeline import resolve_prompts
        from main import pdfs_from_args, print_summary, run_pipeline
        try:
            prompts = resolve_prompts(args.prompt)
        except ValueError as e:
            ap.error(str(e))
        log, header, records = run_pipeline(pdfs_from_args(args), get_client(), prompts, args.workers)
        print_summary(log, header, records)

    rec = append_eval(log, header, compute_metrics(records))
    print_report(rec)
    print(f"Eval record appended to {log.relative_to(ROOT) if log.is_relative_to(ROOT) else log}")

    if args.save_baseline:
        save_baseline(log, rec)
        print(f"Baseline written to {BASELINE.relative_to(ROOT)}")
    elif BASELINE.exists():
        problems = find_regressions(rec["metrics"], json.loads(BASELINE.read_text()))
        print("Regression check vs baseline: " + ("OK" if not problems else "FAILED"))
        for p in problems:
            print(f"  - {p}")
        if problems:
            sys.exit(1)


if __name__ == "__main__":
    main()
