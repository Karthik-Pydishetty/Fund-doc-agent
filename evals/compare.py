"""Print a logged pipeline run scored against ground truth. No API calls.

    python -m evals.compare [--log PATH] [--doc doc_007] [--no-color]

Router confusion matrix, per-field accuracy by doc type, and every mismatch. Metric definitions
live in evals/scoring.py.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from evals.scoring import (  # noqa: F401  (flatten/match/score_doc re-exported for callers)
    TYPES, compute_metrics, flatten, load_log, load_truth, match, score_doc,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def latest_log() -> Path:
    logs = sorted(RUNS.glob("*_pipeline.jsonl"))
    if not logs:
        sys.exit("no runs/*_pipeline.jsonl; run `python main.py` first")
    return logs[-1]


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="pipeline log (default: newest runs/*_pipeline.jsonl)")
    ap.add_argument("--doc", help="print every field for one doc_id")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    red, green, bold, reset = (("", "", "", "") if args.no_color or not sys.stdout.isatty()
                               else ("\033[31m", "\033[32m", "\033[1m", "\033[0m"))

    path = Path(args.log).resolve() if args.log else latest_log()
    header, recs, _ = load_log(path)
    by_file = {r["file"]: r for r in recs}
    truth = [d for d in load_truth() if d["file"] in by_file]

    if args.doc:
        d = next(x for x in truth if x["doc_id"] == args.doc)
        r = by_file[d["file"]]
        print(f"{bold}{d['doc_id']} truth={d['doc_type']} routed={r['route'] and r['route']['doc_type']} "
              f"layout={d['layout']}{reset}  {r['route'] and r['route']['reason']}")
        for k, g, ev, tv, ok in score_doc(d, r):
            mark = f"{green}✓{reset}" if ok else f"{red}✗ MISMATCH{reset}"
            print(f"  {k:<60} {fmt(ev):<22} {fmt(tv):<22} {mark}")
        if r.get("extracted"):
            print(f"  confidence {r['extracted'].get('confidence')}")
            if "commentary_summary" in r["extracted"]:
                print(f"  commentary_summary (not scored): {r['extracted']['commentary_summary']}")
        return

    m = compute_metrics(recs, truth)
    print(f"{bold}Run {path.name}{reset}  model={header['model']}  prompts={header['prompt_versions']}")

    print(f"\n{bold}Router accuracy: {m['router_accuracy']:.1%} ({m['n_docs']} docs){reset}")
    conf = m["router_confusion"]
    cols = TYPES + (["ERROR"] if any(k.endswith("->ERROR") for k in conf) else [])
    print("  truth \\ pred " + "".join(f"{c:>9}" for c in cols))
    for t in TYPES:
        print(f"  {t:<13}" + "".join(f"{conf.get(f'{t}->{p}', 0):>9}" for p in cols))
    for k, n in conf.items():
        t, p = k.split("->")
        if t != p:
            print(f"  {red}MISROUTED {n} {t} doc(s) -> {p}{reset}")

    by_type: dict[str, list] = defaultdict(list)
    for k, (ok, n) in m["field_counts"].items():
        t, field = k.split(".", 1)
        by_type[t].append((field, ok, n))
    for t in ("CAS", "SOI", "LETTER"):
        if t not in by_type:
            continue
        ok_t = sum(x[1] for x in by_type[t])
        n_t = sum(x[2] for x in by_type[t])
        print(f"\n{bold}{t}: {ok_t}/{n_t} fields ({100 * ok_t / n_t:.1f}%){reset}")
        for field, ok, n in by_type[t]:
            print(f"  {field:<34} {'' if ok == n else red}{ok}/{n} ({100 * ok / n:.0f}%){reset}")
    print(f"\n{bold}All extracted fields: {m['field_accuracy_overall']:.1%}{reset}")

    print(f"\n{bold}Mismatches ({len(m['mismatches'])}){reset}")
    for x in m["mismatches"]:
        truth_s = "(not in ground truth)" if x["truth"] is None else fmt(x["truth"])
        print(f"  {red}✗{reset} {x['doc_id']}  {x['field']:<58} extracted={fmt(x['extracted'])}  truth={truth_s}")
    for r in recs:
        if r.get("error"):
            print(f"  {red}ERROR{reset} {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
