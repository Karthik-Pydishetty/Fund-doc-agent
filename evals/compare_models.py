"""Compare two or more logged runs side by side (e.g. Sonnet vs Haiku on the same split). Offline.

    python -m evals.compare_models runs/A_pipeline.jsonl runs/B_pipeline.jsonl
    python -m evals.compare_models --split hard      # newest run of each model on the split

Prints router accuracy, field accuracy per doc type and per field, exception precision/recall,
cost and latency for each run, then every missed field (extracted vs expected) and every
document whose ACCEPTED/EXCEPTION status disagrees with ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.readme import run_cost
from evals.scoring import compute_metrics, load_log, load_truth, split_of
from validation.validate import EXCEPTION, validate_records

ROOT = Path(__file__).resolve().parents[1]


def newest_per_model(split: str) -> list[Path]:
    seen: dict[str, Path] = {}
    for log in sorted((ROOT / "runs").glob("*_pipeline.jsonl"), reverse=True):
        header = json.loads(log.open().readline())
        if split_of(header) == split and header["model"] not in seen and not header.get("inject_faults"):
            seen[header["model"]] = log
    return [seen[m] for m in sorted(seen)]


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)


def load_run(path: Path) -> dict:
    header, records, _ = load_log(path)
    split = split_of(header)
    truth = load_truth(split)
    return {"path": path, "header": header, "records": records, "split": split, "truth": truth,
            "metrics": compute_metrics(records, truth), "cost": run_cost(header, records),
            "validation": {v.doc_id: v for v in validate_records(records)}}


def report(runs: list[dict]) -> str:
    names = [r["header"]["model"] for r in runs]
    w = max(22, *(len(n) + 2 for n in names))
    out = []

    def row(label, vals):
        out.append(f"{label:<44}" + "".join(f"{v:>{w}}" for v in vals))

    out.append(f"Split: {runs[0]['split']}   prompts: {runs[0]['header']['prompt_versions']}")
    out.append("Logs: " + ", ".join(str(r["path"].relative_to(ROOT)) for r in runs) + "\n")
    row("", names)
    row("Router accuracy", [f"{r['metrics']['router_accuracy']:.1%}" for r in runs])
    for t in ("CAS", "SOI", "LETTER"):
        vals = []
        for r in runs:
            fc = [v for k, v in r["metrics"]["field_counts"].items() if k.startswith(t + ".")]
            ok, n = sum(x[0] for x in fc), sum(x[1] for x in fc)
            vals.append(f"{ok / n:.1%} ({ok}/{n})" if n else "—")
        row(f"{t} field accuracy", vals)
    row("All fields", [f"{r['metrics']['field_accuracy_overall']:.1%}" for r in runs])
    row("Exception precision", [f"{r['metrics']['exceptions']['precision']:.2f}" for r in runs])
    row("Exception recall", [f"{r['metrics']['exceptions']['recall']:.2f}" for r in runs])
    row("UNKNOWN sent to exceptions", [r["metrics"]["unknown_flagged"] for r in runs])
    row("Tokens in / out", [f"{r['cost']['tin']:,} / {r['cost']['tout']:,}" for r in runs])
    row("Cost per run (list price)", [f"${r['cost']['usd']:.4f}" if r["cost"]["usd"] is not None else "n/a"
                                      for r in runs])
    row("Cost per doc", [f"${r['cost']['usd'] / r['cost']['n']:.5f}" if r["cost"]["usd"] is not None else "n/a"
                         for r in runs])
    row("Latency per doc, mean / max (s)", [f"{r['cost']['doc_mean']:.1f} / {r['cost']['doc_max']:.1f}"
                                             for r in runs])
    row("Wall clock (s)", [fmt(r["cost"]["wall_s"]) for r in runs])

    out.append("\nPer-field accuracy")
    fields = list(dict.fromkeys(k for r in runs for k in r["metrics"]["field_accuracy"]))
    for f in fields:
        row(f"  {f}", [f"{r['metrics']['field_accuracy'][f]:.0%}" if f in r["metrics"]["field_accuracy"] else "—"
                       for r in runs])

    out.append("\nPer document (challenge, route, fields wrong, confidence, validation vs expected)")
    tmap = {d["doc_id"]: d for d in runs[0]["truth"]}
    for doc_id, d in tmap.items():
        out.append(f"  {doc_id} [{d['doc_type']}] {d.get('challenge') or d['layout']}")
        for r in runs:
            rec = next(x for x in r["records"] if x["file"] == d["file"])
            wrong = sum(1 for m in r["metrics"]["mismatches"] if m["doc_id"] == doc_id)
            v = r["validation"][doc_id]
            expected = EXCEPTION if (d["expected_exception"] or d["doc_type"] == "UNKNOWN") else "ACCEPTED"
            verdict = "ok" if v.status == expected else f"WRONG (expected {expected})"
            conf = (rec.get("extracted") or {}).get("confidence", rec["route"]["confidence"] if rec["route"] else None)
            out.append(f"      {r['header']['model']:<20} route={rec['route']['doc_type'] if rec['route'] else 'ERROR':<8}"
                       f" fields wrong={wrong:<3} conf={conf}  {v.status} {[i.rule for i in v.issues]} {verdict}")

    out.append("\nMissed fields (extracted vs expected)")
    any_miss = False
    for r in runs:
        for m in r["metrics"]["mismatches"]:
            any_miss = True
            exp = "(not in ground truth)" if m["truth"] is None else fmt(m["truth"])
            out.append(f"  {r['header']['model']:<20} {m['doc_id']}  {m['field']:<50} "
                       f"extracted={fmt(m['extracted'])}  expected={exp}")
    if not any_miss:
        out.append("  none: every scored field matched ground truth for every model")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="pipeline logs to compare")
    ap.add_argument("--split", default="hard", help="with no logs: newest run of each model on this split")
    args = ap.parse_args()
    paths = [Path(p).resolve() for p in args.logs] or newest_per_model(args.split)
    if len(paths) < 2:
        raise SystemExit(f"need at least two runs to compare; found {len(paths)}")
    runs = [load_run(p) for p in paths]
    if len({r["split"] for r in runs}) > 1:
        raise SystemExit("runs are from different splits")
    print(report(runs))


if __name__ == "__main__":
    main()
