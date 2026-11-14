"""Report on a self-correction / fault-injection run. Offline; reads ground truth.

    python -m evals.self_correction [LOG ...]     # default: newest fault-injected run of each split

Per run: injected faults and whether the retry recovered them (final value == ground truth),
retry changes rejected by the quote check (split into "fudge attempts" on genuinely broken docs
and rejected corrections on faulted docs), exception precision/recall, the extra cost of retries,
and the blind spot: injected faults that passed validation undetected (no rule covers the field).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PRICING_PER_MTOK
from evals.scoring import compute_metrics, field_group, load_log, load_truth, match, split_of

ROOT = Path(__file__).resolve().parents[1]


def newest_fault_runs() -> list[Path]:
    out: dict[str, Path] = {}
    for log in sorted((ROOT / "runs").glob("*_pipeline.jsonl"), reverse=True):
        h = json.loads(log.open().readline())
        if h.get("inject_faults") and h.get("self_correct") and split_of(h) not in out:
            out[split_of(h)] = log
    return [out[s] for s in sorted(out)]


def _usd(model: str, tin: int, tout: int) -> float | None:
    p = PRICING_PER_MTOK.get(model)
    return (tin * p[0] + tout * p[1]) / 1e6 if p else None


def _truth_value(fields: dict, path: str):
    from agents.self_correct import flatten  # same path format the injector used
    return flatten(fields).get(path)


def analyse(log: Path) -> dict:
    header, records, _ = load_log(log)
    split = split_of(header)
    truth = {d["file"]: d for d in load_truth(split)}
    m = compute_metrics(records, split=split)

    faults, fudges, rejected_fixes, accepted_on_broken, free_text = [], [], [], [], []
    for r in records:
        d = truth[r["file"]]
        sc = r.get("self_correction")
        fi = (r.get("fault_injection") or {}).get("injected")
        if fi:
            final = _truth_value(r["extracted"] or {}, fi["field"])
            expected = _truth_value(d["fields"], fi["field"])
            changes = [c for c in (sc or {}).get("changes", []) if c["field"] == fi["field"]]
            faults.append({"doc": d["doc_id"], **{k: fi[k] for k in ("field", "kind", "original", "corrupted",
                                                                  "category")},
                           "rules": fi["rules_broken"], "final": final, "truth": expected,
                           "recovered": match(final, expected, field_group(fi["field"])),
                           "retried": sc is not None, "change": changes[0] if changes else None,
                           "status_after": r["validation"]["status"]})
        for c in (sc or {}).get("changes", []):
            row = {"doc": d["doc_id"], **c}
            if c["field"] == "commentary_summary":  # free text; older logs diffed it, now ignored
                free_text.append(row)
                continue
            if d["expected_exception"]:
                (accepted_on_broken if c["accepted"] else fudges).append(row)
            elif not c["accepted"]:
                rejected_fixes.append(row)

    first = {"in": 0, "out": 0}
    retry = {"in": 0, "out": 0, "calls": 0}
    for r in records:
        for name, c in r.get("calls", {}).items():
            bucket = retry if name == "retry" else first
            bucket["in"] += c["input_tokens"]
            bucket["out"] += c["output_tokens"]
            if name == "retry":
                retry["calls"] += 1
    fs = header.get("fault_summary") or {}
    return {
        "log": log, "split": split, "model": header["model"], "metrics": m, "faults": faults,
        "fudges": fudges, "accepted_on_broken": accepted_on_broken, "rejected_fixes": rejected_fixes,
        "free_text": free_text,
        "retried_docs": [truth[r["file"]]["doc_id"] for r in records if r.get("self_correction")],
        "first": first, "retry": retry,
        "first_usd": _usd(header["model"], first["in"], first["out"]),
        "retry_usd": _usd(header["model"], retry["in"], retry["out"]),
        "retry_wall_s": header.get("retry_wall_s"), "wall_s": header.get("wall_s"),
        "fault_summary": fs,
    }


def fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def report(a: dict) -> str:
    m, fs = a["metrics"], a["fault_summary"]
    e = m["exceptions"]
    rec = sum(f["recovered"] for f in a["faults"])
    covered = [f for f in a["faults"] if f["category"] == "covered"]
    uncovered = [f for f in a["faults"] if f["category"] != "covered"]
    text = [f for f in a["faults"] if f["category"] == "uncovered_text"]
    undetected = [f for f in a["faults"] if not f["recovered"] and f["status_after"] == "ACCEPTED"]
    lines = [
        f"=== {a['split']} split  ({a['log'].relative_to(ROOT)}, {a['model']}) ===",
        f"Injected faults:                    {len(a['faults'])} ({len(covered)} that break a finance rule, "
        f"{len(text)} in fields no rule covers (names, dates, quarter), "
        f"{len(uncovered) - len(text)} numeric corruptions no rule catches)",
        f"Recovered by the retry:             {rec}/{len(a['faults'])} "
        f"(rule-covered {sum(f['recovered'] for f in covered)}/{len(covered)}, "
        f"uncovered {sum(f['recovered'] for f in uncovered)}/{len(uncovered)})",
        f"Passed validation undetected:       {len(undetected)}/{len(a['faults'])}  <- blind spot: "
        + (", ".join(f"{f['doc']} {f['field']}" for f in undetected) or "none"),
        f"Docs retried:                       {len(a['retried_docs'])} ({', '.join(a['retried_docs'])})",
        f"Fudge attempts rejected (broken docs): {len(a['fudges'])}"
        f"   accepted changes on broken docs: {len(a['accepted_on_broken'])}",
        f"Corrections rejected on faulted docs: {len(a['rejected_fixes'])}"
        f"   (commentary_summary rewrites kept at first pass, not values: {len(a['free_text'])})",
        f"Exception recall (genuinely broken): {e['recall']:.2f}   precision {e['precision']:.2f}"
        f"   (fn {e['fn'] or 'none'}, fp {e['fp'] or 'none'})",
        f"Field accuracy after retries:       {m['field_accuracy_overall']:.1%}",
        f"Extra cost of retries:              {a['retry']['calls']} calls, {a['retry']['in']:,} in / "
        f"{a['retry']['out']:,} out tokens, ${a['retry_usd']:.4f} vs ${a['first_usd']:.4f} first pass "
        f"(+{100 * a['retry_usd'] / a['first_usd']:.0f}%); {a['retry_wall_s']} s of {a['wall_s']} s wall clock",
        "",
        "Injected faults:",
    ]
    for f in a["faults"]:
        ch = f["change"]
        how = ("not retried (no rule failed)" if not f["retried"] else "no change proposed" if not ch else
               f"change {'ACCEPTED' if ch['accepted'] else 'REJECTED'} ({ch['reason']}); quote: {ch['quote']!r}")
        lines.append(f"  {f['doc']} [{f['category']}] {f['field']:<40} {f['kind']:<15} "
                     f"{fmt(f['original'])} -> {fmt(f['corrupted'])}"
                     f"  breaks {f['rules']}  final={fmt(f['final'])} truth={fmt(f['truth'])} "
                     f"{'RECOVERED' if f['recovered'] else 'NOT RECOVERED'}; {how}; status {f['status_after']}")
    for title, rows in (("Fudge attempts rejected on genuinely broken docs:", a["fudges"]),
                        ("Changes ACCEPTED on genuinely broken docs:", a["accepted_on_broken"]),
                        ("Corrections rejected on faulted docs:", a["rejected_fixes"])):
        lines.append(f"\n{title}" + ("" if rows else " none"))
        for c in rows:
            lines.append(f"  {c['doc']} {c['field']}: {fmt(c['old'])} -> {fmt(c['new'])}  "
                         f"({c['reason']}); quote: {c['quote']!r}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    args = ap.parse_args()
    logs = [Path(p).resolve() for p in args.logs] or newest_fault_runs()
    if not logs:
        raise SystemExit("no self-correct + fault-injection runs found")
    print("\n\n".join(report(analyse(p)) for p in logs))


if __name__ == "__main__":
    main()
