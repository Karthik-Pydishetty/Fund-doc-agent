"""Blind spot before vs after, per injected fault, for two seeded fault-injection runs. Offline.

    python -m evals.blind_spot BEFORE_LOG AFTER_LOG

Both runs must use the same split, fault seed and fault count, so they inject the same faults
(faults are chosen by the arithmetic rules only). For each fault the outcome is one of:
  recovered    final value equals ground truth (the retry fixed it)
  flagged      still wrong, but the document ends as an EXCEPTION, so a person sees it
  undetected   still wrong and ACCEPTED: the blind spot
Exception precision/recall are reported over documents WITHOUT an injected fault: an injected-fault
document flagged for review is a correct flag (its extraction really is wrong), even though ground
truth calls the source document clean.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agents.self_correct import flatten
from evals.scoring import exception_scores, field_group, load_log, load_truth, match, split_of
from validation.validate import validate_records

ROOT = Path(__file__).resolve().parents[1]


def outcomes(log: Path) -> tuple[dict, dict]:
    header, records, _ = load_log(log)
    truth = {d["file"]: d for d in load_truth(split_of(header))}
    out = {}
    for r in records:
        fi = (r.get("fault_injection") or {}).get("injected")
        if not fi:
            continue
        d = truth[r["file"]]
        final = flatten(r["extracted"] or {}).get(fi["field"])
        expected = flatten(d["fields"]).get(fi["field"])
        status = r["validation"]["status"]
        verdict = ("recovered" if match(final, expected, field_group(fi["field"]))
                   else "flagged" if status == "EXCEPTION" else "undetected")
        rules = [i["rule"] for i in (r.get("self_correction") or {}).get("first_issues", [])] or \
            (r.get("self_correction") or {}).get("triggered_by", [])
        key = (d["doc_id"], fi["field"], fi["kind"], str(fi["original"]), str(fi["corrupted"]))
        out[key] = {"verdict": verdict, "caught_by": rules, "final": final, "status": status}
    faulted = {r["file"] for r in records if (r.get("fault_injection") or {}).get("injected")}
    clean_recs = [r for r in records if r["file"] not in faulted]
    docs = [d for d in truth.values() if d["file"] not in faulted]
    exc = exception_scores(validate_records(clean_recs), docs)
    return out, {"header": header, "precision": exc["precision"], "recall": exc["recall"],
                 "fp": exc["fp"], "fn": exc["fn"]}


def fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def report(before: Path, after: Path) -> str:
    b, bm = outcomes(before)
    a, am = outcomes(after)
    hb, ha = bm["header"], am["header"]
    same = (split_of(hb) == split_of(ha) and hb.get("fault_summary", {}).get("seed") == ha.get("fault_summary", {}).get("seed")
            and hb.get("inject_faults") == ha.get("inject_faults"))
    lines = [f"=== {split_of(ha)} split: {before.relative_to(ROOT)} (before) vs {after.relative_to(ROOT)} (after) ===",
             f"seed {ha.get('fault_summary', {}).get('seed')}, {ha.get('inject_faults')} faults requested; "
             f"same faults injected in both runs: {'yes' if set(a) == set(b) and same else 'NO'}", ""]
    w = max(len(f"{k[0]} {k[1]}") for k in a) + 2
    lines.append(f"{'fault':<{w}} {'kind':<16} {'before':<11} {'after':<11} caught by (after)")
    for k in sorted(set(a) | set(b)):
        bo, ao = b.get(k, {}).get("verdict", "-"), a.get(k, {}).get("verdict", "-")
        mark = "  <- closed" if bo == "undetected" and ao != "undetected" else ""
        lines.append(f"{k[0] + ' ' + k[1]:<{w}} {k[2]:<16} {bo:<11} {ao:<11} "
                     f"{', '.join(a.get(k, {}).get('caught_by', [])) or '-'}{mark}")
    for name, o in (("before", b), ("after", a)):
        n = len(o)
        c = {v: sum(x["verdict"] == v for x in o.values()) for v in ("recovered", "flagged", "undetected")}
        lines.append(f"\n{name:<6}: recovered {c['recovered']}/{n}, flagged {c['flagged']}/{n}, "
                     f"undetected (blind spot) {c['undetected']}/{n}")
    lines.append(f"\nException precision / recall on documents without an injected fault: "
                 f"before {bm['precision']:.2f} / {bm['recall']:.2f}, after {am['precision']:.2f} / {am['recall']:.2f}"
                 f"  (after: fp {am['fp'] or 'none'}, fn {am['fn'] or 'none'})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    print(report(Path(args.before).resolve(), Path(args.after).resolve()))


if __name__ == "__main__":
    main()
