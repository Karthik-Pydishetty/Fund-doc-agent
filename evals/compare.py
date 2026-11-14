"""Score a logged pipeline run against ground truth. No API calls.

    python -m evals.compare [--log PATH] [--doc doc_007] [--no-color]

Reports router accuracy with a confusion matrix, per-field accuracy by doc type, and every
mismatch. List fields are matched by key (holding name, sector), not position:
`holdings[<name>].cost` is scored as field `holdings.cost`. A holding missing from the
extraction fails all of its fields; extra extracted items are listed. Scoring is end to end:
a misrouted document fails all of its fields. commentary_summary is not scored (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data", ROOT / "runs"
TYPES = ["CAS", "SOI", "LETTER", "UNKNOWN"]
LIST_KEYS = {"holdings": "name", "top_holdings": "name", "top_sector_exposures": "sector"}
UNSCORED = {"confidence", "commentary_summary"}
MONEY_TOL, PCT_TOL = 0.5, 0.05
PCT_HINTS = ("pct", "exposure", "return")


def flatten(fields: dict) -> dict[str, object]:
    flat = {}
    for k, v in fields.items():
        if k in UNSCORED:
            continue
        if k in LIST_KEYS and isinstance(v, list):
            key = LIST_KEYS[k]
            for item in v:
                ident = str(item.get(key, "")).strip()
                for sk, sv in item.items():
                    if sk != key:
                        flat[f"{k}[{ident}].{sk}"] = sv
                flat.setdefault(f"{k}[{ident}]", True)  # presence marker
        else:
            flat[k] = v
    return flat


def field_group(flat_key: str) -> str:
    if "[" not in flat_key:
        return flat_key
    head, rest = flat_key.split("[", 1)
    tail = rest.split("]", 1)[1]
    return head + (tail or "[present]")


def norm_key(flat_key: str) -> str:
    return flat_key.casefold()


def match(extracted, truth, group: str) -> bool:
    if extracted is None:
        return False
    if isinstance(truth, bool):
        return extracted is True
    if isinstance(truth, (int, float)):
        try:
            diff = abs(float(extracted) - truth)
        except (TypeError, ValueError):
            return False
        return diff <= (PCT_TOL if any(h in group for h in PCT_HINTS) else MONEY_TOL)
    return str(extracted).strip().casefold() == str(truth).strip().casefold()


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


def score_doc(truth_doc: dict, rec: dict | None):
    """Yield (flat_key, group, extracted, truth, ok) for every truth field, then extras with truth=None."""
    t_flat = flatten(truth_doc["fields"])
    routed = rec and rec.get("route") and rec["route"]["doc_type"] == truth_doc["doc_type"]
    e_flat = {norm_key(k): v for k, v in flatten(rec["extracted"]).items()} if routed and rec["extracted"] else {}
    for k, tv in t_flat.items():
        ev = e_flat.get(norm_key(k))
        g = field_group(k)
        yield k, g, ev, tv, match(ev, tv, g)
    t_keys = {norm_key(k) for k in t_flat}
    for k, ev in e_flat.items():
        if k not in t_keys and k.endswith("]"):
            yield k, "extra item", ev, None, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="pipeline log (default: newest runs/*_pipeline.jsonl)")
    ap.add_argument("--doc", help="print every field for one doc_id")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    red, green, bold, reset = (("", "", "", "") if args.no_color or not sys.stdout.isatty()
                               else ("\033[31m", "\033[32m", "\033[1m", "\033[0m"))

    path = Path(args.log).resolve() if args.log else latest_log()
    header, *recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by_file = {r["file"]: r for r in recs}
    truth = json.loads((DATA / "ground_truth.json").read_text())["documents"]
    truth = [d for d in truth if d["file"] in by_file]

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

    print(f"{bold}Run {path.name}{reset}  model={header['model']}  prompts={header['prompt_versions']}")

    # Router
    conf = Counter()
    misroutes = []
    for d in truth:
        r = by_file[d["file"]]
        pred = r["route"]["doc_type"] if r.get("route") else "ERROR"
        conf[(d["doc_type"], pred)] += 1
        if pred != d["doc_type"]:
            misroutes.append((d["doc_id"], d["doc_type"], pred, r["route"] and r["route"]["reason"]))
    correct = sum(v for (t, p), v in conf.items() if t == p)
    print(f"\n{bold}Router accuracy: {correct}/{len(truth)} ({100 * correct / len(truth):.1f}%){reset}")
    cols = TYPES + (["ERROR"] if any(p == "ERROR" for _, p in conf) else [])
    print("  truth \\ pred " + "".join(f"{c:>9}" for c in cols))
    for t in TYPES:
        print(f"  {t:<13}" + "".join(f"{conf[(t, p)]:>9}" for p in cols))
    for m in misroutes:
        print(f"  {red}MISROUTED {m[0]}: {m[1]} -> {m[2]}{reset} ({m[3]})")

    # Fields
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    mismatches = []
    for d in truth:
        if d["doc_type"] == "UNKNOWN":
            continue
        for k, g, ev, tv, ok in score_doc(d, by_file[d["file"]]):
            if g == "extra item":
                mismatches.append((d["doc_id"], k, ev, "(not in ground truth)"))
                continue
            stats[d["doc_type"]][g][0] += ok
            stats[d["doc_type"]][g][1] += 1
            if not ok:
                mismatches.append((d["doc_id"], k, ev, tv))

    grand_ok = grand_n = 0
    for t in ("CAS", "SOI", "LETTER"):
        if t not in stats:
            continue
        ok_t = sum(v[0] for v in stats[t].values())
        n_t = sum(v[1] for v in stats[t].values())
        grand_ok, grand_n = grand_ok + ok_t, grand_n + n_t
        print(f"\n{bold}{t}: {ok_t}/{n_t} fields ({100 * ok_t / n_t:.1f}%){reset}")
        for g, (ok, n) in stats[t].items():
            print(f"  {g:<34} {'' if ok == n else red}{ok}/{n} ({100 * ok / n:.0f}%){reset}")
    print(f"\n{bold}All extracted fields: {grand_ok}/{grand_n} ({100 * grand_ok / grand_n:.1f}%){reset}")

    print(f"\n{bold}Mismatches ({len(mismatches)}){reset}")
    for doc_id, k, ev, tv in mismatches:
        print(f"  {red}✗{reset} {doc_id}  {k:<58} extracted={fmt(ev)}  truth={fmt(tv)}")
    errs = [r for r in recs if r.get("error")]
    for r in errs:
        print(f"  {red}ERROR{reset} {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
