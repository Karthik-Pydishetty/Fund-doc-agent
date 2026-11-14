"""Offline scoring of pipeline records against ground truth. The only place metrics are defined.

Field accuracy: list fields are matched by key (holding name, sector), not position, so
`holdings[<name>].cost` scores as field `SOI.holdings.cost`; a missing list item fails all its
fields (and `holdings[present]`). Numbers: money within MONEY_TOL, percentages within PCT_TOL.
Dates are ISO strings; strings compare case-insensitively after trimming. Scoring is end to end:
a misrouted doc fails all its fields. commentary_summary and confidence are not scored.

Exceptions: precision/recall of EXCEPTION vs ground truth expected_exception over CAS/SOI/LETTER
docs; UNKNOWN docs are reported separately (should all be flagged unknown_doc_type).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from validation.validate import EXCEPTION, validate_records

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"
TYPES = ["CAS", "SOI", "LETTER", "UNKNOWN"]
LIST_KEYS = {"holdings": "name", "top_holdings": "name", "top_sector_exposures": "sector"}
UNSCORED = {"confidence", "commentary_summary"}
MONEY_TOL, PCT_TOL = 0.5, 0.05
PCT_HINTS = ("pct", "exposure", "return")


def load_truth() -> list[dict]:
    return json.loads(GROUND_TRUTH.read_text())["documents"]


def load_log(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Return (header, doc records, eval records) from a pipeline log."""
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    header, rest = lines[0], lines[1:]
    return header, [r for r in rest if "file" in r], [r for r in rest if r.get("kind") == "eval"]


# --------------------------------------------------------------------------- field matching

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


def score_doc(truth_doc: dict, rec: dict | None):
    """Yield (flat_key, group, extracted, truth, ok) for every truth field, then extras with truth=None."""
    t_flat = flatten(truth_doc["fields"])
    routed = rec and rec.get("route") and rec["route"]["doc_type"] == truth_doc["doc_type"]
    e_flat = ({k.casefold(): v for k, v in flatten(rec["extracted"]).items()}
              if routed and rec.get("extracted") else {})
    for k, tv in t_flat.items():
        ev = e_flat.get(k.casefold())
        g = field_group(k)
        yield k, g, ev, tv, match(ev, tv, g)
    t_keys = {k.casefold() for k in t_flat}
    for k, ev in e_flat.items():
        if k not in t_keys and k.endswith("]"):
            yield k, "extra item", ev, None, False


# --------------------------------------------------------------------------- metrics

def exception_scores(results, truth_docs) -> dict:
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


def compute_metrics(records: list[dict], truth: list[dict] | None = None) -> dict:
    """All eval metrics for one run. Re-validates records, so older logs score the same way."""
    truth = truth if truth is not None else load_truth()
    by_file = {r["file"]: r for r in records}
    docs = [d for d in truth if d["file"] in by_file]

    confusion = Counter()
    for d in docs:
        r = by_file[d["file"]]
        confusion[(d["doc_type"], r["route"]["doc_type"] if r.get("route") else "ERROR")] += 1
    router_ok = sum(v for (t, p), v in confusion.items() if t == p)

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mismatches = []
    for d in docs:
        if d["doc_type"] == "UNKNOWN":
            continue
        for k, g, ev, tv, ok in score_doc(d, by_file[d["file"]]):
            if g == "extra item":
                mismatches.append({"doc_id": d["doc_id"], "doc_type": d["doc_type"], "group": g, "field": k,
                                   "extracted": ev, "truth": None})
                continue
            key = f"{d['doc_type']}.{g}"
            counts[key][0] += ok
            counts[key][1] += 1
            type_counts[d["doc_type"]][0] += ok
            type_counts[d["doc_type"]][1] += 1
            if not ok:
                mismatches.append({"doc_id": d["doc_id"], "doc_type": d["doc_type"], "group": key, "field": k,
                                   "extracted": ev, "truth": tv})

    exc = exception_scores(validate_records(records), docs)
    all_ok = sum(v[0] for v in counts.values())
    all_n = sum(v[1] for v in counts.values())
    return {
        "n_docs": len(docs),
        "router_accuracy": router_ok / len(docs) if docs else 0.0,
        "router_confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items())},
        "field_accuracy_overall": all_ok / all_n if all_n else 0.0,
        "doc_type_accuracy": {t: ok / n for t, (ok, n) in type_counts.items()},
        "field_accuracy": {k: ok / n for k, (ok, n) in counts.items()},
        "field_counts": {k: list(v) for k, v in counts.items()},
        "exceptions": {k: exc[k] for k in ("precision", "recall", "tp", "fp", "fn")},
        "unknown_flagged": f"{len(exc['unknown_flagged'])}/{len(exc['unknown'])}",
        "mismatches": mismatches,
    }
