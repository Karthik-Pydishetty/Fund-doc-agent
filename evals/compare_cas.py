"""Run the CAS extractor on every CAS doc and print extracted vs. ground truth, field by field.

    python -m evals.compare_cas [--workers 5] [--no-color]

Doc selection uses ground truth (the router doesn't exist yet). Every run is logged to
runs/<timestamp>_compare_cas.jsonl: one header line, then one line per doc.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.extractors import cas as cas_extractor
from agents.llm_client import ExtractionError, get_client
from schemas.cas import CAS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
FIELDS = [f for f in CAS.model_fields if f != "confidence"]
NUM_TOL = 0.5  # money is in whole units; anything off by more than rounding noise is wrong


def match(extracted, truth) -> bool:
    if extracted is None:
        return False
    if isinstance(truth, (int, float)):
        return abs(float(extracted) - truth) <= NUM_TOL
    return str(extracted).strip().casefold() == str(truth).strip().casefold()


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, int) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    red, green, bold, dim, reset = (("", "", "", "", "") if args.no_color or not sys.stdout.isatty()
                                    else ("\033[31m", "\033[32m", "\033[1m", "\033[2m", "\033[0m"))

    docs = [d for d in json.loads((DATA / "ground_truth.json").read_text())["documents"] if d["doc_type"] == "CAS"]
    client = get_client()

    def run(doc):
        try:
            out = cas_extractor.extract_cas(DATA / doc["file"], client).model_dump(mode="json")
            return doc, out, None, client.last_call
        except ExtractionError as e:
            return doc, None, str(e), client.last_call

    started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(args.workers) as pool:
        results = sorted(pool.map(run, docs), key=lambda r: r[0]["doc_id"])

    lines, per_field = [], {f: [0, 0] for f in FIELDS}
    w1, w2 = 20, 46
    for doc, out, err, _ in results:
        tag = f"{red}EXPECTED EXCEPTION: {'; '.join(doc['exception_reasons'])}{reset}" if doc["expected_exception"] else ""
        lines.append(f"\n{bold}{doc['doc_id']}  layout={doc['layout']}  format={doc['number_format']}{reset}  {tag}")
        if err:
            lines.append(f"  {red}EXTRACTION FAILED: {err}{reset}")
            for f in FIELDS:
                per_field[f][1] += 1
            continue
        lines.append(f"  {dim}{'field':<{w1}} {'extracted':<{w2}} {'ground truth':<{w2}}{reset}")
        for f in FIELDS:
            t, e = doc["fields"][f], out.get(f)
            ok = match(e, t)
            per_field[f][0] += ok
            per_field[f][1] += 1
            mark = f"{green}✓{reset}" if ok else f"{red}✗{reset}"
            color = "" if ok else red
            lines.append(f"  {f:<{w1}} {color}{fmt(e):<{w2}}{reset} {fmt(t):<{w2}} {mark}")
        lines.append(f"  {'confidence':<{w1}} {fmt(out.get('confidence'))}")

    total_ok = sum(v[0] for v in per_field.values())
    total = sum(v[1] for v in per_field.values())
    docs_perfect = sum(1 for d, o, e, _ in results if o and all(match(o.get(f), d["fields"][f]) for f in FIELDS))
    lines.append(f"\n{bold}Per-field accuracy (prompt {cas_extractor.PROMPT_VERSION}, model {client.model}){reset}")
    for f, (ok, n) in per_field.items():
        c = "" if ok == n else red
        lines.append(f"  {f:<{w1}} {c}{ok}/{n}{reset}")
    lines.append(f"{bold}Overall: {total_ok}/{total} fields ({100 * total_ok / total:.1f}%), "
                 f"{docs_perfect}/{len(results)} docs fully correct{reset}")
    print("\n".join(lines))

    RUNS.mkdir(exist_ok=True)
    log = RUNS / f"{started:%Y%m%dT%H%M%SZ}_compare_cas.jsonl"
    with log.open("w") as fh:
        fh.write(json.dumps({"timestamp": started.isoformat(), "kind": "compare_cas", "provider": client.provider,
                             "model": client.model, "prompt_versions": {"cas": cas_extractor.PROMPT_VERSION},
                             "metrics": {"field_accuracy": total_ok / total, "docs_fully_correct": docs_perfect,
                                         "per_field": {f: ok / n for f, (ok, n) in per_field.items()}}}) + "\n")
        for doc, out, err, call in results:
            fh.write(json.dumps({"doc_id": doc["doc_id"], "extracted": out, "error": err,
                                 "call": call.__dict__ if call else None}) + "\n")
    print(f"\nRun logged to {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
