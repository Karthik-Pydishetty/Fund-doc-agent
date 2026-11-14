"""Run the CAS extractor on every CAS doc and print extracted vs. ground truth, field by field.

    python -m evals.compare_cas [--workers 5] [--no-color]
    python -m evals.compare_cas --from-log [PATH]   # rebuild the table from a logged run, no API calls

Doc selection uses ground truth (the router doesn't exist yet). Every live run is logged to
runs/<timestamp>_compare_cas.jsonl: one header line, then one line per doc. --from-log with no
PATH uses the newest such file; it re-scores the logged outputs against the current ground truth
and writes no new log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from schemas.cas import CAS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LOG_GLOB = "*_compare_cas.jsonl"
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


def cas_truth() -> list[dict]:
    return [d for d in json.loads((DATA / "ground_truth.json").read_text())["documents"] if d["doc_type"] == "CAS"]


def latest_log() -> Path:
    logs = sorted(RUNS.glob(LOG_GLOB))
    if not logs:
        sys.exit(f"no {LOG_GLOB} in {RUNS.relative_to(ROOT)}/; run without --from-log first")
    return logs[-1]


def load_log(path: Path) -> tuple[dict, list[tuple[dict, dict | None, str | None]]]:
    header, *rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    truth = {d["doc_id"]: d for d in cas_truth()}
    missing = [r["doc_id"] for r in rows if r["doc_id"] not in truth]
    if missing:
        sys.exit(f"{path.name}: docs not CAS in current ground truth: {missing}")
    return header, [(truth[r["doc_id"]], r["extracted"], r["error"]) for r in rows]


def score(results) -> tuple[dict[str, list[int]], int, int, int]:
    per_field = {f: [0, 0] for f in FIELDS}
    for doc, out, _ in results:
        for f in FIELDS:
            per_field[f][0] += bool(out) and match(out.get(f), doc["fields"][f])
            per_field[f][1] += 1
    total_ok = sum(v[0] for v in per_field.values())
    total = sum(v[1] for v in per_field.values())
    perfect = sum(1 for d, o, _ in results if o and all(match(o.get(f), d["fields"][f]) for f in FIELDS))
    return per_field, total_ok, total, perfect


def render(results, model: str, prompt_version: str, color: bool) -> str:
    red, green, bold, dim, reset = (("\033[31m", "\033[32m", "\033[1m", "\033[2m", "\033[0m") if color
                                    else ("", "", "", "", ""))
    w1, w2 = 20, 46
    lines = []
    for doc, out, err in results:
        tag = f"{red}EXPECTED EXCEPTION: {'; '.join(doc['exception_reasons'])}{reset}" if doc["expected_exception"] else ""
        lines.append(f"\n{bold}{doc['doc_id']}  layout={doc['layout']}  format={doc['number_format']}{reset}  {tag}")
        if err:
            lines.append(f"  {red}EXTRACTION FAILED: {err}{reset}")
            continue
        lines.append(f"  {dim}{'field':<{w1}} {'extracted':<{w2}} {'ground truth':<{w2}}{reset}")
        for f in FIELDS:
            t, e = doc["fields"][f], out.get(f)
            ok = match(e, t)
            mark = f"{green}✓{reset}" if ok else f"{red}✗ MISMATCH{reset}"
            lines.append(f"  {f:<{w1}} {'' if ok else red}{fmt(e):<{w2}}{reset} {fmt(t):<{w2}} {mark}")
        lines.append(f"  {'confidence':<{w1}} {fmt(out.get('confidence'))}")

    per_field, total_ok, total, perfect = score(results)
    lines.append(f"\n{bold}Per-field accuracy (prompt {prompt_version}, model {model}){reset}")
    for f, (ok, n) in per_field.items():
        lines.append(f"  {f:<{w1}} {'' if ok == n else red}{ok}/{n}{reset}")
    lines.append(f"{bold}Overall: {total_ok}/{total} fields ({100 * total_ok / total:.1f}%), "
                 f"{perfect}/{len(results)} docs fully correct{reset}")
    return "\n".join(lines)


def run_live(workers: int):
    from agents.extractors import cas as cas_extractor  # imported here so --from-log needs no API client
    from agents.llm_client import ExtractionError, get_client

    client = get_client()

    def run(doc):
        try:
            out = cas_extractor.extract_cas(DATA / doc["file"], client).model_dump(mode="json")
            return doc, out, None, client.last_call
        except ExtractionError as e:
            return doc, None, str(e), client.last_call

    started = dt.datetime.now(dt.timezone.utc)
    with ThreadPoolExecutor(workers) as pool:
        full = sorted(pool.map(run, cas_truth()), key=lambda r: r[0]["doc_id"])
    results = [(d, o, e) for d, o, e, _ in full]
    per_field, total_ok, total, perfect = score(results)

    RUNS.mkdir(exist_ok=True)
    log = RUNS / f"{started:%Y%m%dT%H%M%SZ}_compare_cas.jsonl"
    with log.open("w") as fh:
        fh.write(json.dumps({"timestamp": started.isoformat(), "kind": "compare_cas", "provider": client.provider,
                             "model": client.model, "prompt_versions": {"cas": cas_extractor.PROMPT_VERSION},
                             "metrics": {"field_accuracy": total_ok / total, "docs_fully_correct": perfect,
                                         "per_field": {f: ok / n for f, (ok, n) in per_field.items()}}}) + "\n")
        for doc, out, err, call in full:
            fh.write(json.dumps({"doc_id": doc["doc_id"], "extracted": out, "error": err,
                                 "call": call.__dict__ if call else None}) + "\n")
    return results, client.model, cas_extractor.PROMPT_VERSION, f"Run logged to {log.relative_to(ROOT)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--from-log", nargs="?", const="latest", metavar="PATH",
                    help="rebuild the table from a logged run (default: newest in runs/) without calling the API")
    args = ap.parse_args()
    color = not args.no_color and sys.stdout.isatty()

    if args.from_log:
        path = latest_log() if args.from_log == "latest" else Path(args.from_log).resolve()
        header, results = load_log(path)
        model, prompt = header["model"], header["prompt_versions"]["cas"]
        footer = f"Rebuilt from {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} " \
                 f"(run at {header['timestamp']}); no API calls made"
    else:
        results, model, prompt, footer = run_live(args.workers)

    print(render(results, model, prompt, color))
    print(f"\n{footer}")


if __name__ == "__main__":
    main()
