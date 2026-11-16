"""Render the numeric blocks of README.md from evals/baseline.json, a run log and the code's constants.

    python -m evals.readme [--log PATH]    # default: newest runs/*_pipeline.jsonl

README numbers are never typed by hand. Each generated block sits between
`<!-- BEGIN:<name> -->` and `<!-- END:<name> -->` markers and names its sources;
tests/test_readme.py re-renders the blocks from those same sources and fails if README is stale.
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

from config import PRICING_PER_MTOK
from evals import regression, scoring
from validation import rules
from validation.validate import MIN_CONFIDENCE, validate_record

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BLOCK_RE = re.compile(r"(<!-- BEGIN:(?P<name>[\w-]+) -->\n?)(?P<body>.*?)(<!-- END:(?P=name) -->)", re.S)
SOURCE_RE = re.compile(r"Latest run log: `(?P<log>runs/[^`]+)`")


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _counts(m: dict, doc_type: str) -> tuple[int, int]:
    rows = [v for k, v in m["field_counts"].items() if k.startswith(f"{doc_type}.")]
    return sum(r[0] for r in rows), sum(r[1] for r in rows)


def run_cost(header: dict, records: list[dict]) -> dict:
    calls = [c for r in records for c in r.get("calls", {}).values()]
    tin, tout = sum(c["input_tokens"] for c in calls), sum(c["output_tokens"] for c in calls)
    price = PRICING_PER_MTOK.get(header["model"])
    usd = (tin * price[0] + tout * price[1]) / 1e6 if price else None
    per_doc = [sum(c["latency_s"] for c in r.get("calls", {}).values()) for r in records]
    return {"calls": len(calls), "tin": tin, "tout": tout, "usd": usd, "n": len(records),
            "doc_mean": statistics.mean(per_doc) if per_doc else 0.0,
            "doc_max": max(per_doc) if per_doc else 0.0,
            "api_s": sum(c["latency_s"] for c in calls), "wall_s": header.get("wall_s"),
            "workers": header.get("workers")}


# --------------------------------------------------------------------------- blocks

def block_results(log: Path) -> str:
    base = regression.load_baseline()
    bm = base["metrics"]
    header, records, _ = scoring.load_log(log)
    m = scoring.compute_metrics(records)
    truth = scoring.load_truth()
    types = Counter(d["doc_type"] for d in truth)
    types = {t: types[t] for t in scoring.TYPES if types[t]}
    broken = sum(d["expected_exception"] for d in truth)
    c = run_cost(header, records)

    def exc(e):
        return (f"{e['precision']:.2f} / {e['recall']:.2f} "
                f"({len(e['tp'])} caught, {len(e['fp'])} false alarms, {len(e['fn'])} missed)")

    rows = [
        ("Router accuracy", _pct(bm["router_accuracy"]), f"{_pct(m['router_accuracy'])} ({m['n_docs']} docs)"),
    ]
    for t in ("CAS", "SOI", "LETTER"):
        ok, n = _counts(m, t)
        rows.append((f"{t} field accuracy", _pct(bm["doc_type_accuracy"][t]), f"{_pct(ok / n)} ({ok}/{n} fields)"))
    rows += [
        ("All fields", _pct(bm["field_accuracy_overall"]), _pct(m["field_accuracy_overall"])),
        ("Exception precision / recall", exc(bm["exceptions"]), exc(m["exceptions"])),
        ("UNKNOWN docs sent to exceptions", "", m["unknown_flagged"]),
    ]
    cost = (f"${c['usd']:.2f} per run, ${c['usd'] / c['n']:.4f} per doc" if c["usd"] is not None
            else f"no price for {header['model']} in config.PRICING_PER_MTOK")
    wall = (f"{c['wall_s']:.1f} s wall clock with {c['workers']} workers" if c["wall_s"] is not None
            else "wall clock not recorded in this log")
    out = [
        f"Dataset: {len(truth)} synthetic PDFs ({', '.join(f'{n} {t}' for t, n in types.items())}), "
        f"{broken} intentionally broken. Model `{header['model']}`, prompts "
        + ", ".join(f"`{v}`" for v in header["prompt_versions"].values()) + ".",
        "",
        "| Metric | Baseline (`evals/baseline.json`) | Latest run |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c_} |" for a, b, c_ in rows],
        f"| Tokens | | {c['tin']:,} in / {c['tout']:,} out over {c['calls']} API calls |",
        f"| Cost | | {cost} |",
        f"| Latency | | {c['doc_mean']:.1f} s mean per doc (router + extractor), {c['doc_max']:.1f} s max; {wall} |",
        "",
        f"Latest run log: `{_rel(log)}`. Baseline from `{base['source_log']}` at commit "
        f"`{(base['git']['commit'] or '?')[:8]}`. Cost uses list prices in `config.PRICING_PER_MTOK`.",
    ]
    return "\n".join(out) + "\n"


def block_tolerances() -> str:
    steps = ", ".join(f"{s:,}" for s in sorted(rules.STEPS))
    return "\n".join([
        f"- Money sums (CAS roll-forward, SOI fair-value total): tolerance = printed rounding step x number "
        f"of printed terms. The step is inferred per document from its amounts (one of {steps}, or "
        f"{rules.CENT_STEP} for cents; capped at {rules.MAX_STEP:,}). A full-unit CAS roll-forward has "
        f"{rules.ROLLFORWARD_TERMS} terms, so it gets {rules.ROLLFORWARD_TERMS} units.",
        f"- `unfunded_commitment <= total_commitment` is strict: {rules.STRICT_TOL:g} unit.",
        f"- Percentage sums and exposure identities: +/-{rules.PCT_TOL:g} percentage points.",
        f"- Extractor or router confidence below {MIN_CONFIDENCE:g} sends a doc to exceptions.",
        f"- Scoring tolerances: money within {scoring.MONEY_TOL:g}, percentages within {scoring.PCT_TOL:g}.",
        f"- Regression gate: fails if any field drops more than {regression.FIELD_DROP_POINTS:g} points, "
        f"or router accuracy or exception recall drops at all.",
    ]) + "\n"


def block_doc021(log: Path) -> str:
    _, records, _ = scoring.load_log(log)
    r = next(x for x in records if x["file"] == "doc_021.pdf")
    v = validate_record(r)
    issue = next(i for i in v.issues if i.rule == "cas_rollforward")
    return (f"In `{_rel(log)}`, doc_021's printed ending NAV is {issue.actual:,.0f} while its roll-forward gives "
            f"{issue.expected:,.0f} (off by {issue.difference:,.0f}; tolerance {issue.tolerance:,.0f}). "
            f"The extractor reported confidence {r['extracted']['confidence']}, above the "
            f"{MIN_CONFIDENCE:g} threshold, so confidence alone would have accepted it; only the "
            f"`{issue.rule}` rule caught it.")  # inline block: no newline


def block_confidence(log: Path) -> str:
    """Confidence is miscalibrated both ways: high on a broken doc (main split), low on a correct one (hard split)."""
    high = block_doc021(log)
    hard_log = ROOT / regression.load_baseline(regression.BASELINES["hard"])["source_log"]
    _, records, _ = scoring.load_log(hard_log)
    r = next(x for x in records if x["file"] == "hard_010.pdf")
    truth = next(d for d in scoring.load_truth("hard") if d["file"] == "hard_010.pdf")
    wrong = sum(1 for k, g, *_rest, ok in scoring.score_doc(truth, r) if not ok)
    low = (f" The opposite also happens: in `{_rel(hard_log)}`, hard_010 (long exposure given only as a share of "
           f"gross, so long, short and net must be computed) was extracted with {wrong} wrong fields, yet the "
           f"extractor reported confidence {r['extracted']['confidence']}, below the {MIN_CONFIDENCE:g} threshold, "
           f"so a correct document went to exceptions. The threshold is unchanged for now.")
    return high + low  # inline block: no newline


def _report():
    from evals import report  # lazy: evals.report imports this module
    return report


def block_hard() -> str:
    """Hard split composition and Sonnet vs. Haiku on it (baseline run + newest current run of each other model)."""
    rep = _report()
    runs = rep.load_runs()
    base = regression.load_baseline(regression.BASELINES["hard"])
    current = [r for r in runs if r["split"] == "hard" and not r["fault"] and not r["superseded_by"]]
    sonnet = next(r for r in runs if r["name"] == base["source_log"])
    others = {}
    for r in current:
        if r["model"] != sonnet["model"]:
            others[r["model"]] = r
    cols = [sonnet, *[others[m] for m in sorted(others)]]
    truth = scoring.load_truth("hard")
    types = Counter(d["doc_type"] for d in truth)
    types = {t: types[t] for t in scoring.TYPES if types[t]}
    broken = sum(d["expected_exception"] for d in truth)

    def type_cell(r, t):
        ok, n = _counts(r["metrics"], t)
        return f"{_pct(ok / n)} ({ok}/{n})"

    def cost_cell(r):
        return f"${r['cost']['usd']:.4f} per run, ${r['cost']['usd'] / r['cost']['n']:.5f} per doc"

    rows = [("Router accuracy", lambda r: _pct(r["metrics"]["router_accuracy"])),
            *[(f"{t} fields", lambda r, t=t: type_cell(r, t)) for t in ("CAS", "SOI", "LETTER")],
            ("Exception precision / recall", lambda r: f"{r['metrics']['exceptions']['precision']:.2f} / "
                                                      f"{r['metrics']['exceptions']['recall']:.2f}"),
            ("Tokens in / out", lambda r: f"{r['cost']['tin']:,} / {r['cost']['tout']:,}"),
            ("Cost (list price)", cost_cell),
            ("Latency per doc, mean / max", lambda r: f"{r['cost']['doc_mean']:.1f} s / {r['cost']['doc_max']:.1f} s")]
    fps = sorted({fp for r in cols for fp in r["metrics"]["exceptions"]["fp"]})
    fp_notes = []
    for fp in fps:
        confs = []
        for r in cols:
            rec = next(x for x in r["records"] if x["file"] == f"{fp}.pdf")
            confs.append(f"{r['model']} {rec['extracted']['confidence']}")
        fp_notes.append(f"{fp} ({'; '.join(confs)})")
    out = [f"Hard split: {len(truth)} documents ({', '.join(f'{n} {t}' for t, n in types.items())}), "
           f"{broken} intentionally broken, v1 prompts unchanged.", "",
           "| Measure | " + " | ".join(f"`{r['model']}`" for r in cols) + " |",
           "|---|" + "---|" * len(cols),
           *[f"| {name} | " + " | ".join(fn(r) for r in cols) + " |" for name, fn in rows], "",
           "Runs: " + ", ".join(f"`{r['name']}`" for r in cols) + f" (the first is `evals/baseline_hard.json`'s source). "
           + (f"The only exception errors are correct documents sent to review for low self-reported confidence: "
              f"{'; '.join(fp_notes)}." if fp_notes else "No false alarms.")]
    return "\n".join(out) + "\n"


def block_selfcorrect() -> str:
    """Self-correction with seeded fault injection, before vs. after the deterministic checks."""
    rep = _report()
    from evals.blind_spot import outcomes
    from evals.self_correction import analyse
    pairs = rep.fault_pairs(rep.load_runs())
    rows, still = [], []
    for split, (b, a) in pairs.items():
        for label, run in (("before checks", b), ("after checks", a)):
            o, m = outcomes(run["path"])
            sc = analyse(run["path"])
            n = len(o)
            c = {v: sum(x["verdict"] == v for x in o.values()) for v in ("recovered", "flagged", "undetected")}
            rows.append(f"| {split} | {label} | `{run['path'].name}` | {c['recovered']}/{n} | {c['flagged']}/{n} | "
                        f"{c['undetected']}/{n} | {len(sc['fudges'])} | {len(sc['accepted_on_broken'])} | "
                        f"{m['precision']:.2f} / {m['recall']:.2f} | "
                        f"${sc['retry_usd']:.4f} (+{100 * sc['retry_usd'] / sc['first_usd']:.0f}%) |")
            if run is a:
                still += [f"{k[0]} `{k[1]}` ({k[2]})" for k, v in o.items() if v["verdict"] == "undetected"]
    seed = pairs and next(iter(pairs.values()))[1]["header"].get("fault_summary", {}).get("seed")
    return ("| Split | Checks | Run | Recovered | Flagged | Undetected | Fudges rejected | "
            "Changes accepted on broken docs | Exception P / R (docs without a fault) | Extra cost of retries |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            f"Both runs of a split use fault seed {seed} and the same fault count, so the same faults are injected. "
            f"Still undetected after the checks: {'; '.join(still) or 'none'}.\n")


def block_webhook() -> str:
    logs = []
    for log in sorted((ROOT / "runs").glob("*_pipeline.jsonl")):
        header, records, _ = scoring.load_log(log)
        if scoring.split_of(header) == "webhook":
            logs.append((log, records))
    latest = {}
    for _log, records in logs:
        for rec in records:
            latest[rec["webhook"]["document_id"]] = rec
    if not latest:
        return "No documents have been sent to the webhook service yet.\n"
    status = Counter(r["webhook"]["status"] for r in latest.values())
    lines = [f"- `{r['webhook']['filename']}` -> {r['webhook']['status']}"
             + (f" ({', '.join(i['rule'] for i in r['validation']['issues'])})" if r["validation"]["issues"] else "")
             for r in latest.values()]
    return (f"Documents received so far: {len(latest)} ({', '.join(f'{n} {s}' for s, n in sorted(status.items()))}), "
            f"logged in " + ", ".join(f"`{_rel(lg)}`" for lg, _ in logs) + ".\n\n" + "\n".join(lines) + "\n")


def render_blocks(log: Path) -> dict[str, str]:
    return {"results": block_results(log), "tolerances": block_tolerances(), "confidence": block_confidence(log),
            "hard": block_hard(), "selfcorrect": block_selfcorrect(), "webhook": block_webhook()}


def apply_blocks(text: str, blocks: dict[str, str]) -> str:
    seen = set()

    def sub(mt):
        seen.add(mt["name"])
        return mt.group(1) + blocks[mt["name"]] + mt.group(4)

    out = BLOCK_RE.sub(sub, text)
    missing = set(blocks) - seen
    if missing:
        raise SystemExit(f"README.md is missing markers for: {sorted(missing)}")
    return out


def log_named_in_readme(text: str) -> Path | None:
    mt = SOURCE_RE.search(text)
    return ROOT / mt["log"] if mt else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="run log to report (default: newest runs/*_pipeline.jsonl)")
    args = ap.parse_args()
    log = Path(args.log).resolve() if args.log else regression.latest_run_log()
    README.write_text(apply_blocks(README.read_text(), render_blocks(log)))
    print(f"README.md blocks rendered from {_rel(log)} and {regression.BASELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
