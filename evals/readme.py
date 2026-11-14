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


def render_blocks(log: Path) -> dict[str, str]:
    return {"results": block_results(log), "tolerances": block_tolerances(), "doc021": block_doc021(log)}


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
