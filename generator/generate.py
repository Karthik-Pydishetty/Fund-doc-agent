"""Generate synthetic fund PDFs and data/ground_truth.json.

    python -m generator.generate [--seed 42] [--out data]

Deterministic: the same seed produces byte-identical PDFs and ground truth.
File names are neutral (doc_001.pdf ...) and shuffled so the doc type can't be
read off the file name or position.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import cas, letter, other, soi
from .common import FUNDS, INVESTORS

SEED = 42
GENERATOR_VERSION = "1.0"
N_PER_TYPE = 10

# Intentionally broken docs: index within its type -> defect.
BROKEN = {("CAS", 2): "rollforward", ("CAS", 7): "unfunded",
          ("SOI", 4): "total_mismatch", ("LETTER", 5): "gross_mismatch"}

CAS_STYLES = {"table": ["full", "plain", "decimals", "k", "mm"],
              "multi_period": ["mm", "full", "k", "plain", "decimals"],
              "prose": ["full", "words"]}


def _plan() -> list[tuple[str, int, str]]:
    plan = []
    for doc_type, layouts in (("CAS", cas.LAYOUTS), ("SOI", soi.LAYOUTS), ("LETTER", letter.LAYOUTS)):
        plan += [(doc_type, i, layouts[i % len(layouts)]) for i in range(N_PER_TYPE)]
    plan += [("UNKNOWN", i, lay) for i, lay in enumerate(other.LAYOUTS)]
    return plan


def generate(out_dir: str | Path = "data", seed: int = SEED) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("doc_*.pdf"):
        old.unlink()

    plan = _plan()
    random.Random(seed).shuffle(plan)
    private = [f for f in FUNDS if f.strategy == "private"]
    long_short = [f for f in FUNDS if f.strategy == "long_short"]
    layout_count: dict[tuple[str, str], int] = {}

    docs = []
    for n, (doc_type, idx, layout) in enumerate(plan, start=1):
        rng = random.Random(f"{seed}:{doc_type}:{idx}")  # per-doc stream, independent of shuffle
        doc_id = f"doc_{n:03d}"
        path = out / f"{doc_id}.pdf"
        broken = BROKEN.get((doc_type, idx))
        k = layout_count.get((doc_type, layout), 0)
        layout_count[(doc_type, layout)] = k + 1
        number_format = None

        if doc_type == "CAS":
            number_format = CAS_STYLES[layout][k % len(CAS_STYLES[layout])]
            s = cas.sample(rng, rng.choice(private), rng.choice(INVESTORS), number_format, broken)
            cas.render(str(path), s, layout, number_format, rng)
        elif doc_type == "SOI":
            number_format = {"flat": "full", "grouped": "plain", "thousands": "thousands"}[layout]
            s = soi.sample(rng, rng.choice(private), layout, broken)
            soi.render(str(path), s, layout, rng)
        elif doc_type == "LETTER":
            number_format = "percent"
            s = letter.sample(rng, rng.choice(long_short), broken)
            letter.render(str(path), s, layout, rng)
        else:
            other.render(str(path), rng.choice(FUNDS), rng.choice(INVESTORS), layout, rng)
            s = {"fields": {}, "reasons": []}

        docs.append({
            "doc_id": doc_id,
            "file": path.name,
            "doc_type": doc_type,
            "layout": layout,
            "number_format": number_format,
            "expected_exception": bool(s["reasons"]),
            "exception_reasons": s["reasons"],
            "fields": s["fields"],
        })

    truth = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "conventions": {
            "dates": "ISO 8601 (YYYY-MM-DD)",
            "money": "absolute units in the document currency (thousands/mm/million expanded)",
            "cas_outflows": "contributions, distributions, management_fees, incentive_fees are positive "
                            "magnitudes; net_gain_loss is signed",
            "percentages": "percent units, e.g. 142.5 means 142.5%",
            "quarter": "Q<n> <YYYY>",
        },
        "documents": docs,
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    truth = generate(args.out, args.seed)
    docs = truth["documents"]
    counts = {t: sum(d["doc_type"] == t for d in docs) for t in ("CAS", "SOI", "LETTER", "UNKNOWN")}
    print(f"Wrote {len(docs)} PDFs to {args.out}/ {counts}; "
          f"{sum(d['expected_exception'] for d in docs)} flagged expected_exception")


if __name__ == "__main__":
    main()
