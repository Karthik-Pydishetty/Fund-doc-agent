"""Seeded fault injection, to measure the self-correction loop when the model itself makes no misreads.

After the first-pass extraction, pick N documents the validator accepted ("clean"; no ground truth
is used) and corrupt one field in each. Docs cycle through three categories (covered, uncovered_text,
covered, uncovered_numeric, ...) so every run has a mix:
  covered            a numeric field whose corruption makes a finance rule fail, so the loop retries
                     dropped_digit 17,176,709 -> 1,716,709 | sign_flip 1,183,581 -> -1,183,581 |
                     scale_10x 120.9 -> 1209.0 or 12.09
  uncovered_text     a field no finance rule checks: fund/manager/investor name (adjacent letters
                     swapped, or trailing entity suffix dropped), a date (year off by one), the quarter
  uncovered_numeric  a numeric corruption no rule notices, e.g. a holding's cost, a top-holding
                     weight, or a sign flip that still satisfies unfunded <= commitment
Uncovered faults pass validation, are never retried, and are the loop's blind spot.
Every injected fault is recorded on its record ("fault_injection") and summarised in the log header.
"""
from __future__ import annotations

import copy
import datetime as dt
import random
import re

from agents.self_correct import flatten, set_path
from validation.rules import ARITHMETIC_RULES
from validation.validate import SCHEMAS, validate_record

NUMERIC_KINDS = ("dropped_digit", "sign_flip", "scale_10x")
NAME_FIELDS = ("fund_name", "manager", "investor_name")
DATE_FIELDS = ("period_start", "period_end", "as_of_date")
CATEGORY_CYCLE = ("covered", "uncovered_text", "covered", "uncovered_numeric")


def _fmt(v: float) -> str:
    return f"{v:.10f}".rstrip("0").rstrip(".").lstrip("-")


def corrupt_number(value: float, kind: str, rng: random.Random) -> float | None:
    if value == 0:
        return None
    if kind == "sign_flip":
        return -value
    if kind == "scale_10x":
        return value * 10 if rng.random() < 0.5 else value / 10
    s = _fmt(value)
    digits = [i for i, ch in enumerate(s) if ch.isdigit()]
    if len(digits) < 2:
        return None
    i = rng.choice(digits)
    out = s[:i] + s[i + 1:]
    if not out or out.startswith(".") or out.endswith("."):
        return None
    new = float(out) * (-1 if value < 0 else 1)
    return None if new == value else new


def corrupt_name(name: str, rng: random.Random) -> str | None:
    if "," in name and rng.random() < 0.5:
        return name.rsplit(",", 1)[0]  # "Granite Peak Capital Partners III, L.P." -> "... III"
    words = [i for i, w in enumerate(name.split(" ")) if len(w) >= 4 and w.isalpha()]
    if not words:
        return None
    parts = name.split(" ")
    w = parts[rng.choice(words)]
    j = rng.randrange(len(w) - 1)
    swapped = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    if swapped == w:
        return None
    parts[parts.index(w)] = swapped
    return " ".join(parts)


def corrupt_date(iso: str) -> str | None:
    d = dt.date.fromisoformat(iso)
    try:
        return d.replace(year=d.year - 1).isoformat()
    except ValueError:  # Feb 29
        return None


def corrupt_quarter(q: str) -> str | None:
    m = re.fullmatch(r"Q([1-4]) (\d{4})", q)
    if not m:
        return None
    n, y = int(m[1]), int(m[2])
    return f"Q{n - 1} {y}" if n > 1 else f"Q4 {y - 1}"


def breaks_rule(doc_type: str, fields: dict) -> list[str]:
    f = SCHEMAS[doc_type].model_validate(fields).model_dump()
    return [i.rule for rule in ARITHMETIC_RULES[doc_type] if (i := rule(f))]


def candidates(rec: dict, rng: random.Random) -> list[dict]:
    """Every single-field corruption of this record considered, with the finance rules each one breaks."""
    doc_type, fields = rec["route"]["doc_type"], rec["extracted"]
    out = []

    def add(path, kind, old, new):
        if new is None or new == old:
            return
        trial = copy.deepcopy(fields)
        set_path(trial, path, new)
        rules = breaks_rule(doc_type, trial)
        category = "covered" if rules else ("uncovered_numeric" if isinstance(old, (int, float)) else "uncovered_text")
        out.append({"field": path, "kind": kind, "original": old, "corrupted": new, "rules_broken": rules,
                    "category": category})

    for path, value in flatten(fields).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            for kind in NUMERIC_KINDS:
                add(path, kind, value, corrupt_number(float(value), kind, rng))
        elif path in NAME_FIELDS:
            add(path, "name_typo", value, corrupt_name(value, rng))
        elif path in DATE_FIELDS:
            add(path, "date_year_shift", value, corrupt_date(value))
        elif path == "quarter":
            add(path, "quarter_shift", value, corrupt_quarter(value))
    return out


def inject_faults(records: list[dict], n: int, seed: int) -> dict:
    """Corrupt one field in n seeded-random clean records (in place), alternating covered/uncovered."""
    clean = sorted((r for r in records if (r.get("validation") or {}).get("status") == "ACCEPTED"
                    and (r.get("route") or {}).get("doc_type") in SCHEMAS and r.get("extracted")),
                   key=lambda r: r["file"])
    chosen = sorted(random.Random(f"faults:{seed}").sample(clean, min(n, len(clean))), key=lambda r: r["file"])
    summary = {"requested": n, "seed": seed, "eligible_docs": len(clean), "injected": 0,
               "by_category": {c: 0 for c in CATEGORY_CYCLE}}
    for i, rec in enumerate(chosen):
        rng = random.Random(f"faults:{seed}:{rec['file']}")
        cands = candidates(rec, rng)
        want = CATEGORY_CYCLE[i % len(CATEGORY_CYCLE)]
        pool = [c for c in cands if c["category"] == want] or cands
        if not pool:
            continue
        fault = rng.choice(pool)
        rec["fault_injection"] = {"injected": fault, "candidates": len(cands),
                                  "covered_candidates": sum(c["category"] == "covered" for c in cands)}
        rec["first_pass_extracted"] = copy.deepcopy(rec["extracted"])
        set_path(rec["extracted"], fault["field"], fault["corrupted"])
        v = validate_record(rec)
        rec["validation"] = {"status": v.status, "issues": [x.to_dict() for x in v.issues]}
        summary["injected"] += 1
        summary["by_category"][fault["category"]] += 1
    return summary
