"""Finance consistency rules. Each rule takes extracted fields and returns an Issue or None.

Tolerances match how precisely each document prints its numbers. Validation can't see how the PDF
was formatted (and must not read ground truth), so it infers the printed rounding step from the
extracted money values themselves: the largest step in STEPS that divides every non-zero amount.
  "$1,234,567"                -> step 1        "$1,235k" / "(in thousands)" -> step 1,000
  "$12.35mm" / "€12.35 million" -> step 10,000  amounts with cents          -> step 0.01
Each printed term can be off by up to half a step, so a sum of N printed terms gets
tolerance = step x N (2x headroom). A full-unit CAS roll-forward (7 terms) therefore gets 7 units.
Steps are capped at MAX_STEP so a document whose amounts happen to be round can't widen its own
tolerance beyond what "mm to 2 dp" printing needs; a doc printed more coarsely than that would be
sent to review, which is the safe direction.

unfunded_commitment <= total_commitment is strict: STRICT_TOL ($1), whatever the print precision.
Percentages are printed to 0-2 dp: sums and differences get +/-PCT_TOL percentage points.
All boundaries are inclusive.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

STEPS = (10_000, 1_000, 100, 10, 1)
MAX_STEP = STEPS[0]
CENT_STEP = 0.01
STRICT_TOL = 1.0
PCT_TOL = 0.5
EPS = 1e-9  # float noise only

CAS_MONEY = ("beginning_nav", "contributions", "distributions", "management_fees", "incentive_fees",
             "net_gain_loss", "ending_nav", "total_commitment", "unfunded_commitment")
ROLLFORWARD_TERMS = 7  # beginning, contributions, distributions, two fees, gain/loss, ending


@dataclass
class Issue:
    rule: str
    reason: str
    expected: float | str | None = None
    actual: float | str | None = None
    difference: float | None = None
    tolerance: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def printed_step(values) -> float:
    """Rounding step implied by a document's amounts (see module docstring)."""
    vals = [abs(float(v)) for v in values if v]
    if not vals:
        return 1.0
    if any(abs(v - round(v)) > 1e-6 for v in vals):
        return CENT_STEP
    ints = [round(v) for v in vals]
    return float(next(s for s in STEPS if all(i % s == 0 for i in ints)))


def cas_step(f: dict) -> float:
    return printed_step(f[k] for k in CAS_MONEY)


def soi_step(f: dict) -> float:
    return printed_step([f["total_fair_value"]] + [h[k] for h in f["holdings"] for k in ("cost", "fair_value")])


def _check_equal(rule: str, reason: str, expected: float, actual: float, tol: float) -> Issue | None:
    diff = actual - expected
    if abs(diff) <= tol + EPS:
        return None
    return Issue(rule, reason, round(expected, 4), round(actual, 4), round(diff, 4), round(tol, 4))


# --------------------------------------------------------------------------- CAS

def cas_rollforward(f: dict) -> Issue | None:
    expected = (f["beginning_nav"] + f["contributions"] - f["distributions"] - f["management_fees"]
                - f["incentive_fees"] + f["net_gain_loss"])
    return _check_equal(
        "cas_rollforward",
        "ending_nav != beginning_nav + contributions - distributions - management_fees - incentive_fees "
        "+ net_gain_loss",
        expected, f["ending_nav"], cas_step(f) * ROLLFORWARD_TERMS)


def cas_unfunded_le_commitment(f: dict) -> Issue | None:
    excess = f["unfunded_commitment"] - f["total_commitment"]
    if excess <= STRICT_TOL + EPS:
        return None
    return Issue("cas_unfunded_le_commitment", "unfunded_commitment exceeds total_commitment",
                 f"<= {f['total_commitment']:.4f}", f["unfunded_commitment"], round(excess, 4), STRICT_TOL)


# --------------------------------------------------------------------------- SOI

def soi_fair_value_sum(f: dict) -> Issue | None:
    n_terms = len(f["holdings"]) + 1  # each holding plus the printed total
    return _check_equal("soi_fair_value_sum", "sum of holdings fair_value != total_fair_value",
                        sum(h["fair_value"] for h in f["holdings"]), f["total_fair_value"],
                        soi_step(f) * n_terms)


def soi_pct_sum(f: dict) -> Issue | None:
    return _check_equal("soi_pct_sum", "sum of holdings pct_of_nav is not ~100",
                        100.0, sum(h["pct_of_nav"] for h in f["holdings"]), PCT_TOL)


# --------------------------------------------------------------------------- LETTER

def letter_gross(f: dict) -> Issue | None:
    return _check_equal("letter_gross", "gross_exposure != long_exposure + short_exposure",
                        f["long_exposure"] + f["short_exposure"], f["gross_exposure"], PCT_TOL)


def letter_net(f: dict) -> Issue | None:
    return _check_equal("letter_net", "net_exposure != long_exposure - short_exposure",
                        f["long_exposure"] - f["short_exposure"], f["net_exposure"], PCT_TOL)


# --------------------------------------------------------------------------- sanity checks (Phase 9b)
# Deterministic checks for fields the arithmetic rules can't see. They read only the extraction
# (and the fund registry, an ops master list, not ground truth).

DAYS_MIN, DAYS_MAX = 80, 100  # "about one quarter"
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "funds.json"
NEAR_MISS_CUTOFF = 0.6
NON_NEGATIVE = {
    "CAS": ("management_fees", "incentive_fees", "total_commitment", "unfunded_commitment"),
    "SOI": ("total_fair_value",),
}
SOI_HOLDING_NON_NEGATIVE = ("cost", "fair_value")
QUARTER_LABEL = re.compile(r"^Q([1-4]) (\d{4})$")


def _date(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def _quarter_of(d: dt.date) -> tuple[dt.date, dt.date]:
    q = (d.month - 1) // 3
    start = dt.date(d.year, 3 * q + 1, 1)
    nxt = dt.date(d.year + (q == 3), (3 * q + 3) % 12 + 1, 1)
    return start, nxt - dt.timedelta(days=1)


def cas_period_order(f: dict) -> Issue | None:
    start, end = _date(f["period_start"]), _date(f["period_end"])
    if end > start:
        return None
    return Issue("cas_period_order", "period_end is not after period_start",
                 f"> {start.isoformat()}", end.isoformat(), (end - start).days)


def cas_period_span(f: dict) -> Issue | None:
    start, end = _date(f["period_start"]), _date(f["period_end"])
    days = (end - start).days + 1
    if end <= start or DAYS_MIN <= days <= DAYS_MAX:
        return None  # order is cas_period_order's finding
    return Issue("cas_period_span", f"reporting period is {days} days, not about one quarter",
                 f"{DAYS_MIN}-{DAYS_MAX} days", days, None)


def cas_period_quarter(f: dict) -> Issue | None:
    """The stated period must be one calendar quarter (the label the statement implies)."""
    start, end = _date(f["period_start"]), _date(f["period_end"])
    if end <= start or not DAYS_MIN <= (end - start).days + 1 <= DAYS_MAX:
        return None  # already reported by order / span
    q_start, q_end = _quarter_of(start)
    if (start, end) == (q_start, q_end):
        return None
    return Issue("cas_period_quarter", "period does not match a calendar quarter",
                 f"{q_start.isoformat()} to {q_end.isoformat()}", f"{start.isoformat()} to {end.isoformat()}")


def letter_quarter_label(f: dict) -> Issue | None:
    if QUARTER_LABEL.match(str(f["quarter"]).strip()):
        return None
    return Issue("letter_quarter_label", "quarter is not a valid 'Q<1-4> <YYYY>' label", "Q<1-4> <YYYY>",
                 f["quarter"])


def _non_negative(rule: str, values: list[tuple[str, float]]) -> Issue | None:
    neg = [(k, v) for k, v in values if v < 0]
    if not neg:
        return None
    return Issue(rule, "negative value(s) where only zero or positive amounts are possible: "
                 + ", ".join(f"{k}={v:,.2f}" for k, v in neg), ">= 0", neg[0][1], neg[0][1])


def cas_non_negative(f: dict) -> Issue | None:
    return _non_negative("cas_non_negative", [(k, f[k]) for k in NON_NEGATIVE["CAS"]])


def soi_non_negative(f: dict) -> Issue | None:
    vals = [(k, f[k]) for k in NON_NEGATIVE["SOI"]]
    vals += [(f"holdings[{h['name']}].{k}", h[k]) for h in f["holdings"] for k in SOI_HOLDING_NON_NEGATIVE]
    return _non_negative("soi_non_negative", vals)


@lru_cache(maxsize=1)
def fund_registry(path: Path = REGISTRY_PATH) -> tuple[str, ...]:
    return tuple(x["fund_name"] for x in json.loads(Path(path).read_text())["funds"])


def fund_in_registry(f: dict) -> Issue | None:
    name = str(f["fund_name"])
    names = fund_registry()
    if name in names:
        return None
    close = difflib.get_close_matches(name, names, n=1, cutoff=NEAR_MISS_CUTOFF)
    reason = f"fund name not in registry, closest: {close[0]}" if close else "fund name not in registry"
    return Issue("fund_in_registry", reason, close[0] if close else "a registered fund name", name)


# Arithmetic identities (Phase 4). The fault injector chooses faults by these alone, so a seeded
# run injects the same faults however many sanity checks are added later.
ARITHMETIC_RULES = {
    "CAS": [cas_rollforward, cas_unfunded_le_commitment],
    "SOI": [soi_fair_value_sum, soi_pct_sum],
    "LETTER": [letter_gross, letter_net],
}
SANITY_RULES = {
    "CAS": [cas_period_order, cas_period_span, cas_period_quarter, cas_non_negative, fund_in_registry],
    "SOI": [soi_non_negative, fund_in_registry],
    "LETTER": [letter_quarter_label, fund_in_registry],
}
RULES = {t: ARITHMETIC_RULES[t] + SANITY_RULES[t] for t in ARITHMETIC_RULES}
