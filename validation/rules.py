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

from dataclasses import asdict, dataclass

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


RULES = {
    "CAS": [cas_rollforward, cas_unfunded_le_commitment],
    "SOI": [soi_fair_value_sum, soi_pct_sum],
    "LETTER": [letter_gross, letter_net],
}
