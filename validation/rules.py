"""Finance consistency rules. Each rule takes extracted fields and returns an Issue or None.

Tolerances match printed precision:
- Money: amounts printed as "$1.23mm" or "(in thousands)" are rounded, so sums may be off by a few
  rounding units. Tolerance = max(ABS_TOL, REL_TOL x reference amount), inclusive. 0.5% covers the
  worst case of ~7 components each rounded to the nearest $10k on an eight-figure balance, while
  a genuine break of 1%+ still fails.
- Percentages: printed to 0-2 dp, so sums and differences get +/-PCT_TOL percentage points, inclusive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

REL_TOL = 0.005
ABS_TOL = 1.0
PCT_TOL = 0.5
EPS = 1e-9  # float noise only; boundaries are inclusive


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


def money_tol(*refs: float) -> float:
    return max(ABS_TOL, REL_TOL * max(abs(r) for r in refs))


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
        expected, f["ending_nav"], money_tol(f["beginning_nav"], f["ending_nav"]))


def cas_unfunded_le_commitment(f: dict) -> Issue | None:
    tol = money_tol(f["total_commitment"])
    excess = f["unfunded_commitment"] - f["total_commitment"]
    if excess <= tol + EPS:
        return None
    return Issue("cas_unfunded_le_commitment", "unfunded_commitment exceeds total_commitment",
                 f"<= {f['total_commitment']:.4f}", f["unfunded_commitment"], round(excess, 4), round(tol, 4))


# --------------------------------------------------------------------------- SOI

def soi_fair_value_sum(f: dict) -> Issue | None:
    total = sum(h["fair_value"] for h in f["holdings"])
    return _check_equal("soi_fair_value_sum", "sum of holdings fair_value != total_fair_value",
                        total, f["total_fair_value"], money_tol(f["total_fair_value"], total))


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
