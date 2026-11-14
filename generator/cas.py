"""Capital Account Statements (CAS).

Layouts:
  table        single roll-forward table + commitment summary table
  multi_period QTD and inception-to-date columns side by side (extract the QTD column)
  prose        investor letter; every figure is in running text

Ground-truth conventions: contributions, distributions and fees are positive
magnitudes even when printed in parentheses; net_gain_loss is signed.
"""
from __future__ import annotations

import random

from reportlab.platypus import Spacer

from .common import (
    Fund, P, QUANTUM, build_pdf, fmt_date, fmt_money, quantize, quarter_bounds, styles, table,
)

LAYOUTS = ["table", "multi_period", "prose"]

LABELS = {
    "beginning_nav": ["Beginning Balance", "Opening Capital", "Beginning Capital Account",
                      "Capital Balance, Beginning of Period", "Opening Net Asset Value"],
    "contributions": ["Contributions", "Capital Contributions", "Capital Called", "Paid-in Capital"],
    "distributions": ["Distributions", "Capital Distributions", "Distributions Paid",
                      "Return of Capital and Income Distributed"],
    "management_fees": ["Management Fees", "Management Fee", "Advisory Fees", "Investment Management Fee"],
    "incentive_fees": ["Incentive Fees", "Carried Interest Allocation", "Performance Allocation",
                       "Incentive Allocation"],
    "net_gain_loss": ["Net Gain (Loss)", "Net Realized and Unrealized Gain (Loss)",
                      "Net Change in Unrealized Value", "Investment Income / (Loss)"],
    "ending_nav": ["Ending Balance", "Closing Capital", "Ending Capital Account",
                   "Capital Balance, End of Period", "Closing Net Asset Value"],
    "total_commitment": ["Total Commitment", "Capital Commitment", "Committed Capital"],
    "unfunded_commitment": ["Unfunded Commitment", "Remaining Commitment", "Uncalled Capital"],
    "paid_in": ["Contributions to Date", "Cumulative Paid-in Capital", "Capital Called to Date"],
}


def sample(rng: random.Random, fund: Fund, investor: str, style: str, broken: str | None) -> dict:
    q = QUANTUM[style]
    year, qtr = rng.choice([(2024, 4), (2025, 1), (2025, 2), (2025, 3), (2024, 3), (2024, 2)])
    start, end = quarter_bounds(year, qtr)

    commit = rng.randrange(20, 201) * 250_000
    unfunded = quantize(commit * rng.uniform(0.1, 0.65), q)
    paid_in = commit - unfunded
    beginning = quantize(paid_in * rng.uniform(0.8, 1.5), q)
    contributions = quantize(commit * rng.uniform(0.02, 0.1), q) if rng.random() < 0.6 else 0
    distributions = quantize(beginning * rng.uniform(0.02, 0.12), q) if rng.random() < 0.5 else 0
    mgmt = max(q, quantize(commit * rng.uniform(0.00375, 0.005), q))
    gain = quantize(beginning * rng.uniform(-0.08, 0.12), q)
    incentive = quantize(gain * rng.uniform(0.1, 0.2), q) if gain > 0 and rng.random() < 0.6 else 0
    ending = beginning + contributions - distributions - mgmt - incentive + gain

    reasons = []
    if broken == "rollforward":
        delta = max(q, quantize(beginning * rng.uniform(0.01, 0.04), q))
        ending += delta if rng.random() < 0.5 else -delta
        reasons.append("CAS ending_nav does not equal the roll-forward of beginning_nav")
    elif broken == "unfunded":
        unfunded = commit + quantize(commit * rng.uniform(0.05, 0.2), q)
        reasons.append("CAS unfunded_commitment exceeds total_commitment")

    fields = {
        "fund_name": fund.name, "manager": fund.manager, "investor_name": investor,
        "period_start": start.isoformat(), "period_end": end.isoformat(), "currency": fund.currency,
        "beginning_nav": beginning, "contributions": contributions, "distributions": distributions,
        "management_fees": mgmt, "incentive_fees": incentive, "net_gain_loss": gain,
        "ending_nav": ending, "total_commitment": commit, "unfunded_commitment": unfunded,
    }
    return {"fields": fields, "reasons": reasons, "_paid_in": paid_in, "_dates": (start, end)}


def _labels(rng: random.Random) -> dict[str, str]:
    return {k: rng.choice(v) for k, v in LABELS.items()}


def _amt(v: int, style: str, ccy: str, zero_dash: bool) -> str:
    return "—" if (v == 0 and zero_dash) else fmt_money(v, style, ccy)


def render(path: str, s: dict, layout: str, style: str, rng: random.Random) -> None:
    f, ccy = s["fields"], s["fields"]["currency"]
    start, end = s["_dates"]
    date_style = rng.choice(["long", "us", "dmy_abbr", "dmy_long"])
    font = rng.choice(["Helvetica", "Times-Roman"])
    st = styles(font)
    lab = _labels(rng)
    zero_dash = rng.random() < 0.5
    outflow_paren = rng.random() < 0.6  # print outflows as (x) vs "Less: x"
    a = lambda v: _amt(v, style, ccy, zero_dash)  # noqa: E731

    def outflow(key: str) -> tuple[str, str]:
        v = f[key]
        if outflow_paren:
            return lab[key], ("—" if v == 0 and zero_dash else fmt_money(-v, style, ccy))
        return f"Less: {lab[key]}", a(v)

    period = f"{fmt_date(start, date_style)} to {fmt_date(end, date_style)}"
    story = []

    if layout in ("table", "multi_period"):
        story += [P(f["fund_name"], st["title"]),
                  P(rng.choice(["Capital Account Statement", "Statement of Partner's Capital",
                                "Investor Capital Statement"]), st["subtitle"])]
        mgr_label = rng.choice(["General Partner", "Investment Manager", "Manager"])
        unit_note = ({"plain": f"Amounts in {ccy}", "k": f"{ccy}; k = thousands",
                      "mm": f"{ccy}; mm = millions"}.get(style, ccy))
        story.append(table([
            ["Investor:", f["investor_name"]],
            [f"{mgr_label}:", f["manager"]],
            ["Reporting Period:", period],
            ["Currency:", unit_note],
        ], [1.6, 4.8], font, header=False, right_cols_from=9))
        story.append(Spacer(1, 14))

        flow_rows = [
            (lab["beginning_nav"], a(f["beginning_nav"])),
            (lab["contributions"], a(f["contributions"])),
            outflow("distributions"),
            outflow("management_fees"),
            outflow("incentive_fees"),
            (lab["net_gain_loss"], a(f["net_gain_loss"])),
            (lab["ending_nav"], a(f["ending_nav"])),
        ]
        if layout == "table":
            rows = [["Capital Account Activity", "Amount"]] + [list(r) for r in flow_rows]
            story.append(table(rows, [4.0, 2.2], font, bold_rows=(len(rows) - 1,)))
        else:
            itd = _itd(f, s["_paid_in"], style, rng)
            itd_vals = [0, itd["contributions"], -itd["distributions"] if outflow_paren else itd["distributions"],
                        -itd["management_fees"] if outflow_paren else itd["management_fees"],
                        -itd["incentive_fees"] if outflow_paren else itd["incentive_fees"],
                        itd["net_gain_loss"], f["ending_nav"]]
            rows = [["", "Quarter to Date", "Inception to Date"]]
            for (label, qtd), iv in zip(flow_rows, itd_vals):
                rows.append([label, qtd, a(iv)])
            story.append(table(rows, [3.4, 1.6, 1.6], font, bold_rows=(len(rows) - 1,)))

        story += [Spacer(1, 14), P("Commitment Summary", st["h"])]
        story.append(table([
            [lab["total_commitment"], a(f["total_commitment"])],
            [lab["paid_in"], a(s["_paid_in"])],
            [lab["unfunded_commitment"], a(f["unfunded_commitment"])],
        ], [4.0, 2.2], font, header=False))
        story += [Spacer(1, 16), P(
            "This statement is unaudited and is subject to adjustment upon completion of the "
            "annual audit. Please contact investor relations with any questions.", st["small"])]

    else:  # prose
        w = lambda v: fmt_money(v, style, ccy)  # noqa: E731
        gain = f["net_gain_loss"]
        story += [P(f["manager"], st["title"]),
                  P(fmt_date(end, date_style), st["body"]),
                  P(f"Dear {f['investor_name']},", st["body"]),
                  P(f"Re: {f['fund_name']} - capital account summary", st["body"])]
        paras = [
            f"We are pleased to provide a summary of your capital account in the Fund for the period "
            f"{period}. Your {lab['beginning_nav'].lower()} at the start of the period was {w(f['beginning_nav'])}.",
            (f"During the period you funded capital calls totalling {w(f['contributions'])}"
             if f["contributions"] else "There were no capital calls during the period")
            + (f", and the Fund made distributions to you of {w(f['distributions'])}."
               if f["distributions"] else ", and no distributions were made to you."),
            f"Your account was charged management fees of {w(f['management_fees'])}"
            + (f" and an incentive allocation of {w(f['incentive_fees'])}." if f["incentive_fees"]
               else "; no incentive allocation was charged this period."),
            (f"Your share of the Fund's net gain for the period was {w(gain)}" if gain >= 0
             else f"Your share of the Fund's net loss for the period was {w(-gain)}")
            + f", bringing your capital balance as of {fmt_date(end, date_style)} to {w(f['ending_nav'])}.",
            f"Of your total commitment of {w(f['total_commitment'])}, {w(f['unfunded_commitment'])} "
            f"remains unfunded and available to be called.",
            "As always, thank you for your continued partnership. Figures are unaudited.",
        ]
        story += [P(p, st["body"]) for p in paras]
        story += [Spacer(1, 10), P("Sincerely,", st["body"]), P("Investor Relations", st["body"])]

    build_pdf(path, story, title="Capital Account Statement", font=font)


def _itd(f: dict, paid_in: int, style: str, rng: random.Random) -> dict:
    """Inception-to-date column that rolls from zero to the same (printed) ending balance."""
    q = QUANTUM[style]
    d = {
        "contributions": paid_in,
        "distributions": quantize(paid_in * rng.uniform(0.0, 0.3), q),
        "management_fees": f["management_fees"] + quantize(f["total_commitment"] * rng.uniform(0.01, 0.05), q),
        "incentive_fees": f["incentive_fees"] + (quantize(paid_in * rng.uniform(0, 0.03), q)
                                                 if rng.random() < 0.5 else 0),
    }
    d["net_gain_loss"] = (f["ending_nav"] - d["contributions"] + d["distributions"]
                          + d["management_fees"] + d["incentive_fees"])
    return d
