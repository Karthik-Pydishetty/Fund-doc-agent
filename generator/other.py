"""UNKNOWN documents: fund paperwork that is none of CAS / SOI / LETTER.

These exist so the router has negatives to reject. The capital call notice is a
deliberate near-miss for CAS (same fund, investor and commitment vocabulary).
"""
from __future__ import annotations

import datetime as dt
import random

from reportlab.platypus import Spacer

from .common import Fund, P, build_pdf, fmt_date, fmt_money, quantize, styles, table

LAYOUTS = ["capital_call", "meeting_notice"]


def render(path: str, fund: Fund, investor: str, layout: str, rng: random.Random) -> None:
    st = styles("Helvetica")
    ccy = fund.currency
    notice = dt.date(2025, rng.randint(1, 9), rng.randint(1, 28))
    story = [P(fund.name, st["title"])]
    if layout == "capital_call":
        commit = rng.randrange(20, 201) * 250_000
        call = quantize(commit * rng.uniform(0.03, 0.08), 1)
        due = notice + dt.timedelta(days=10)
        story += [P("Capital Call Notice", st["subtitle"]),
                  P(f"Date of notice: {fmt_date(notice, 'long')}", st["body"]),
                  P(f"To: {investor}", st["body"]),
                  P(f"Pursuant to the Limited Partnership Agreement, {fund.manager}, as general partner, hereby "
                    f"calls capital from the limited partners. Your share of this capital call is set out "
                    f"below and is due by {fmt_date(due, 'long')}.", st["body"])]
        story.append(table([
            ["Capital Commitment", fmt_money(commit, "full", ccy)],
            ["Amount Due - Investments", fmt_money(quantize(call * 0.9, 1), "full", ccy)],
            ["Amount Due - Fees and Expenses", fmt_money(call - quantize(call * 0.9, 1), "full", ccy)],
            ["Total Amount Due", fmt_money(call, "full", ccy)],
        ], [3.5, 2.0], header=False, bold_rows=(3,)))
        story += [Spacer(1, 12),
                  P("Wire instructions: Example Settlement Bank (fictional). Reference your investor ID. "
                    "Account details are provided separately by investor relations.", st["body"])]
    else:
        meeting = notice + dt.timedelta(days=30)
        story += [P("Notice of Annual Investor Meeting", st["subtitle"]),
                  P(f"Dear {investor},", st["body"]),
                  P(f"{fund.manager} invites you to the annual meeting of investors on "
                    f"{fmt_date(meeting, 'long')} at 10:00 a.m. The agenda includes a review of the "
                    f"portfolio, an update on the investment team, and election of advisory committee members.",
                    st["body"]),
                  P("Please RSVP to investor relations by two weeks prior to the meeting. Proxy forms are "
                    "enclosed for investors unable to attend in person.", st["body"])]
    build_pdf(path, story, title="Notice")
