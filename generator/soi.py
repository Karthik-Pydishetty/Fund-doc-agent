"""Schedules of Investments (SOI).

Layouts:
  flat       one table, full currency amounts, total row
  grouped    holdings grouped under sector header rows with sector subtotals (traps)
  thousands  amounts printed in thousands, different column order, total also in prose

pct_of_nav is printed rounded, so sum(pct_of_nav) is ~100, not exactly 100.
"""
from __future__ import annotations

import random
from itertools import groupby

from reportlab.platypus import Spacer

from .common import (
    GEOGRAPHIES, SECTORS, Fund, P, build_pdf, company_names, fmt_date, fmt_money, quantize,
    quarter_bounds, styles, table,
)

LAYOUTS = ["flat", "grouped", "thousands"]


def sample(rng: random.Random, fund: Fund, layout: str, broken: str | None) -> dict:
    q = 1_000 if layout == "thousands" else 1
    year, qtr = rng.choice([(2024, 4), (2025, 1), (2025, 2), (2025, 3)])
    _, as_of = quarter_bounds(year, qtr)
    n = rng.randint(6, 12)
    sectors = rng.sample(SECTORS, k=rng.randint(3, 5))
    dp = rng.choice([1, 2])

    holdings = []
    for name in company_names(rng, n):
        cost = quantize(rng.uniform(1.5e6, 40e6), q)
        holdings.append({"name": name, "sector": rng.choice(sectors), "geography": rng.choice(GEOGRAPHIES),
                         "cost": cost, "fair_value": quantize(cost * rng.uniform(0.6, 1.9), q)})
    if layout == "grouped":
        holdings.sort(key=lambda h: (h["sector"], -h["fair_value"]))
    total = sum(h["fair_value"] for h in holdings)
    for h in holdings:
        h["pct_of_nav"] = round(h["fair_value"] / total * 100, dp)

    reasons = []
    if broken == "total_mismatch":
        total += quantize(total * rng.uniform(0.03, 0.08), q)
        reasons.append("SOI holdings fair values do not sum to total_fair_value")

    fields = {"fund_name": fund.name, "as_of_date": as_of.isoformat(), "holdings": holdings,
              "total_fair_value": total}
    return {"fields": fields, "reasons": reasons, "_as_of": as_of, "_dp": dp, "_ccy": fund.currency}


def render(path: str, s: dict, layout: str, rng: random.Random) -> None:
    f, ccy, dp, as_of = s["fields"], s["_ccy"], s["_dp"], s["_as_of"]
    font = rng.choice(["Helvetica", "Times-Roman"])
    st = styles(font)
    date_s = fmt_date(as_of, rng.choice(["long", "us", "dmy_long"]))
    pct = lambda v: f"{v:.{dp}f}%"  # noqa: E731
    story = [P(f["fund_name"], st["title"]),
             P(rng.choice(["Schedule of Investments", "Condensed Schedule of Investments",
                           "Portfolio Holdings"]), st["subtitle"]),
             P(f"As of {date_s}" + ("" if layout == "flat" else " (unaudited)"), st["body"])]
    hs = f["holdings"]

    if layout == "flat":
        m = lambda v: fmt_money(v, "full", ccy)  # noqa: E731
        rows = [["Investment", "Sector", "Geography", "Cost", "Fair Value", "% of Net Assets"]]
        rows += [[h["name"], h["sector"], h["geography"], m(h["cost"]), m(h["fair_value"]), pct(h["pct_of_nav"])]
                 for h in hs]
        rows.append(["Total Investments", "", "", m(sum(h["cost"] for h in hs)), m(f["total_fair_value"]),
                     pct(sum(h["pct_of_nav"] for h in hs))])
        story.append(table(rows, [2.0, 1.45, 1.0, 0.85, 0.85, 0.85], font, bold_rows=(len(rows) - 1,),
                           right_cols_from=3))

    elif layout == "grouped":
        m = lambda v: fmt_money(v, "plain", ccy)  # noqa: E731
        story.append(P(f"All amounts in {ccy}.", st["small"]))
        rows = [["Description", "Country", "Cost", "Fair Value", "% of NAV"]]
        bold = []
        for sector, grp in groupby(hs, key=lambda h: h["sector"]):
            grp = list(grp)
            rows.append([sector, "", "", "", ""])
            bold.append(len(rows) - 1)
            for h in grp:
                rows.append(["   " + h["name"], h["geography"], m(h["cost"]), m(h["fair_value"]),
                             pct(h["pct_of_nav"])])
            rows.append([f"   Total {sector}", "", m(sum(h["cost"] for h in grp)),
                         m(sum(h["fair_value"] for h in grp)), pct(sum(h["pct_of_nav"] for h in grp))])
        rows.append(["Total Investments at Fair Value", "", m(sum(h["cost"] for h in hs)),
                     m(f["total_fair_value"]), pct(sum(h["pct_of_nav"] for h in hs))])
        story.append(table(rows, [2.7, 1.2, 1.0, 1.0, 0.8], font, bold_rows=tuple(bold) + (len(rows) - 1,),
                           right_cols_from=2))

    else:  # thousands
        k = lambda v: f"{v / 1000:,.0f}"  # noqa: E731
        unit = {"USD": "U.S. dollars", "EUR": "euros", "GBP": "pounds sterling"}[ccy]
        story.append(P(f"(Amounts in thousands of {unit})", st["small"]))
        rows = [["Portfolio Company", "Country", "Industry", "Fair Value", "Cost", "% NAV"]]
        rows += [[h["name"], h["geography"], h["sector"], k(h["fair_value"]), k(h["cost"]), pct(h["pct_of_nav"])]
                 for h in hs]
        rows.append(["Total", "", "", k(f["total_fair_value"]), k(sum(h["cost"] for h in hs)),
                     pct(sum(h["pct_of_nav"] for h in hs))])
        story.append(table(rows, [2.0, 1.0, 1.5, 0.9, 0.9, 0.7], font, bold_rows=(len(rows) - 1,),
                           right_cols_from=3))
        story += [Spacer(1, 10), P(
            f"As of {date_s}, the Fund held {len(hs)} portfolio investments with an aggregate fair value "
            f"of {fmt_money(f['total_fair_value'], 'full', ccy)}.", st["body"])]

    story += [Spacer(1, 14), P(
        "Investments are valued at fair value in accordance with the Fund's valuation policy. "
        "Percentages are of net assets and may not sum due to rounding.", st["small"])]
    build_pdf(path, story, title="Schedule of Investments", font=font)
