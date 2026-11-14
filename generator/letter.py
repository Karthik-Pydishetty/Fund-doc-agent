"""Quarterly investor letters for long/short funds (LETTER).

Layouts:
  table     prose letter with an exposure table and a sector table
  prose     everything in running text
  snapshot  key-metrics block and bullet lists up top, commentary below

Exposures, returns and weights are percentages of NAV (142.5 means 142.5%).
"""
from __future__ import annotations

import random

from reportlab.platypus import Spacer

from .common import SECTORS, Fund, P, build_pdf, company_names, fmt_pct, styles, table

LAYOUTS = ["table", "prose", "snapshot"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}
QEND = {1: "March 31", 2: "June 30", 3: "September 30", 4: "December 31"}

MACRO = [
    "Rates volatility remained elevated as markets repriced the path of policy easing.",
    "Dispersion across single names widened, which is a constructive backdrop for stock picking.",
    "Earnings revisions were broadly positive, although guidance was cautious in cyclical areas.",
    "Market breadth narrowed, with a handful of large-cap names accounting for most index gains.",
]
OUTLOOK = {
    "cautious": "We enter the coming quarter with a cautious stance and have reduced net exposure at the margin.",
    "constructive": "We remain constructive and see a rich opportunity set on both the long and short side.",
    "neutral": "We are maintaining a balanced book and expect stock-specific drivers to dominate.",
}


def sample(rng: random.Random, fund: Fund, broken: str | None) -> dict:
    year, qtr = rng.choice([(2024, 3), (2024, 4), (2025, 1), (2025, 2), (2025, 3)])
    dp = rng.choice([0, 1])
    long_ = round(rng.uniform(90, 180), dp)
    short = round(rng.uniform(30, min(long_ - 5, 130)), dp)
    gross, net = round(long_ + short, dp), round(long_ - short, dp)

    reasons = []
    if broken == "gross_mismatch":
        gross = round(gross + rng.uniform(8, 25), dp)
        reasons.append("LETTER gross_exposure does not equal long_exposure + short_exposure")

    sectors = sorted(({"sector": s, "exposure_pct": round(rng.uniform(-12, 35), 1)}
                      for s in rng.sample(SECTORS, 4)), key=lambda x: -x["exposure_pct"])
    holdings = sorted(({"name": n, "pct_of_nav": round(rng.uniform(2.5, 8.0), 1)}
                       for n in company_names(rng, 5)), key=lambda x: -x["pct_of_nav"])
    ret = 0.0
    while abs(ret) < 0.3:
        ret = round(rng.uniform(-6, 9), 1)

    up, best = ret > 0, sectors[0]["sector"]
    drag = rng.choice([s["sector"] for s in sectors[1:]])
    outlook = rng.choice(list(OUTLOOK))
    macro = rng.choice(MACRO)
    if up:
        driver = (f"Performance was driven primarily by long positions in {best}, partially offset by "
                  f"losses on short positions in {drag}.")
        summary = (f"The fund returned {ret}% in Q{qtr} {year}, driven by {best} longs and partially "
                   f"offset by {drag} shorts; outlook {outlook}.")
    else:
        driver = (f"The main detractor was our long book in {best}, which more than offset gains from "
                  f"short positions in {drag}.")
        summary = (f"The fund lost {abs(ret)}% in Q{qtr} {year}, hurt by {best} longs despite gains on "
                   f"{drag} shorts; outlook {outlook}.")

    fields = {
        "fund_name": fund.name, "quarter": f"Q{qtr} {year}",
        "long_exposure": long_, "short_exposure": short, "gross_exposure": gross, "net_exposure": net,
        "top_sector_exposures": sectors, "top_holdings": holdings, "quarterly_return": ret,
        "commentary_summary": summary,
    }
    return {"fields": fields, "reasons": reasons,
            "_q": (year, qtr), "_dp": dp, "_text": {"driver": driver, "macro": macro, "outlook": OUTLOOK[outlook]}}


def _quarter_phrase(year: int, qtr: int, rng: random.Random) -> str:
    return rng.choice([f"Q{qtr} {year}", f"the {ORD[qtr]} quarter of {year}", f"{qtr}Q{str(year)[2:]}",
                       f"the quarter ended {QEND[qtr]}, {year}"])


def _return_phrase(ret: float, rng: random.Random) -> str:
    if ret > 0:
        return rng.choice([f"returned {fmt_pct(ret, plus=True)} net of fees",
                           f"gained {fmt_pct(ret)} net of fees", f"generated a net return of {fmt_pct(ret)}"])
    return rng.choice([f"returned {fmt_pct(ret, paren_neg=True)} net of fees",
                       f"declined {fmt_pct(-ret)} net of fees", f"returned {fmt_pct(ret)} net"])


def render(path: str, s: dict, layout: str, rng: random.Random) -> None:
    f, dp = s["fields"], s["_dp"]
    year, qtr = s["_q"]
    txt = s["_text"]
    font = rng.choice(["Helvetica", "Times-Roman"])
    st = styles(font)
    e = lambda v: f"{v:.{dp}f}%"  # noqa: E731
    qp = _quarter_phrase(year, qtr, rng)
    rp = _return_phrase(f["quarterly_return"], rng)
    holdings_str = ", ".join(f"{h['name']} ({h['pct_of_nav']:.1f}%)" for h in f["top_holdings"])
    sectors_str = "; ".join(f"{x['sector']} {x['exposure_pct']:.1f}%" for x in f["top_sector_exposures"])
    title = rng.choice([f"{ORD[qtr].title()} Quarter {year} Letter to Investors",
                        f"Quarterly Investor Letter - Q{qtr} {year}", f"Q{qtr} {year} Partner Update"])

    story = [P(f["fund_name"], st["title"]), P(title, st["subtitle"])]

    if layout == "snapshot":
        story.append(P("Portfolio Snapshot", st["h"]))
        story.append(table([
            ["Quarter", f"Q{qtr} {year}"],
            ["Net return (quarter)", fmt_pct(f["quarterly_return"], paren_neg=rng.random() < 0.5)],
            ["Long exposure", e(f["long_exposure"])],
            ["Short exposure", e(f["short_exposure"])],
            ["Gross exposure", e(f["gross_exposure"])],
            ["Net exposure", e(f["net_exposure"])],
        ], [2.5, 1.5], font, header=False, grid=True))
        story.append(P("Top net sector exposures (% of NAV)", st["h"]))
        story += [P(f"{x['sector']}: {x['exposure_pct']:.1f}%", st["bullet"]) for x in f["top_sector_exposures"]]
        story.append(P("Largest long positions (% of NAV)", st["h"]))
        story += [P(f"{h['name']} - {h['pct_of_nav']:.1f}%", st["bullet"]) for h in f["top_holdings"]]
        story.append(P("Commentary", st["h"]))
        story += [P(f"Dear Partners, the Fund {rp} in {qp}. {txt['driver']}", st["body"]),
                  P(txt["macro"], st["body"]), P(txt["outlook"], st["body"])]

    elif layout == "table":
        story += [P("Dear Partners,", st["body"]),
                  P(f"The Fund {rp} in {qp}. {txt['driver']} {txt['macro']}", st["body"]),
                  P("Exposure at quarter end, as a percentage of net assets:", st["body"])]
        story.append(table([["Long", "Short", "Gross", "Net"],
                            [e(f["long_exposure"]), e(f["short_exposure"]), e(f["gross_exposure"]),
                             e(f["net_exposure"])]], [1.2] * 4, font, grid=True, right_cols_from=0))
        story += [Spacer(1, 10), P("Net exposure by sector (top four):", st["body"])]
        story.append(table([["Sector", "Net Exposure"]] +
                           [[x["sector"], f"{x['exposure_pct']:.1f}%"] for x in f["top_sector_exposures"]],
                           [2.6, 1.3], font))
        story += [Spacer(1, 10),
                  P(f"Our five largest long positions at quarter end were {holdings_str}.", st["body"]),
                  P(txt["outlook"], st["body"])]

    else:  # prose
        story += [P("Dear Partners,", st["body"]),
                  P(f"For {qp}, the Fund {rp}. {txt['driver']}", st["body"]),
                  P(txt["macro"], st["body"]),
                  P(f"We ended the period {e(f['long_exposure'])} long and {e(f['short_exposure'])} short, "
                    f"for gross exposure of {e(f['gross_exposure'])} and net exposure of "
                    f"{e(f['net_exposure'])} of NAV.", st["body"]),
                  P(f"By sector, our largest net exposures were {sectors_str}. The top five positions were "
                    f"{holdings_str}, expressed as a percentage of NAV.", st["body"]),
                  P(txt["outlook"], st["body"])]

    story += [Spacer(1, 8), P("Thank you for your continued support.", st["body"]),
              P("Sincerely,", st["body"]), P(f"The {f['fund_name'].split(',')[0]} Investment Team", st["body"]),
              Spacer(1, 10),
              P("Performance figures are estimates, net of fees, and unaudited. Past performance is not "
                "indicative of future results.", st["small"])]
    build_pdf(path, story, title="Investor Letter", font=font)
