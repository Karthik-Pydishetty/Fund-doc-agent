"""Hard split: 10 realistic, harder documents in data/hard/ with their own ground_truth.json.

    python -m generator.hard [--seed 42] [--out data/hard]

Same conventions as the main split (CLAUDE.md). Every difficulty is recoverable by a careful
reader; nothing is corrupted beyond repair. data/ (the main split) is never touched.

  cas_footer_units    amounts unlabeled in the table; "in $000s" appears only in a footer note
  cas_q3_9m           "Q3 2025" and "9M 2025" columns, no QTD/YTD wording
  cas_combined_fees   one "management fees and incentive allocation" line, split only in a footnote
  cas_two_column      two-column page layout (pdfplumber interleaves the columns line by line)
  cas_ocr             recoverable OCR noise in the text layer: O for 0, stray spaces inside numbers
  soi_two_page        ~38 holdings across two pages, table header repeated, total on page 2
  soi_pct_broken      one holding's % of NAV misprinted, so weights don't sum to 100 (expected_exception)
  letter_pct_of_gross long exposure given only as % of gross; long/short/net must be computed
  letter_paren_shorts short exposure and a net-short sector printed in parentheses
  k1_cover            Schedule K-1 cover letter (UNKNOWN)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter as LETTER_SIZE
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Spacer

from . import cas, letter, soi
from .common import (
    FOOTER, FUNDS, GEOGRAPHIES, INVESTORS, SECTORS, P, company_names, fmt_date, fmt_money, fmt_pct, quantize,
    styles, table,
)

SEED = 42
VERSION = "hard-1.0"
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = [f for f in FUNDS if f.strategy == "private" and f.currency == "USD"]
LONG_SHORT = [f for f in FUNDS if f.strategy == "long_short"]
Q3_2025 = (dt.date(2025, 7, 1), dt.date(2025, 9, 30))


# --------------------------------------------------------------------------- PDF plumbing

def _build(path: Path, story: list, title: str, footer_note: str | None = None,
           later_header: str | None = None, two_column: bool = False) -> None:
    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        if footer_note:
            canvas.drawString(0.75 * inch, 0.68 * inch, footer_note)
        canvas.drawString(0.75 * inch, 0.5 * inch, FOOTER)
        canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
        if later_header and doc.page > 1:
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.black)
            canvas.drawString(0.75 * inch, 10.5 * inch, later_header)
        canvas.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=LETTER_SIZE, title=title, author="synthetic-generator",
                          leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch,
                          bottomMargin=0.95 * inch)
    w, h = doc.width, doc.height
    if two_column:
        gap = 0.3 * inch
        cw = (w - gap) / 2
        frames = [Frame(doc.leftMargin, doc.bottomMargin, cw, h, id="left"),
                  Frame(doc.leftMargin + cw + gap, doc.bottomMargin, cw, h, id="right")]
    else:
        frames = [Frame(doc.leftMargin, doc.bottomMargin, w, h, id="main")]
    doc.addPageTemplates([PageTemplate(id="p", frames=frames, onPage=deco)])
    doc.build(story)


def _recompute_ending(f: dict) -> None:
    f["ending_nav"] = (f["beginning_nav"] + f["contributions"] - f["distributions"] - f["management_fees"]
                       - f["incentive_fees"] + f["net_gain_loss"])


def _cas_fields(rng: random.Random, style: str = "full", force_incentive: bool = False) -> dict:
    fund, investor = rng.choice(PRIVATE), rng.choice(INVESTORS)
    s = cas.sample(rng, fund, investor, style, None)
    f = s["fields"]
    f["period_start"], f["period_end"] = (d.isoformat() for d in Q3_2025)
    if force_incentive and f["incentive_fees"] == 0:
        f["net_gain_loss"] = abs(f["net_gain_loss"]) or quantize(f["beginning_nav"] * 0.03, cas.QUANTUM[style])
        f["incentive_fees"] = quantize(f["net_gain_loss"] * 0.15, cas.QUANTUM[style])
        _recompute_ending(f)
    return {"fields": f, "paid_in": s["_paid_in"]}


def _header_rows(f: dict, period: str) -> list[list[str]]:
    return [["Investor:", f["investor_name"]], ["General Partner:", f["manager"]], ["Period:", period]]


# --------------------------------------------------------------------------- CAS

def cas_footer_units(rng, path):
    c = _cas_fields(rng, "k")
    f = c["fields"]
    st = styles()
    k = lambda v: "-" if v == 0 else (f"({abs(v) / 1000:,.0f})" if v < 0 else f"{v / 1000:,.0f}")  # noqa: E731
    story = [P(f["fund_name"], st["title"]), P("Partner Capital Account Statement", st["subtitle"]),
             table(_header_rows(f, "July 1, 2025 through September 30, 2025"), [1.6, 4.8], header=False,
                   right_cols_from=9), Spacer(1, 12),
             table([["", "Amount"],
                    ["Opening capital", k(f["beginning_nav"])], ["Capital contributions", k(f["contributions"])],
                    ["Distributions", k(-f["distributions"])], ["Management fee", k(-f["management_fees"])],
                    ["Incentive allocation", k(-f["incentive_fees"])],
                    ["Net investment gain (loss)", k(f["net_gain_loss"])],
                    ["Closing capital", k(f["ending_nav"])]], [4.0, 2.0], bold_rows=(7,)),
             Spacer(1, 12),
             table([["Committed capital", k(f["total_commitment"])], ["Called to date", k(c["paid_in"])],
                    ["Remaining commitment", k(f["unfunded_commitment"])]], [4.0, 2.0], header=False)]
    _build(path, story, "Capital Account Statement", footer_note="Amounts in $000s unless otherwise stated.")
    return "CAS", f, None


def cas_q3_9m(rng, path):
    c = _cas_fields(rng, "full")
    f = c["fields"]
    nine = {"beginning_nav": quantize(f["beginning_nav"] * rng.uniform(0.82, 0.97), 1),
            "contributions": f["contributions"] + quantize(f["total_commitment"] * rng.uniform(0.02, 0.06), 1),
            "distributions": f["distributions"] + quantize(f["beginning_nav"] * rng.uniform(0.0, 0.05), 1),
            "management_fees": f["management_fees"] * 3,
            "incentive_fees": f["incentive_fees"] + quantize(f["beginning_nav"] * rng.uniform(0, 0.01), 1)}
    nine["net_gain_loss"] = (f["ending_nav"] - nine["beginning_nav"] - nine["contributions"] + nine["distributions"]
                             + nine["management_fees"] + nine["incentive_fees"])
    st = styles()
    m = lambda v: fmt_money(v, "full")  # noqa: E731
    keys = [("Balance, beginning of period", "beginning_nav", 1), ("Contributions", "contributions", 1),
            ("Distributions", "distributions", -1), ("Management fees", "management_fees", -1),
            ("Incentive allocation", "incentive_fees", -1), ("Net gain (loss)", "net_gain_loss", 1)]
    rows = [["", "Q3 2025", "9M 2025"]] + [[lab, m(sign * f[k]), m(sign * nine[k])] for lab, k, sign in keys]
    rows.append(["Balance, end of period", m(f["ending_nav"]), m(f["ending_nav"])])
    story = [P(f["fund_name"], st["title"]), P("Statement of Changes in Partner's Capital", st["subtitle"]),
             P("Periods ended September 30, 2025", st["body"]),
             table([["Investor:", f["investor_name"]], ["General Partner:", f["manager"]]], [1.6, 4.8],
                   header=False, right_cols_from=9), Spacer(1, 12),
             table(rows, [3.0, 1.6, 1.6], bold_rows=(len(rows) - 1,)), Spacer(1, 12),
             table([["Capital commitment", m(f["total_commitment"])],
                    ["Unfunded commitment", m(f["unfunded_commitment"])]], [3.0, 1.6], header=False)]
    _build(path, story, "Statement of Changes in Partner's Capital")
    return "CAS", f, None


def cas_combined_fees(rng, path):
    c = _cas_fields(rng, "full", force_incentive=True)
    f = c["fields"]
    st = styles()
    m = lambda v: fmt_money(v, "full")  # noqa: E731
    story = [P(f["fund_name"], st["title"]), P("Capital Account Statement", st["subtitle"]),
             table(_header_rows(f, "Quarter ended September 30, 2025"), [1.6, 4.8], header=False,
                   right_cols_from=9), Spacer(1, 12),
             table([["Capital Account", "USD"],
                    ["Beginning balance", m(f["beginning_nav"])], ["Contributions", m(f["contributions"])],
                    ["Distributions", m(-f["distributions"])],
                    ["Management fees and incentive allocation (Note 2)",
                     m(-(f["management_fees"] + f["incentive_fees"]))],
                    ["Net realized and unrealized gain (loss)", m(f["net_gain_loss"])],
                    ["Ending balance", m(f["ending_nav"])]], [4.2, 2.0], bold_rows=(6,)),
             Spacer(1, 12),
             table([["Total commitment", m(f["total_commitment"])],
                    ["Unfunded commitment", m(f["unfunded_commitment"])]], [4.2, 2.0], header=False),
             Spacer(1, 16), P("Notes", st["h"]),
             P("Note 1. Figures are unaudited.", st["small"]),
             P(f"Note 2. Fees and allocations for the quarter comprise a management fee of "
               f"{m(f['management_fees'])} and an incentive allocation of {m(f['incentive_fees'])}.", st["small"])]
    _build(path, story, "Capital Account Statement")
    return "CAS", f, None


def cas_two_column(rng, path):
    c = _cas_fields(rng, "full")
    f = c["fields"]
    st = styles()
    m = lambda v: fmt_money(v, "full")  # noqa: E731
    left = [P(f["fund_name"], st["h"]), P("Investor Capital Statement", st["body"]),
            P(f"Investor: {f['investor_name']}", st["body"]), P(f"Manager: {f['manager']}", st["body"]),
            P(f"Period: {fmt_date(Q3_2025[0], 'us')} - {fmt_date(Q3_2025[1], 'us')}", st["body"]),
            P("Capital activity", st["h"]),
            table([["Opening balance", m(f["beginning_nav"])], ["Contributions", m(f["contributions"])],
                   ["Distributions", m(-f["distributions"])], ["Management fees", m(-f["management_fees"])],
                   ["Incentive allocation", m(-f["incentive_fees"])], ["Net gain (loss)", m(f["net_gain_loss"])],
                   ["Closing balance", m(f["ending_nav"])]], [1.9, 1.3], header=False, bold_rows=(6,))]
    right = [P("Commitment summary", st["h"]),
             table([["Commitment", m(f["total_commitment"])], ["Paid in to date", m(c["paid_in"])],
                    ["Unfunded", m(f["unfunded_commitment"])]], [1.9, 1.3], header=False),
             P("Notes", st["h"]),
             P("Amounts are in U.S. dollars and unaudited. Distributions and fees are shown as negative "
               "amounts. Contact investor relations with any questions about this statement.", st["body"])]
    _build(path, left + [FrameBreak()] + right, "Investor Capital Statement", two_column=True)
    return "CAS", f, None


def _ocr(s: str, rng: random.Random) -> str:
    out = []
    for i, ch in enumerate(s):
        if ch == "0" and rng.random() < 0.5:
            ch = "O"
        out.append(ch)
        if ch == "," and i < len(s) - 1 and rng.random() < 0.35:
            out.append(" ")
    return "".join(out)


def cas_ocr(rng, path):
    c = _cas_fields(rng, "full")
    f = c["fields"]
    st = styles()
    o = lambda v: _ocr(fmt_money(v, "full"), rng)  # noqa: E731
    story = [P(f["fund_name"], st["title"]), P("CAPITAL  ACCOUNT  STATEMENT", st["subtitle"]),
             table([["Investor:", f["investor_name"]], ["Manager:", f["manager"]],
                    ["Period:", _ocr("07/01/2025 - 09/30/2025", rng)]], [1.6, 4.8], header=False,
                   right_cols_from=9), Spacer(1, 12),
             table([["Description", "Amount"],
                    ["Beginning capital", o(f["beginning_nav"])], ["Contributions", o(f["contributions"])],
                    ["Distributions", o(-f["distributions"])], ["Management fees", o(-f["management_fees"])],
                    ["Incentive allocation", o(-f["incentive_fees"])], ["Net gain (loss)", o(f["net_gain_loss"])],
                    ["Ending capital", o(f["ending_nav"])]], [4.0, 2.0], bold_rows=(7,)),
             Spacer(1, 12),
             table([["Total commitment", o(f["total_commitment"])],
                    ["Unfunded commitment", o(f["unfunded_commitment"])]], [4.0, 2.0], header=False),
             Spacer(1, 12), P("Scanned copy. Text layer produced by optical character recognition.", st["small"])]
    _build(path, story, "Capital Account Statement")
    return "CAS", f, None


# --------------------------------------------------------------------------- SOI

def _soi_rows(f: dict, dp: int) -> list[list[str]]:
    m = lambda v: fmt_money(v, "full")  # noqa: E731
    rows = [["Investment", "Sector", "Country", "Cost", "Fair Value", "% of Net Assets"]]
    rows += [[h["name"], h["sector"], h["geography"], m(h["cost"]), m(h["fair_value"]), f"{h['pct_of_nav']:.{dp}f}%"]
             for h in f["holdings"]]
    rows.append(["Total investments", "", "", m(sum(h["cost"] for h in f["holdings"])), m(f["total_fair_value"]),
                 f"{sum(h['pct_of_nav'] for h in f['holdings']):.{dp}f}%"])
    return rows


def soi_two_page(rng, path):
    fund = rng.choice(PRIVATE)
    sectors = rng.sample(SECTORS, 6)
    holdings = []
    for name in company_names(rng, 38):
        cost = quantize(rng.uniform(1.5e6, 30e6), 1)
        holdings.append({"name": name, "sector": rng.choice(sectors), "geography": rng.choice(GEOGRAPHIES),
                         "cost": cost, "fair_value": quantize(cost * rng.uniform(0.6, 1.9), 1)})
    total = sum(h["fair_value"] for h in holdings)
    for h in holdings:
        h["pct_of_nav"] = round(h["fair_value"] / total * 100, 2)
    f = {"fund_name": fund.name, "as_of_date": "2025-09-30", "holdings": holdings, "total_fair_value": total}
    st = styles()
    t = table(_soi_rows(f, 2), [1.8, 1.35, 0.95, 0.95, 0.95, 1.0], bold_rows=(len(holdings) + 1,),
              right_cols_from=3)
    t.repeatRows = 1
    story = [P(f["fund_name"], st["title"]), P("Schedule of Investments", st["subtitle"]),
             P("As of September 30, 2025", st["body"]), t]
    _build(path, story, "Schedule of Investments", later_header=f"{fund.name} - Schedule of Investments (continued)")
    return "SOI", f, None


def soi_pct_broken(rng, path):
    fund = rng.choice(PRIVATE)
    s = soi.sample(rng, fund, "flat", None)
    f, dp = s["fields"], s["_dp"]
    f["as_of_date"] = "2025-09-30"
    victim = rng.choice(f["holdings"])
    victim["pct_of_nav"] = round(victim["pct_of_nav"] + rng.uniform(5, 9), dp)
    st = styles()
    story = [P(f["fund_name"], st["title"]), P("Schedule of Investments", st["subtitle"]),
             P("As of September 30, 2025", st["body"]),
             table(_soi_rows(f, dp), [1.8, 1.35, 0.95, 0.95, 0.95, 1.0], bold_rows=(len(f["holdings"]) + 1,),
                   right_cols_from=3)]
    _build(path, story, "Schedule of Investments")
    return "SOI", f, "SOI holdings pct_of_nav do not sum to ~100"


# --------------------------------------------------------------------------- LETTER

def _letter_base(rng):
    s = letter.sample(rng, rng.choice(LONG_SHORT), None)
    s["fields"]["quarter"] = "Q3 2025"
    s["fields"]["commentary_summary"] = re.sub(r"Q\d \d{4}", "Q3 2025", s["fields"]["commentary_summary"])
    return s


def letter_pct_of_gross(rng, path):
    s = _letter_base(rng)
    f, txt = s["fields"], s["_text"]
    gross, share = rng.randint(150, 260), rng.randint(55, 80)
    f["gross_exposure"] = float(gross)
    f["long_exposure"] = round(gross * share / 100, 2)
    f["short_exposure"] = round(gross - f["long_exposure"], 2)
    f["net_exposure"] = round(f["long_exposure"] - f["short_exposure"], 2)
    st = styles()
    sectors = "; ".join(f"{x['sector']} {x['exposure_pct']:.1f}%" for x in f["top_sector_exposures"])
    holds = ", ".join(f"{h['name']} ({h['pct_of_nav']:.1f}%)" for h in f["top_holdings"])
    story = [P(f["fund_name"], st["title"]), P("Third Quarter 2025 Investor Letter", st["subtitle"]),
             P("Dear Partners,", st["body"]),
             P(f"The Fund returned {fmt_pct(f['quarterly_return'], paren_neg=True)} net of fees in the third quarter "
               f"of 2025. {txt['driver']} {txt['macro']}", st["body"]),
             P(f"Gross exposure finished the quarter at {gross}% of NAV, with long positions making up {share}% "
               f"of gross exposure and short positions the remainder.", st["body"]),
             P(f"Our largest net sector exposures were {sectors}. The five largest positions, as a percentage of "
               f"NAV, were {holds}.", st["body"]),
             P(txt["outlook"], st["body"]), P("Sincerely, The Investment Team", st["body"])]
    _build(path, story, "Investor Letter")
    return "LETTER", f, None


def letter_paren_shorts(rng, path):
    s = _letter_base(rng)
    f, txt = s["fields"], s["_text"]
    last = f["top_sector_exposures"][-1]
    last["exposure_pct"] = -round(rng.uniform(3, 12), 1)
    f["top_sector_exposures"].sort(key=lambda x: -x["exposure_pct"])
    st = styles()
    pct = lambda v: f"({abs(v):.1f})%" if v < 0 else f"{v:.1f}%"  # noqa: E731
    story = [P(f["fund_name"], st["title"]), P("Q3 2025 Letter to Partners", st["subtitle"]),
             P(f"Dear Partners, the Fund returned {fmt_pct(f['quarterly_return'], paren_neg=True)} for the quarter. "
               f"{txt['driver']}", st["body"]),
             P("Exposure (% of NAV)", st["h"]),
             table([["Long", "Short", "Gross", "Net"],
                    [pct(f["long_exposure"]), pct(-f["short_exposure"]), pct(f["gross_exposure"]),
                     pct(f["net_exposure"])]], [1.2] * 4, grid=True, right_cols_from=0),
             P("Net sector exposure (% of NAV)", st["h"]),
             table([["Sector", "Net"]] + [[x["sector"], pct(x["exposure_pct"])] for x in f["top_sector_exposures"]],
                   [2.6, 1.2]),
             P("Top positions (% of NAV)", st["h"]),
             table([["Position", "Weight"]] + [[h["name"], f"{h['pct_of_nav']:.1f}%"] for h in f["top_holdings"]],
                   [2.6, 1.2]),
             Spacer(1, 10), P(txt["macro"], st["body"]), P(txt["outlook"], st["body"])]
    _build(path, story, "Investor Letter")
    return "LETTER", f, None


# --------------------------------------------------------------------------- UNKNOWN

def k1_cover(rng, path):
    fund, investor = rng.choice(PRIVATE), rng.choice(INVESTORS)
    st = styles()
    story = [P(fund.manager, st["title"]), P("March 14, 2025", st["body"]), P(f"Dear {investor},", st["body"]),
             P(f"Re: {fund.name} - Schedule K-1 (Form 1065) for tax year 2024", st["body"]),
             P("Enclosed is your Schedule K-1 reporting your distributive share of the Partnership's income, "
               "deductions and credits for the 2024 tax year. Please provide it to your tax advisor. The "
               "capital account analysis in Part II, item L of the K-1 is prepared on a tax basis and will "
               "differ from the capital balances shown in your quarterly capital account statements.", st["body"]),
             P("State composite filings and withholding details are summarized on the attached supplemental "
               "schedule. If you require an amended K-1 or have questions about your tax reporting, please "
               "contact our tax department.", st["body"]),
             P("Sincerely,", st["body"]), P("Tax Reporting", st["body"])]
    _build(path, story, "K-1 Cover Letter")
    return "UNKNOWN", {}, None


BUILDERS = [cas_footer_units, cas_q3_9m, cas_combined_fees, cas_two_column, cas_ocr,
            soi_two_page, soi_pct_broken, letter_pct_of_gross, letter_paren_shorts, k1_cover]


def generate(out_dir: str | Path = ROOT / "data" / "hard", seed: int = SEED) -> dict:
    out = Path(out_dir)
    if out.resolve() == (ROOT / "data").resolve():
        raise SystemExit("refusing to write into data/ (main split is frozen)")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("hard_*.pdf"):
        old.unlink()
    order = list(range(len(BUILDERS)))
    random.Random(f"{seed}:hard:order").shuffle(order)
    docs = []
    for n, i in enumerate(order, start=1):
        build = BUILDERS[i]
        rng = random.Random(f"{seed}:hard:{build.__name__}")
        doc_id = f"hard_{n:03d}"
        doc_type, fields, reason = build(rng, out / f"{doc_id}.pdf")
        docs.append({"doc_id": doc_id, "file": f"{doc_id}.pdf", "doc_type": doc_type, "layout": build.__name__,
                     "challenge": _CHALLENGES[build.__name__],
                     "number_format": None, "expected_exception": reason is not None,
                     "exception_reasons": [reason] if reason else [], "fields": fields})
    truth = {"generator_version": VERSION, "seed": seed, "split": "hard",
             "conventions": "same as data/ground_truth.json (see CLAUDE.md)", "documents": docs}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    return truth


_CHALLENGES = {
    "cas_footer_units": "units ('in $000s') appear only in a footer note",
    "cas_q3_9m": "'Q3 2025' and '9M 2025' columns with no QTD/YTD wording; extract Q3",
    "cas_combined_fees": "management and incentive fees on one line, split only in Note 2",
    "cas_two_column": "two-column page layout; extracted text interleaves the columns",
    "cas_ocr": "OCR text layer: letter O for zero, stray spaces inside numbers",
    "soi_two_page": "38 holdings over two pages, header repeated, total on page 2",
    "soi_pct_broken": "one holding's % of NAV misprinted; weights don't sum to 100",
    "letter_pct_of_gross": "long exposure given only as % of gross; long/short/net must be computed",
    "letter_paren_shorts": "short exposure and a net-short sector printed in parentheses",
    "k1_cover": "Schedule K-1 cover letter; not a CAS despite capital account wording",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ROOT / "data" / "hard"))
    args = ap.parse_args()
    truth = generate(args.out, args.seed)
    print(f"Wrote {len(truth['documents'])} hard PDFs to {args.out}: "
          + ", ".join(f"{d['doc_id']}={d['layout']}" for d in truth["documents"]))


if __name__ == "__main__":
    main()
