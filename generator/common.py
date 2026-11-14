"""Shared helpers for the synthetic document generator: fictional funds,
number/date formatting, and the PDF builder.

Everything here is fictional. No real fund, manager or investor is modelled.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

# Byte-identical PDFs across runs (no timestamps / random document IDs).
rl_config.invariant = 1

FOOTER = "Synthetic test document. Fictional fund and figures; not investment material."


# --------------------------------------------------------------------------- funds

@dataclass(frozen=True)
class Fund:
    name: str
    manager: str
    strategy: str  # "private" (CAS/SOI) or "long_short" (LETTER)
    currency: str


FUNDS = [
    Fund("Granite Peak Capital Partners III, L.P.", "Granite Peak Advisors LLC", "private", "USD"),
    Fund("Bluewater Infrastructure Fund II, L.P.", "Bluewater Capital Management LP", "private", "USD"),
    Fund("Kestrel Growth Equity Fund IV, SCSp", "Kestrel Partners S.a r.l.", "private", "EUR"),
    Fund("Thornbury Private Credit Opportunities Fund", "Thornbury Asset Management LLP", "private", "GBP"),
    Fund("Silverline Long/Short Equity Fund, Ltd.", "Silverline Investment Partners LP", "long_short", "USD"),
    Fund("Ironbark Global Opportunities Fund, L.P.", "Ironbark Capital LLC", "long_short", "USD"),
    Fund("Calder Ridge Market Neutral Partners", "Calder Ridge Management LP", "long_short", "USD"),
]

INVESTORS = [
    "Halvorsen Family Trust",
    "Meridian Valley Teachers' Retirement System",
    "Ostrander Foundation",
    "Northgate University Endowment",
    "Pellworth Insurance Company",
    "Quintara Holdings Ltd.",
    "Redfern Municipal Employees' Pension Plan",
    "Sable Creek Family Office LLC",
]

SECTORS = [
    "Information Technology", "Health Care", "Industrials", "Consumer Discretionary",
    "Financials", "Energy", "Communication Services", "Materials", "Consumer Staples",
    "Real Estate", "Utilities",
]

GEOGRAPHIES = ["United States", "United Kingdom", "Germany", "France", "Canada",
               "Netherlands", "Japan", "Australia", "Nordics", "India"]

_NAME_A = ["Northwind", "Alder", "Brightwell", "Cobalt", "Driftwood", "Emberline", "Foxglove",
           "Halcyon", "Juniper", "Larkspur", "Moraine", "Nimbus", "Orchard", "Pinecrest",
           "Quarry", "Rookwood", "Saltmarsh", "Tidewater", "Umber", "Verity", "Wexford", "Yarrow"]
_NAME_B = ["Logistics", "Health", "Software", "Systems", "Foods", "Energy", "Robotics",
           "Analytics", "Materials", "Networks", "Diagnostics", "Brands", "Industries", "Labs"]
_NAME_C = ["Holdings", "Inc.", "Group", "GmbH", "Ltd.", "Corp.", "AG", "plc", "S.A.", "LLC"]


def company_names(rng: random.Random, n: int) -> list[str]:
    # Ordered de-dup (a set's iteration order varies with PYTHONHASHSEED across processes).
    names: list[str] = []
    while len(names) < n:
        name = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)} {rng.choice(_NAME_C)}"
        if name not in names:
            names.append(name)
    return names


# --------------------------------------------------------------------------- periods

QUARTERS = [(y, q) for y in (2024, 2025) for q in (1, 2, 3, 4)][:-1]  # 2024Q1 .. 2025Q3


def quarter_bounds(year: int, q: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, 3 * (q - 1) + 1, 1)
    end_month = 3 * q
    nxt = dt.date(year + (end_month == 12), end_month % 12 + 1, 1)
    return start, nxt - dt.timedelta(days=1)


DATE_STYLES = ["long", "us", "dmy_abbr", "iso", "dmy_long"]


def fmt_date(d: dt.date, style: str) -> str:
    if style == "long":
        return f"{d:%B} {d.day}, {d.year}"
    if style == "us":
        return f"{d:%m/%d/%Y}"
    if style == "dmy_abbr":
        return f"{d:%d-%b-%Y}"
    if style == "iso":
        return d.isoformat()
    if style == "dmy_long":
        return f"{d.day} {d:%B} {d.year}"
    raise ValueError(style)


# --------------------------------------------------------------------------- money

SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

# Every value is a multiple of its style's quantum, so the printed number is exact
# and ground truth equals what a careful reader would extract.
QUANTUM = {"full": 1, "plain": 1, "decimals": 1, "k": 1_000, "mm": 10_000, "words": 10_000}


def quantize(x: float, q: int) -> int:
    return int(round(x / q)) * q


def fmt_money(v: int, style: str, ccy: str = "USD") -> str:
    """Format an amount. Negatives use parentheses except in `decimals` (leading minus)."""
    sym = SYMBOLS[ccy]
    neg, a = v < 0, abs(v)
    if style == "full":
        s = f"{sym}{a:,.0f}"
    elif style == "plain":
        s = f"{a:,.0f}"
    elif style == "decimals":
        s = f"{sym}{a:,.2f}"
        return f"-{s}" if neg else s
    elif style == "k":
        s = f"{sym}{a / 1_000:,.0f}k"
    elif style == "mm":
        s = f"{sym}{a / 1_000_000:,.2f}mm"
    elif style == "words":
        s = f"{sym}{a / 1_000_000:,.2f} million"
    else:
        raise ValueError(style)
    return f"({s})" if neg else s


def fmt_pct(v: float, dp: int = 1, paren_neg: bool = False, plus: bool = False) -> str:
    if v < 0 and paren_neg:
        return f"({abs(v):.{dp}f})%"
    sign = "+" if plus and v > 0 else ""
    return f"{sign}{v:.{dp}f}%"


# --------------------------------------------------------------------------- PDF building

_BASE = getSampleStyleSheet()


def styles(font: str = "Helvetica") -> dict[str, ParagraphStyle]:
    bold = f"{font}-Bold" if font == "Helvetica" else "Times-Bold"
    return {
        "title": ParagraphStyle("t", parent=_BASE["Title"], fontName=bold, fontSize=16, spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=_BASE["Normal"], fontName=bold, fontSize=12,
                                   alignment=1, spaceAfter=10),
        "h": ParagraphStyle("h", parent=_BASE["Heading3"], fontName=bold, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=_BASE["Normal"], fontName=font, fontSize=10, leading=14,
                               spaceAfter=8),
        "small": ParagraphStyle("s", parent=_BASE["Normal"], fontName=font, fontSize=8, leading=10,
                                textColor=colors.grey),
        "bullet": ParagraphStyle("bl", parent=_BASE["Normal"], fontName=font, fontSize=10, leading=14,
                                 leftIndent=14, bulletIndent=4),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    """Paragraph with XML-escaped text (fund names contain '/', notes may contain '&')."""
    return Paragraph(escape(text), style)


def table(rows: list[list[str]], col_widths: list[float], font: str = "Helvetica",
          header: bool = True, bold_rows: tuple[int, ...] = (), grid: bool = False,
          right_cols_from: int = 1) -> Table:
    t = Table(rows, colWidths=[w * inch for w in col_widths], hAlign="LEFT")
    bold = "Helvetica-Bold" if font == "Helvetica" else "Times-Bold"
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (right_cols_from, 0), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        cmds += [("FONTNAME", (0, 0), (-1, 0), bold),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
                 ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF3"))]
    for r in bold_rows:
        cmds += [("FONTNAME", (0, r), (-1, r), bold), ("LINEABOVE", (0, r), (-1, r), 0.5, colors.black)]
    if grid:
        cmds.append(("GRID", (0, 0), (-1, -1), 0.25, colors.grey))
    t.setStyle(TableStyle(cmds))
    return t


def build_pdf(path: str, story: list, title: str, font: str = "Helvetica") -> None:
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.75 * inch, 0.5 * inch, FOOTER)
        canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=letter, title=title, author="synthetic-generator",
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.85 * inch)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
