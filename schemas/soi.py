"""Schedule of Investments schema. Units follow the ground-truth conventions in CLAUDE.md."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

MONEY = ("Absolute amount in the document currency, fully expanded: '36,957' under an '(Amounts in "
         "thousands)' heading -> 36957000; '$1,234,567' -> 1234567.")
PCT = "Percent units as printed, not a fraction: '11.3%' -> 11.3."


class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Portfolio company / investment name, as printed.")
    sector: str = Field(description="Sector or industry of the holding, as printed (for grouped schedules, "
                                    "the sector heading it sits under).")
    geography: str = Field(description="Country or region of the holding, as printed.")
    cost: float = Field(description="Cost of the holding. " + MONEY)
    fair_value: float = Field(description="Fair value of the holding. " + MONEY)
    pct_of_nav: float = Field(description="Holding's percentage of net assets / NAV. " + PCT)


class SOI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fund_name: str = Field(description="Full legal name of the fund, as printed.")
    as_of_date: dt.date = Field(description="Date the schedule is as of (YYYY-MM-DD).")
    holdings: list[Holding] = Field(description="Every individual investment. Exclude sector headings, "
                                                "subtotal rows and total rows.")
    total_fair_value: float = Field(description="Total fair value of investments exactly as printed in the "
                                                "total row. " + MONEY)
    confidence: float = Field(ge=0, le=1, description="Your confidence (0-1) that every field above is correct.")
