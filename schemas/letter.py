"""Quarterly investor letter schema. Units follow the ground-truth conventions in CLAUDE.md."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PCT = "Percent of NAV in percent units as printed, not a fraction: '142.5%' -> 142.5."


class SectorExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector: str = Field(description="Sector name, as printed.")
    exposure_pct: float = Field(description="Net exposure to the sector. Signed. " + PCT)


class TopHolding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Position / company name, as printed.")
    pct_of_nav: float = Field(description="Position size. " + PCT)


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fund_name: str = Field(description="Full legal name of the fund, as printed.")
    quarter: str = Field(description="Reporting quarter formatted 'Q<n> <YYYY>', e.g. 'third quarter of 2025', "
                                     "'3Q25' or 'quarter ended September 30, 2025' -> 'Q3 2025'.")
    long_exposure: float = Field(description="Long exposure. " + PCT)
    short_exposure: float = Field(description="Short exposure as a positive number. " + PCT)
    gross_exposure: float = Field(description="Gross exposure exactly as printed. " + PCT)
    net_exposure: float = Field(description="Net exposure exactly as printed. " + PCT)
    top_sector_exposures: list[SectorExposure] = Field(description="Sector exposures listed in the letter, "
                                                                   "in the order given.")
    top_holdings: list[TopHolding] = Field(description="Top / largest positions listed in the letter, in the "
                                                       "order given.")
    quarterly_return: float = Field(description="Fund's net return for the quarter. Signed: '(0.5)%', "
                                                "'declined 0.5%' and '-0.5%' -> -0.5. " + PCT)
    commentary_summary: str = Field(description="One or two sentences summarising the performance drivers "
                                                "and outlook in the commentary.")
    confidence: float = Field(ge=0, le=1, description="Your confidence (0-1) that every field above is correct.")
