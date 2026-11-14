"""Capital Account Statement schema. Units follow the ground-truth conventions in CLAUDE.md."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

MONEY = ("Absolute amount in the statement currency, fully expanded: '£0.18mm' -> 180000, "
         "'€15.09 million' -> 15090000, '$1,235k' -> 1235000.")
OUTFLOW = MONEY + " Always a positive magnitude, even if printed as (12,500) or 'Less: ...'. 0 if none or '—'."


class CAS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fund_name: str = Field(description="Full legal name of the fund, as printed.")
    manager: str = Field(description="Manager / general partner / investment manager entity name.")
    investor_name: str = Field(description="Name of the investor (limited partner) the statement is for.")
    period_start: dt.date = Field(description="First day of the reporting period (YYYY-MM-DD).")
    period_end: dt.date = Field(description="Last day of the reporting period (YYYY-MM-DD).")
    currency: str = Field(description="ISO 4217 code of the statement currency, e.g. USD, EUR, GBP.")
    beginning_nav: float = Field(description="Capital account balance at the start of the period. " + MONEY)
    contributions: float = Field(description="Capital contributed / called during the period. " + OUTFLOW)
    distributions: float = Field(description="Distributions paid to the investor during the period. " + OUTFLOW)
    management_fees: float = Field(description="Management / advisory fees charged in the period. " + OUTFLOW)
    incentive_fees: float = Field(description="Incentive fee, carried interest or performance allocation "
                                              "charged in the period. " + OUTFLOW)
    net_gain_loss: float = Field(description="Net investment gain or loss for the period. Signed: a loss is "
                                             "negative. " + MONEY)
    ending_nav: float = Field(description="Capital account balance at the end of the period, exactly as "
                                          "printed. " + MONEY)
    total_commitment: float = Field(description="Investor's total capital commitment. " + MONEY)
    unfunded_commitment: float = Field(description="Remaining unfunded / uncalled commitment. " + MONEY)
    confidence: float = Field(ge=0, le=1, description="Your confidence (0-1) that every field above is correct.")
