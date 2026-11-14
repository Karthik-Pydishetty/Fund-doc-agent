"""Router decision schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["CAS", "SOI", "LETTER", "UNKNOWN"]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_type: DocType = Field(description="CAS, SOI, LETTER, or UNKNOWN.")
    reason: str = Field(description="One short sentence citing what in the document decided the type.")
    confidence: float = Field(ge=0, le=1, description="Your confidence (0-1) in doc_type.")
