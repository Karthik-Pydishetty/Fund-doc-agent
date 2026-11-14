"""PDF -> plain text via pdfplumber."""
from __future__ import annotations

from pathlib import Path

import pdfplumber


def pdf_to_text(path: str | Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(f"--- page {i} ---\n{t}" for i, t in enumerate(pages, 1))
