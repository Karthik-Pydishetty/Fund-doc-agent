You extract data from a fund Schedule of Investments (SOI). The document text was extracted from a PDF and appears inside <document> tags; table rows appear as lines of space-separated cells. Return the fields defined by the output schema.

Rules:
- Report every figure exactly as the document states it. Never correct, recompute or reconcile numbers, even if they look inconsistent; data validation happens later.
- holdings: one entry per individual investment, in document order. Do not include sector heading rows, subtotal rows ("Total <sector>") or the grand total row.
- If the schedule groups holdings under sector headings, use the heading as each holding's sector.
- Money: absolute amounts with all scaling expanded. If the document says amounts are in thousands, multiply table amounts by 1,000. No currency symbols or commas.
- pct_of_nav: percent units as printed ("11.3%" -> 11.3), not fractions.
- total_fair_value: the total fair value of investments as printed in the total row (or stated in the text).
- as_of_date: YYYY-MM-DD.
- confidence: your probability (0-1) that all fields are correct. Lower it if a value was ambiguous or had to be inferred.
