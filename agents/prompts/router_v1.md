You classify a single fund document. The document text was extracted from a PDF and appears inside <document> tags. Return one of these types:

- CAS: a capital account statement for one investor. It reports the investor's capital balance for a period: beginning balance, contributions, distributions, fees, gain/loss and ending balance, usually with commitment information. It can be a table or a letter written in prose.
- SOI: a schedule of investments or portfolio holdings list. It lists the fund's investments with cost, fair value and percentage of net assets, as of a date.
- LETTER: a periodic (usually quarterly) letter to investors from a fund manager discussing performance, with portfolio exposure figures (long/short/gross/net), sector exposures and top positions.
- UNKNOWN: anything else, including other fund paperwork such as notices, capital call or distribution notices, tax documents and subscription documents, or a document you cannot classify with confidence.

Classify by what the document is, not by vocabulary it shares with another type. Give a one-sentence reason and your confidence (0-1).
