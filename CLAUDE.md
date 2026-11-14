# Fund Document Extraction Agent

One-week prototype: synthetic fund PDFs -> router -> specialist extractors -> validation -> eval harness with regression tests.

## Hard rules
- Synthetic data only. Never use real fund documents.
- Model provider sits behind `agents/llm_client.py` (one interface, Anthropic implementation first). No provider SDK calls anywhere else.
- Reproducible: fixed random seed in the generator, prompts versioned as files (`agents/prompts/cas_v1.md`, ...), every pipeline run logged.
- API key comes from `.env` (`ANTHROPIC_API_KEY`). Never hardcode or commit it. `.env` is in `.gitignore`.
- Extractors must never read `data/ground_truth.json`. Only `evals/` reads it.
- Build one phase at a time. A phase is done only when its code runs and its tests pass. Commit at the end of each phase.
- Don't start stretch goals until phases 1-5 are done.

## Stack
Python 3.11+, anthropic SDK, pydantic v2, reportlab, pdfplumber, pytest, python-dotenv.

## Repo layout
```
generator/            synthetic PDF + ground truth generation
data/                 generated PDFs + ground_truth.json
agents/
  llm_client.py       provider-agnostic interface
  router.py
  extractors/         cas.py, soi.py, letter.py
  prompts/            versioned prompt files
schemas/              pydantic models
validation/           rules + exceptions report
evals/                scoring, run logging, baseline
tests/                unit + regression tests
runs/                 JSONL run logs
main.py               runs the full pipeline
```

## Phases
1. **Generator**: ~30 PDFs (10 CAS, 10 SOI, 10 LETTER), 2-3 layouts per type, varied labels ("Beginning Balance" / "Opening Capital"), varied number formats ("$1,234,567", "1.23mm", "(45,000)"), some data in tables and some in prose. Include about 4 intentionally broken docs (NAV doesn't roll forward, SOI totals don't sum, gross != long + short) and flag them in ground truth as `expected_exception: true`. Write `data/ground_truth.json` with the doc type and every field value. Fixed seed.
2. **CAS extractor end to end**: llm_client, CAS pydantic schema, prompt v1, structured output via tool use, PDF text via pdfplumber.
3. **Router + SOI and LETTER extractors**: router classifies CAS | SOI | LETTER | UNKNOWN.
4. **Validation layer**: pydantic validity plus finance rules with tolerances:
   - CAS: beginning_nav + contributions - distributions - management_fees - incentive_fees + net_gain_loss = ending_nav
   - CAS: unfunded_commitment <= total_commitment
   - SOI: sum(fair_value) ~= total_fair_value; sum(pct_of_nav) ~= 100
   - LETTER: gross = long + short; net = long - short
   - Low confidence or a failed rule -> `exceptions_report.csv` with the reason.
5. **Eval harness**: field-level accuracy vs ground truth (numbers within tolerance, dates normalized, strings fuzzy-matched), router accuracy, exception precision/recall. Log each run to `runs/*.jsonl` (timestamp, model, prompt versions, metrics). Save a baseline; pytest regression test fails if any field's accuracy drops more than 5 points vs baseline.
6. **README**: problem, architecture diagram, results table from the latest run, how to run.
7. Stretch (only if asked): MCP server with `get_fund_summary(fund_name)`; two-model comparison.

## Schemas
- CAS: fund_name, manager, investor_name, period_start, period_end, currency, beginning_nav, contributions, distributions, management_fees, incentive_fees, net_gain_loss, ending_nav, total_commitment, unfunded_commitment, confidence
- SOI: fund_name, as_of_date, holdings[] (name, sector, geography, cost, fair_value, pct_of_nav), total_fair_value, confidence
- LETTER: fund_name, quarter, long_exposure, short_exposure, gross_exposure, net_exposure, top_sector_exposures[], top_holdings[], quarterly_return, commentary_summary, confidence

## Commands
- Generate data: `python -m generator.generate`
- Run pipeline: `python main.py`
- Tests: `pytest`

## Current phase
Phase 2
