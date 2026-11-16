# Fund Document Extraction Agent

One-week prototype: synthetic fund PDFs -> router -> specialist extractors -> validation -> eval harness with regression tests.

## Hard rules
- Synthetic data only. Never use real fund documents.
- Model provider sits behind `agents/llm_client.py` (one interface, Anthropic implementation first). No provider SDK calls anywhere else.
- Reproducible: fixed random seed in the generator, prompts versioned as files (`agents/prompts/cas_v1.md`, ...), every pipeline run logged.
- API key comes from `.env` (`ANTHROPIC_API_KEY`). Never hardcode or commit it. `.env` is in `.gitignore`.
- Extractors must never read `data/ground_truth.json`. Only `evals/` reads it.
- Build one phase at a time. A phase is done only when its code runs and its tests pass. Commit at the end of each phase.
- `data/` is frozen: never regenerate or edit it. New test material goes in a separate split (e.g. `data/hard/`).
- Only `agents/llm_client.py` calls a provider SDK. Only `evals/` reads ground truth.

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
7. **Hard split + Sonnet vs. Haiku**: a second, harder test set in `data/hard/` (own ground truth, own baseline; `data/` untouched). Run v1 prompts on it with Sonnet and Haiku side by side (accuracy, router, exceptions, cost, latency, every missed field). Sonnet v1 becomes the hard-split baseline.
8. ~~**v2 prompts for hard-split misses**~~ **Skipped**: v1 prompts scored 100% field accuracy on both splits with both Sonnet and Haiku, so there were no misses to target.
9. **Self-correction loop**: when validation fails or confidence is low, re-ask the extractor with the specific problem, bounded retries; must never "fix" a document that is genuinely inconsistent.
9b. **Close the blind spot with deterministic checks** (done): CAS period order / ~one-quarter span (80-100 days) / calendar-quarter alignment, LETTER quarter label format, non-negativity (fees, commitments, SOI cost, fair value, total), and `fund_name` must exactly match `data/funds.json` (near miss -> "fund name not in registry, closest: X"). `data/funds.json` is the one file added to `data/` (an ops fund master list, not ground truth). Faults are still chosen by the arithmetic rules only, so seeded runs inject the same faults. Compare runs per fault with `python -m evals.blind_spot BEFORE AFTER`.
10. **Read-only MCP server** (done): `mcp_server/` on the official MCP Python SDK (`mcp` 2.x, `MCPServer`) over stdio; three read-only tools over the latest logged run. No writes, no pipeline runs, no ground truth.
11. **Final README update**: hard split, model comparison, v2 and self-correction results, MCP usage.

## Schemas
- CAS: fund_name, manager, investor_name, period_start, period_end, currency, beginning_nav, contributions, distributions, management_fees, incentive_fees, net_gain_loss, ending_nav, total_commitment, unfunded_commitment, confidence
- SOI: fund_name, as_of_date, holdings[] (name, sector, geography, cost, fair_value, pct_of_nav), total_fair_value, confidence
- LETTER: fund_name, quarter, long_exposure, short_exposure, gross_exposure, net_exposure, top_sector_exposures[], top_holdings[], quarterly_return, commentary_summary, confidence
  - top_sector_exposures[]: {sector, exposure_pct}; top_holdings[]: {name, pct_of_nav}

## Ground-truth conventions (extractor schemas MUST use the same units)
- **Percent units, never fractions.** Every percentage is stored as the number printed before the `%`: a -0.5% return is `-0.5` (not `-0.005`), 119.2% long exposure is `119.2`, a 1.0% holding weight is `1.0`. Applies to quarterly_return, long/short/gross/net_exposure, exposure_pct, pct_of_nav. Prompts and pydantic field descriptions must say so explicitly.
- **Money in absolute units** of the document currency: "£0.18mm" -> `180000`, "€15.09 million" -> `15090000`, "36,957" in a table headed "(in thousands)" -> `36957000`.
- **CAS signs:** contributions, distributions, management_fees, incentive_fees are positive magnitudes even when printed as `(12,500)` or "Less: ..."; net_gain_loss is signed ("net loss of €0.78 million" -> `-780000`).
- **Dates** ISO `YYYY-MM-DD`; **quarter** `Q<n> <YYYY>` ("third quarter of 2025", "3Q25", "quarter ended September 30, 2025" -> `Q3 2025`).
- **Broken docs record the values as printed**, not corrected ones. E.g. doc_021 prints ending NAV €14.14 million although its roll-forward gives €13.77 million; ground truth is `14140000`. The extractor is graded on reading the page accurately and must never "fix" figures; catching the inconsistency is the validator's job (Phase 4), scored as exception precision/recall (Phase 5).
- **Rounded percentages:** pct_of_nav is printed rounded (1-2 dp), so clean SOIs sum to ~100, not exactly 100. Measured on the current data: clean docs deviate at most ±0.10 (range 99.90-100.10). The Phase 4 tolerance is **±0.5**; `tests/test_generator.py` asserts every clean SOI stays inside it. Fair values have no rounding: sum(fair_value) == total_fair_value exactly on clean docs.

## Eval notes (Phase 5)
- `commentary_summary` is free text; two good summaries can share no wording. Exclude it from field-level accuracy (report it separately at most).
- `data/` (PDFs + ground_truth.json) is committed and frozen as the test set. If the generator changes, regenerating `data/` is a deliberate, separately committed change; re-baseline evals after it.

## Model config
- One config value: `MODEL` in `config.py` (default `claude-sonnet-5-5`, override with `FUND_AGENT_MODEL` env var). Nothing else names a model.
- Sonnet 5.5 rejects forced `tool_choice` (`any`/`tool`) with a 400. For structured output use `tool_choice: auto` + `strict: true` on the tool, or structured outputs (`output_config.format`).

## Commands
- Generate data: `python -m generator.generate`
- Run pipeline: `python main.py`
- Tests: `pytest`
- Score latest pipeline run (offline): `python -m evals.compare` (`--doc doc_007` for one doc)
- CAS side-by-side: `python -m evals.compare_cas` (`--from-log` to rebuild offline)
- Validate latest pipeline log (offline) -> exceptions_report.csv: `python -m validation.run`
- Exception precision/recall vs ground truth (offline): `python -m evals.exceptions`
- Full eval (API): `python -m evals.run_eval` (`--prompt cas=v2` to try a prompt file without code changes; `--from-log` to score an existing log; `--save-baseline` to reset evals/baseline.json)
- Regression gate (offline, part of `pytest`): `tests/test_regression.py` vs `evals/baseline.json`; `EVAL_LOG=path pytest tests/test_regression.py` to check a specific log
- Hard split: `python -m generator.hard` (writes data/hard/ only); `python -m evals.run_eval --split hard [--model claude-haiku-5-5]`; regression gate checks each split vs its own baseline (`evals/baseline.json`, `evals/baseline_hard.json`) using the newest run by the baseline's model
- Model comparison (offline): `python -m evals.compare_models --split hard` (or pass two run logs)
- Self-correction: `--self-correct` on main.py / run_eval (one retry on finance-rule failures; changed values need a verbatim quote checked in code). Measure it with `--inject-faults N [--fault-seed S]` (experiment runs; ignored by the regression gate, never saved as a baseline), then `python -m evals.self_correction`
- MCP server (read-only, stdio): `python -m mcp_server` (tools: list_exceptions, get_document, get_fund_summary). Reads the newest main-split, non-fault run log (override `FUND_DOCS_LOG=<log>` or `FUND_DOCS_SPLIT=hard`) and `exceptions_report.csv` only when it matches that run; never ground truth, never a model, never a write. Tests: `tests/test_mcp_server.py`

- Results report (offline): `python -m evals.report` -> `reports/results.html` (self-contained; headline, gate, per-field, Sonnet vs Haiku, self-correction, exceptions, run history). Tests: `tests/test_report.py`

## Current phase
Phase 11
