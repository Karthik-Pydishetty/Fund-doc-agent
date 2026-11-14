"""README numbers are generated, never typed. Offline."""
import re
from pathlib import Path

from evals import readme

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "README.md").read_text()


def test_generated_blocks_are_current():
    """Re-render every block from the run log the README cites (and baseline.json); must match exactly."""
    log = readme.log_named_in_readme(TEXT)
    assert log is not None and log.exists(), "README results block must name an existing run log"
    assert readme.apply_blocks(TEXT, readme.render_blocks(log)) == TEXT, \
        "README.md is stale: run `python -m evals.readme`"


def test_no_hand_typed_numbers_outside_generated_blocks():
    prose = readme.BLOCK_RE.sub("", TEXT)
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)       # code blocks
    prose = re.sub(r"`[^`]*`", "", prose)                      # inline code (file names, prompt versions)
    prose = re.sub(r"<!--.*?-->", "", prose)
    prose = re.sub(r"\bdoc_\d+\b|\bPython 3\.11\+|\bv\d\b", "", prose)  # identifiers, not results
    stray = re.findall(r"[^\s]*\d[^\s]*", prose)
    assert not stray, f"numbers typed by hand in README.md (generate them instead): {stray}"


def test_required_sections_present():
    for heading in ("## The problem", "## Architecture", "## Quickstart", "## Results",
                    "## Design decisions", "## Limitations"):
        assert heading in TEXT
    assert "```mermaid" in TEXT
