"""Read-only MCP server over the latest pipeline run (stdio transport).

    python -m mcp_server            # what Claude Desktop launches

Tools: list_exceptions, get_document, get_fund_summary. Each call re-reads the run log, so a new
pipeline run shows up without restarting. Read-only: no model calls, no writes, no ground truth.
Choose a different run with FUND_DOCS_LOG=<path to runs/..._pipeline.jsonl> or FUND_DOCS_SPLIT=hard.
"""
from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from mcp_server import store

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)

server = MCPServer(
    name="fund-docs",
    instructions=(
        "Read-only access to the fund document pipeline's latest logged run: synthetic capital account "
        "statements (CAS), schedules of investments (SOI) and quarterly investor letters. Figures are as "
        "extracted from each PDF and checked by finance rules; documents that failed a check are in the "
        "exception queue. Every reply names the run log it came from."
    ),
)


@server.tool(annotations=READ_ONLY, structured_output=True)
def list_exceptions() -> dict[str, Any]:
    """List every document in the exception queue of the latest run: its doc_id, document type, each
    failed check (rule, reason, expected vs. actual, difference, tolerance), the extractor's confidence,
    and any self-correction retry history (proposed changes, quotes, accepted or rejected)."""
    return store.list_exceptions(store.load())


@server.tool(annotations=READ_ONLY, structured_output=True)
def get_document(doc_id: str) -> dict[str, Any]:
    """Return one document from the latest run: how it was routed, every extracted field, its validation
    status with failed checks, and its retry log. doc_id looks like 'doc_021' (or 'hard_004' for the
    hard split); 'doc_021.pdf' also works."""
    return store.get_document(store.load(), doc_id)


@server.tool(annotations=READ_ONLY, structured_output=True)
def get_fund_summary(fund_name: str) -> dict[str, Any]:
    """Summarise one fund across its documents in the latest run: the latest capital account per investor,
    the latest schedule of investments (total, largest holdings), the latest investor letter (return,
    exposures), and which of its documents need review. fund_name may be partial or misspelled
    ('granite peak', 'Bluewatr Infrastructure'); the reply says how it was matched."""
    return store.get_fund_summary(store.load(), fund_name)


def main() -> None:
    server.run()  # stdio


if __name__ == "__main__":
    main()
