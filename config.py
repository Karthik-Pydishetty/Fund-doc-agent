"""Project-wide settings. The model is chosen here and nowhere else."""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("FUND_AGENT_MODEL", "claude-sonnet-5-5")

# List prices, USD per million tokens (input, output). Used only to report run cost.
PRICING_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00),
}
