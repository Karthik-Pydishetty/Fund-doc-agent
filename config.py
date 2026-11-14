"""Project-wide settings. The model is chosen here and nowhere else."""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("FUND_AGENT_MODEL", "claude-sonnet-5-5")
