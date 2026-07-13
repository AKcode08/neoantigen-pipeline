"""Configuration and environment variables."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# API Keys
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NETMHCPAN_PATH = os.getenv("NETMHCPAN_PATH", "")
ALPHAFOLD_PATH = os.getenv("ALPHAFOLD_PATH", "")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
