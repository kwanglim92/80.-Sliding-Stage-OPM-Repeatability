"""Sliding Stage OPM Repeatability Analyzer — Entry Point."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ui.main_window import run_app

if __name__ == "__main__":
    run_app()
