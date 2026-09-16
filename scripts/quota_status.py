#!/usr/bin/env python3
"""
Compact quota status output for status bars.

Usage:
    python scripts/quota_status.py

Output:
    QUOTA Claude: 80% Gemini: 100%
"""
import os
import sys
from pathlib import Path

# Get the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Change to project root so config files are found
os.chdir(PROJECT_ROOT)

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxy_app.quota_viewer import run_quota_compact_status

if __name__ == "__main__":
    run_quota_compact_status()
