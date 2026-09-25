"""
Pipeline entry point for Amazon Business Entity Resolution 2026.
Smoke test and project setup verification.
"""

import sys
from pathlib import Path

# Add project code directory to sys.path to enable imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.config import PROJECT_ROOT, RANDOM_SEED, TOP_K


def main() -> None:
    print("=" * 50)
    print("Amazon Business Entity Resolution Pipeline")
    print("Status: project setup complete")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Top-K Candidates: {TOP_K}")
    print("=" * 50)


if __name__ == "__main__":
    main()
