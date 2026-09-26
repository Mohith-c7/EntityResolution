"""Local stdlib validator; use the organizer helper too when available."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.pipeline.export import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()
    errors = validate_submission(args.matching, args.candidate, args.test_dir)
    if errors:
        for number, error in enumerate(errors[:20], 1):
            print(f"{number}. {error}")
        if len(errors) > 20:
            print(f"{len(errors)-20} further issues")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
