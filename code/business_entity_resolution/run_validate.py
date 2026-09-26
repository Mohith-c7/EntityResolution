"""Validate output coverage, ID lists, and exact candidate-subset rules."""

import argparse
from src.pipeline.export import validate_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching",required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--test-dir",required=True)
    args=parser.parse_args()
    errors=validate_submission(args.matching,args.candidate,args.test_dir)
    if errors:
        for number,error in enumerate(errors,1):
            print(f"{number}. {error}")
        raise SystemExit(1)
    print("PASS")


if __name__=="__main__":
    main()
