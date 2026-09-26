"""Generate matching_results.tsv and the exact candidate set scored offline."""

import argparse
from pathlib import Path

from src.pipeline.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--index-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--reference-limit", type=int, default=0, help="Smoke test only; positive limits do not produce a complete submission.")
    args = parser.parse_args()
    run_inference(args.test_dir, args.index_dir, args.artifact_dir, args.output_dir,
        workers=args.workers, batch_size=args.batch_size, reference_limit=args.reference_limit)


if __name__ == "__main__":
    main()
