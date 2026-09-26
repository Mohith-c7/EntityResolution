"""
CLI script to generate the official shared grouped validation split.

Implements ARCHITECTURE.md Section 12 requirements:
- Deterministic seed (default: 42)
- 15% entity-disjoint validation holdout
- Strict connected-components grouping (zero target/S1 leakage)
- Stratification across country, singleton status, and match-count bucket
- Saves full manifest, lightweight metadata, and plaintext ID lists
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import pandas as pd

# Add source directory to Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "code" / "business_entity_resolution"))

from src.data.split import create_grouped_validation_split, save_split_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate official entity-disjoint validation split.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/train",
        help="Path to directory containing train TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/splits",
        help="Directory where split manifest and ID lists will be saved.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of entities to allocate to validation holdout (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratification (default: 42).",
    )
    parser.add_argument(
        "--include-distractors",
        action="store_true",
        help="If set, loads Source 2 and Source 3 IDs to partition unmatched distractors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    gt_path = data_dir / "train_ground_truth.tsv"
    s1_path = data_dir / "train_source1.tsv"

    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)
    if not s1_path.exists():
        logger.error("Source 1 file not found: %s", s1_path)
        sys.exit(1)

    logger.info("Loading ground truth from %s...", gt_path)
    t0 = time.time()
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded %d Ground Truth rows in %.2fs", len(gt_df), time.time() - t0)

    logger.info("Loading Source 1 metadata from %s...", s1_path)
    t1 = time.time()
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], dtype=str, keep_default_na=False)
    logger.info("Loaded %d Source 1 rows in %.2fs", len(s1_df), time.time() - t1)

    s2_ids = None
    s3_ids = None
    if args.include_distractors:
        s2_path = data_dir / "train_source2.tsv"
        s3_path = data_dir / "train_source3.tsv"
        if s2_path.exists():
            logger.info("Loading Source 2 IDs from %s...", s2_path)
            s2_df = pd.read_csv(s2_path, sep="\t", usecols=["entity_id"], dtype=str, keep_default_na=False)
            s2_ids = set(s2_df["entity_id"].astype(str).str.strip())
        if s3_path.exists():
            logger.info("Loading Source 3 IDs from %s...", s3_path)
            s3_df = pd.read_csv(s3_path, sep="\t", usecols=["entity_id"], dtype=str, keep_default_na=False)
            s3_ids = set(s3_df["entity_id"].astype(str).str.strip())

    logger.info("Generating entity-disjoint grouped validation split (val_fraction=%.2f, seed=%d)...", args.val_fraction, args.seed)
    t2 = time.time()
    split_result = create_grouped_validation_split(
        gt_df=gt_df,
        s1_df=s1_df,
        s2_ids=s2_ids,
        s3_ids=s3_ids,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )
    logger.info("Split generated successfully in %.2fs!", time.time() - t2)

    meta = split_result["metadata"]
    logger.info("=== Validation Split Summary ===")
    logger.info("Total Source 1 Entities: %d", meta["total_source1_entities"])
    logger.info("Train S1: %d (%.2f%%)", meta["train"]["s1_count"], meta["train"]["s1_proportion"] * 100)
    logger.info("Val S1:   %d (%.2f%%)", meta["validation"]["s1_count"], meta["validation"]["s1_proportion"] * 100)
    logger.info("Train Matched Targets: %d", meta["train"]["target_count"])
    logger.info("Val Matched Targets:   %d", meta["validation"]["target_count"])
    logger.info("Train Country Distribution: %s", meta["train"]["country_distribution"])
    logger.info("Val Country Distribution:   %s", meta["validation"]["country_distribution"])
    logger.info("Train Singleton Ratio:      %.2f%%", meta["train"]["singleton_ratio"] * 100)
    logger.info("Val Singleton Ratio:        %.2f%%", meta["validation"]["singleton_ratio"] * 100)
    logger.info("Zero Leakage Verified:      %s", meta["zero_leakage_verified"])

    logger.info("Saving split bundle to %s...", out_dir)
    saved_files = save_split_manifest(split_result, out_dir)
    for key, p in saved_files.items():
        logger.info("  %s: %s (size: %s)", key, p, f"{p.stat().st_size:,} bytes")

    logger.info("Official validation split successfully published!")


if __name__ == "__main__":
    main()
