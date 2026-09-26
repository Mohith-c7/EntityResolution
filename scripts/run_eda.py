"""
Runner script for Stage 0 EDA, Data Profiling, and Executable Contract Verification.
Orchestrates streaming profiling via src.data.profiling and saves reports/eda/eda_profile_results.json.
Enforces strict file presence (unless --partial is given) and executes all data contracts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project code package is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.data.loader import load_tsv
from src.data.profiling import (
    find_ground_truth_examples,
    print_profiling_summary,
    profile_ground_truth,
    profile_source_dataset,
)
from src.data.schema import (
    validate_ground_truth_contract,
    validate_source_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 0 EDA profiling and contract validation.")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Allow partial inspection if some dataset files are missing. Default is strict (all 7 required).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Stage 0 EDA profiling & contract validation runner (partial=%s)...", args.partial)

    required_files = [
        TRAIN_DIR / "train_ground_truth.tsv",
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    missing_files = [p for p in required_files if not p.is_file()]
    if missing_files and not args.partial:
        raise FileNotFoundError(
            f"Official full-data run requires all 7 files, but {len(missing_files)} file(s) are missing:\n"
            + "\n".join(f"  - {p}" for p in missing_files)
            + "\nPass --partial if you intend to run in partial mode."
        )

    results: dict[str, Any] = {
        "datasets": {},
        "ground_truth": {},
        "examples": [],
        "contract_validation": {
            "all_passed": True,
            "issues": [],
            "source_issues": {},
            "ground_truth_issues": [],
            "disjointness_issues": [],
        },
    }

    # 1. Profile Source Datasets & validate schema
    sources = [
        (TRAIN_DIR / "train_source1.tsv", "Train S1", "S1-"),
        (TRAIN_DIR / "train_source2.tsv", "Train S2", "S2-"),
        (TRAIN_DIR / "train_source3.tsv", "Train S3", "S3-"),
        (TEST_DIR / "test_source1.tsv", "Test S1", "S1-"),
        (TEST_DIR / "test_source2.tsv", "Test S2", "S2-"),
        (TEST_DIR / "test_source3.tsv", "Test S3", "S3-"),
    ]

    id_sets: dict[str, set[str]] = {}

    for f_path, d_name, prefix in sources:
        if f_path.is_file():
            logger.info("Profiling source dataset: %s (%s)", d_name, f_path.name)
            results["datasets"][d_name] = profile_source_dataset(f_path, d_name)

            # Contract check on source dataset
            logger.info("Validating source dataset contract: %s", d_name)
            # Sample 200,000 rows for deep per-row schema contract checks to conserve RAM
            source_sample = load_tsv(f_path, nrows=200000)
            src_issues = validate_source_dataset(source_sample, d_name, expected_prefix=prefix)
            if src_issues:
                results["contract_validation"]["source_issues"][d_name] = src_issues
                results["contract_validation"]["issues"].extend(src_issues)
                results["contract_validation"]["all_passed"] = False

            # Load full ID set memory-consciously for cross-file referential integrity
            id_sets[d_name] = set(load_tsv(f_path, usecols=["entity_id"])["entity_id"])

    # 2. Check train / test ID disjointness
    for src in ["S1", "S2", "S3"]:
        train_k = f"Train {src}"
        test_k = f"Test {src}"
        if train_k in id_sets and test_k in id_sets:
            overlap = id_sets[train_k] & id_sets[test_k]
            if overlap:
                msg = f"Train/Test ID leakage detected in {src}: {len(overlap)} overlapping entity IDs"
                results["contract_validation"]["disjointness_issues"].append(msg)
                results["contract_validation"]["issues"].append(msg)
                results["contract_validation"]["all_passed"] = False
            else:
                logger.info("Verified Train/Test ID disjointness in %s: 0 overlapping IDs.", src)

    # 3. Profile Ground Truth & validate Ground Truth contract
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    if gt_path.is_file():
        logger.info("Profiling Ground Truth: %s", gt_path.name)
        results["ground_truth"] = profile_ground_truth(gt_path)

        logger.info("Validating Ground Truth contract (including cross-S1 target ownership exclusivity)...")
        # Load ground truth
        gt_df = load_tsv(gt_path)
        s1_ids = id_sets.get("Train S1", set())
        s2_ids = id_sets.get("Train S2", set())
        s3_ids = id_sets.get("Train S3", set())

        gt_issues = validate_ground_truth_contract(
            gt_df,
            s1_df=s1_ids,
            s2_ids=s2_ids,
            s3_ids=s3_ids,
            check_target_ownership=True,
        )
        if gt_issues:
            results["contract_validation"]["ground_truth_issues"] = gt_issues
            results["contract_validation"]["issues"].extend(gt_issues)
            results["contract_validation"]["all_passed"] = False
        else:
            logger.info("Verified Ground Truth contract: 100% passed with zero issues.")

        # Extract Ground Truth examples
        logger.info("Extracting Ground Truth matching examples...")
        results["examples"] = find_ground_truth_examples(TRAIN_DIR, n_examples=15)

    # Save raw JSON results
    out_json = REPORTS_DIR / "eda_profile_results.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Profiling & Contract Verification complete. JSON saved to %s", out_json)
    print_profiling_summary(results)

    if results["contract_validation"]["all_passed"]:
        logger.info("ALL DATA CONTRACTS AND CROSS-FILE INVARIANTS: 100% PASSED.")
    else:
        logger.warning(
            "Contract validation reported %d issue(s).",
            len(results["contract_validation"]["issues"]),
        )


if __name__ == "__main__":
    main()
