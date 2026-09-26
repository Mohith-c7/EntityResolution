"""
Runner script for Stage 0 EDA and Data Profiling.
Orchestrates streaming profiling via src.data.profiling and saves reports/eda/eda_profile_results.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project code package is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.data.profiling import (
    find_ground_truth_examples,
    print_profiling_summary,
    profile_ground_truth,
    profile_source_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logger.info("Starting Stage 0 EDA profiling runner...")

    results: dict[str, Any] = {
        "datasets": {},
        "ground_truth": {},
        "examples": [],
    }

    # 1. Profile Ground Truth
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    if gt_path.is_file():
        results["ground_truth"] = profile_ground_truth(gt_path)

    # 2. Profile Source Datasets
    sources = [
        (TRAIN_DIR / "train_source1.tsv", "Train S1"),
        (TRAIN_DIR / "train_source2.tsv", "Train S2"),
        (TRAIN_DIR / "train_source3.tsv", "Train S3"),
        (TEST_DIR / "test_source1.tsv", "Test S1"),
        (TEST_DIR / "test_source2.tsv", "Test S2"),
        (TEST_DIR / "test_source3.tsv", "Test S3"),
    ]

    for f_path, d_name in sources:
        if f_path.is_file():
            results["datasets"][d_name] = profile_source_dataset(f_path, d_name)

    # 3. Ground Truth Matching Examples
    if gt_path.is_file():
        results["examples"] = find_ground_truth_examples(TRAIN_DIR, n_examples=15)

    # Save raw JSON results
    out_json = REPORTS_DIR / "eda_profile_results.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Profiling complete. JSON saved to %s", out_json)
    print_profiling_summary(results)


if __name__ == "__main__":
    main()
