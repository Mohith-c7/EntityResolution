"""
Submission validation script for Amazon Business Entity Resolution 2026.
Validates schemas, formatting, candidate freezing, ID coverage, duplicate rows,
and cross-file subset invariants prior to submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def load_test_ids(test_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """
    Load test entity IDs memory-consciously using usecols.
    """
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    if not s1_path.is_file():
        raise FileNotFoundError(f"Missing test source 1 file: {s1_path}")
    if not s2_path.is_file():
        raise FileNotFoundError(f"Missing test source 2 file: {s2_path}")
    if not s3_path.is_file():
        raise FileNotFoundError(f"Missing test source 3 file: {s3_path}")

    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)
    s2_df = pd.read_csv(s2_path, sep="\t", usecols=["entity_id"], dtype=str)
    s3_df = pd.read_csv(s3_path, sep="\t", usecols=["entity_id"], dtype=str)

    s1_ids = set(s1_df["entity_id"])
    s2_ids = set(s2_df["entity_id"])
    s3_ids = set(s3_df["entity_id"])

    return s1_ids, s2_ids, s3_ids


def validate_submission(
    matching_path: Path | str,
    candidate_path: Path | str,
    test_dir: Path | str | None = None,
    test_s1_ids: set[str] | None = None,
    test_s2_ids: set[str] | None = None,
    test_s3_ids: set[str] | None = None,
) -> list[str]:
    """
    Validate submission files matching_results.tsv and candidate_pairs.tsv.

    Args:
        matching_path: Path to matching_results.tsv
        candidate_path: Path to candidate_pairs.tsv
        test_dir: Path to test dataset directory (containing test_source*.tsv)
        test_s1_ids: Optional pre-loaded set of test S1 IDs (for testing/offline use)
        test_s2_ids: Optional pre-loaded set of test S2 IDs (for testing/offline use)
        test_s3_ids: Optional pre-loaded set of test S3 IDs (for testing/offline use)

    Returns:
        List of error strings (empty if submission is completely valid).
    """
    errors: list[str] = []

    m_path = Path(matching_path)
    c_path = Path(candidate_path)

    # 1. File existence
    if not m_path.is_file():
        errors.append(f"Matching file not found: {m_path}")
    if not c_path.is_file():
        errors.append(f"Candidate file not found: {c_path}")

    if errors:
        return errors

    # Load test IDs if needed
    if test_s1_ids is None or test_s2_ids is None or test_s3_ids is None:
        if test_dir is not None:
            t_dir = Path(test_dir)
            try:
                t_s1, t_s2, t_s3 = load_test_ids(t_dir)
                test_s1_ids = test_s1_ids or t_s1
                test_s2_ids = test_s2_ids or t_s2
                test_s3_ids = test_s3_ids or t_s3
            except Exception as e:
                errors.append(f"Failed to load test IDs from {t_dir}: {e}")
                return errors
        else:
            errors.append("No test IDs or test_dir provided for validation.")
            return errors

    # 2. Validate matching_results.tsv
    try:
        m_df = pd.read_csv(m_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        errors.append(f"Failed to parse matching_results.tsv as TSV: {e}")
        return errors

    if list(m_df.columns) != MATCHING_COLUMNS:
        errors.append(
            f"Invalid columns in matching_results.tsv. Expected {MATCHING_COLUMNS}, found {list(m_df.columns)}"
        )
        return errors

    m_s1_series = m_df["source1_entity_id"]
    m_s1_set = set(m_s1_series)

    # Duplicate S1 rows in matching
    if len(m_s1_series) != len(m_s1_set):
        dup_count = len(m_s1_series) - len(m_s1_set)
        errors.append(f"Found {dup_count:,} duplicate S1 rows in matching_results.tsv")

    # S1 ID coverage check
    missing_m_s1 = test_s1_ids - m_s1_set
    extra_m_s1 = m_s1_set - test_s1_ids
    if missing_m_s1:
        errors.append(f"matching_results.tsv is missing {len(missing_m_s1):,} test S1 entities")
    if extra_m_s1:
        errors.append(f"matching_results.tsv contains {len(extra_m_s1):,} S1 IDs not in test_source1")

    # Parse match lists
    matching_map: dict[str, list[str]] = {}
    for _, row in m_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_m = str(row["matched_entity_ids"]).strip()
        if not raw_m:
            cands = []
        else:
            cands = [c.strip() for c in raw_m.split(",") if c.strip()]

        # Check duplicate matches within list
        if len(cands) != len(set(cands)):
            errors.append(f"Duplicate candidate IDs found in match list for {s1_id}")

        for cid in cands:
            if cid.startswith("S1-"):
                errors.append(f"Forbidden S1 ID '{cid}' found inside matched_entity_ids for {s1_id}")
            elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                errors.append(f"Invalid candidate prefix for '{cid}' in match list for {s1_id} (must be S2- or S3-)")
            else:
                if cid.startswith("S2-") and test_s2_ids and cid not in test_s2_ids:
                    errors.append(f"Candidate '{cid}' in match list does not exist in test Source 2")
                elif cid.startswith("S3-") and test_s3_ids and cid not in test_s3_ids:
                    errors.append(f"Candidate '{cid}' in match list does not exist in test Source 3")

        matching_map[s1_id] = cands

    # 3. Validate candidate_pairs.tsv
    try:
        c_df = pd.read_csv(c_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        errors.append(f"Failed to parse candidate_pairs.tsv as TSV: {e}")
        return errors

    if list(c_df.columns) != CANDIDATE_COLUMNS:
        errors.append(
            f"Invalid columns in candidate_pairs.tsv. Expected {CANDIDATE_COLUMNS}, found {list(c_df.columns)}"
        )
        return errors

    c_s1_series = c_df["source1_entity_id"]
    c_s1_set = set(c_s1_series)

    # Duplicate S1 rows in candidate
    if len(c_s1_series) != len(c_s1_set):
        dup_count = len(c_s1_series) - len(c_s1_set)
        errors.append(f"Found {dup_count:,} duplicate S1 rows in candidate_pairs.tsv")

    # S1 ID coverage check
    missing_c_s1 = test_s1_ids - c_s1_set
    extra_c_s1 = c_s1_set - test_s1_ids
    if missing_c_s1:
        errors.append(f"candidate_pairs.tsv is missing {len(missing_c_s1):,} test S1 entities")
    if extra_c_s1:
        errors.append(f"candidate_pairs.tsv contains {len(extra_c_s1):,} S1 IDs not in test_source1")

    # Parse candidate lists
    candidate_map: dict[str, set[str]] = {}
    for _, row in c_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_c = str(row["candidate_entity_ids"]).strip()
        if not raw_c:
            cands = []
        else:
            cands = [c.strip() for c in raw_c.split(",") if c.strip()]

        # Check duplicate candidates within list
        if len(cands) != len(set(cands)):
            errors.append(f"Duplicate candidate IDs found in candidate list for {s1_id}")

        for cid in cands:
            if cid.startswith("S1-"):
                errors.append(f"Forbidden S1 ID '{cid}' found inside candidate_entity_ids for {s1_id}")
            elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                errors.append(f"Invalid candidate prefix for '{cid}' in candidate list for {s1_id} (must be S2- or S3-)")
            else:
                if cid.startswith("S2-") and test_s2_ids and cid not in test_s2_ids:
                    errors.append(f"Candidate '{cid}' in candidate list does not exist in test Source 2")
                elif cid.startswith("S3-") and test_s3_ids and cid not in test_s3_ids:
                    errors.append(f"Candidate '{cid}' in candidate list does not exist in test Source 3")

        candidate_map[s1_id] = set(cands)

    # 4. Cross-file subset invariant: final_matches(S1) ⊆ candidate_pairs(S1)
    for s1_id, matches in matching_map.items():
        cands_set = candidate_map.get(s1_id, set())
        for mid in matches:
            if mid not in cands_set:
                errors.append(
                    f"Match '{mid}' for entity {s1_id} is not in the frozen candidate pool (violates subset invariant)"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission outputs for Business Entity Resolution.")
    parser.add_argument("--matching", required=True, help="Path to output/matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")

    args = parser.parse_args()

    errors = validate_submission(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
    )

    if errors:
        print(f"FAILED: Found {len(errors)} submission validation errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)
    else:
        print("SUCCESS: Submission passed all validation checks with exit code 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()
