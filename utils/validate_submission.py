"""
Submission validation script for Amazon Business Entity Resolution 2026.
Validates schemas, formatting, candidate freezing, ID coverage, duplicate rows,
and cross-file subset invariants prior to submission using streaming standard library.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def load_test_ids(test_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """
    Load test entity IDs memory-consciously using standard library csv streaming.
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

    def _read_col_ids(p: Path) -> set[str]:
        ids = set()
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if "entity_id" not in (reader.fieldnames or []):
                raise ValueError(f"Missing entity_id column in {p}")
            for row in reader:
                ids.add(row["entity_id"])
        return ids

    return _read_col_ids(s1_path), _read_col_ids(s2_path), _read_col_ids(s3_path)


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

    # Helper to stream and validate file structure
    def _validate_file_stream(
        p: Path,
        expected_cols: list[str],
        file_label: str,
        is_candidate: bool,
    ) -> tuple[set[str], dict[str, list[str]]]:
        seen_s1: set[str] = set()
        data_map: dict[str, list[str]] = {}

        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header_line = f.readline()
            if not header_line:
                errors.append(f"File {p.name} is empty.")
                return seen_s1, data_map

            header_cols = header_line.rstrip("\r\n").split("\t")
            if header_cols != expected_cols:
                errors.append(
                    f"Invalid columns in {p.name}. Expected {expected_cols}, found {header_cols}"
                )
                return seen_s1, data_map

            for line_idx, line in enumerate(f, start=2):
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue

                if "\t" not in stripped:
                    errors.append(
                        f"Malformed row (no tab separator) in {p.name} at line {line_idx}: {stripped!r}"
                    )
                    continue

                parts = stripped.split("\t")
                if len(parts) != 2:
                    errors.append(
                        f"Malformed row (expected exactly 2 fields) in {p.name} at line {line_idx}: found {len(parts)} fields"
                    )
                    continue

                s1_id, raw_cands = parts
                s1_id_clean = s1_id.strip()

                if not s1_id_clean or s1_id != s1_id_clean:
                    errors.append(f"Malformed or empty Source 1 ID in {p.name} at line {line_idx}: {s1_id!r}")

                if s1_id_clean in seen_s1:
                    errors.append(f"Found duplicate S1 rows in {p.name}: {s1_id_clean}")
                seen_s1.add(s1_id_clean)

                if not raw_cands.strip():
                    cands = []
                else:
                    tokens = raw_cands.split(",")
                    # Check for empty or malformed tokens like ",S2-A,,"
                    if any(not tok.strip() for tok in tokens) or any(tok != tok.strip() for tok in tokens):
                        errors.append(
                            f"Malformed ID list containing empty or whitespace tokens in {p.name} for {s1_id_clean}: {raw_cands!r}"
                        )
                        cands = [t.strip() for t in tokens if t.strip()]
                    else:
                        cands = tokens

                # Check duplicate candidate IDs within list
                if len(cands) != len(set(cands)):
                    list_type = "candidate" if is_candidate else "match"
                    errors.append(f"Duplicate candidate IDs found in {list_type} list for {s1_id_clean}")

                for cid in cands:
                    if cid.startswith("S1-"):
                        target_col = "candidate_entity_ids" if is_candidate else "matched_entity_ids"
                        errors.append(f"Forbidden S1 ID '{cid}' found inside {target_col} for {s1_id_clean}")
                    elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                        errors.append(f"Invalid candidate prefix for '{cid}' in {p.name} for {s1_id_clean} (must be S2- or S3-)")
                    else:
                        # Unconditional membership check (strict check regardless of whether set is empty)
                        if cid.startswith("S2-") and test_s2_ids is not None and cid not in test_s2_ids:
                            errors.append(f"Candidate '{cid}' does not exist in test Source 2 ({p.name})")
                        elif cid.startswith("S3-") and test_s3_ids is not None and cid not in test_s3_ids:
                            errors.append(f"Candidate '{cid}' does not exist in test Source 3 ({p.name})")

                data_map[s1_id_clean] = cands

        return seen_s1, data_map

    # 2. Validate matching_results.tsv
    m_s1_set, matching_map = _validate_file_stream(
        m_path, MATCHING_COLUMNS, "matching_results.tsv", is_candidate=False
    )

    # Coverage check for matching
    missing_m_s1 = test_s1_ids - m_s1_set
    extra_m_s1 = m_s1_set - test_s1_ids
    if missing_m_s1:
        errors.append(f"matching_results.tsv is missing {len(missing_m_s1):,} test S1 entities")
    if extra_m_s1:
        errors.append(f"matching_results.tsv contains {len(extra_m_s1):,} S1 IDs not in test_source1")

    # 3. Validate candidate_pairs.tsv
    c_s1_set, candidate_map = _validate_file_stream(
        c_path, CANDIDATE_COLUMNS, "candidate_pairs.tsv", is_candidate=True
    )

    # Coverage check for candidates
    missing_c_s1 = test_s1_ids - c_s1_set
    extra_c_s1 = c_s1_set - test_s1_ids
    if missing_c_s1:
        errors.append(f"candidate_pairs.tsv is missing {len(missing_c_s1):,} test S1 entities")
    if extra_c_s1:
        errors.append(f"candidate_pairs.tsv contains {len(extra_c_s1):,} S1 IDs not in test_source1")

    # 4. Check candidate subset invariant (final matches must be subset of candidates)
    for s1_id, match_cands in matching_map.items():
        if match_cands:
            cand_pool = set(candidate_map.get(s1_id, []))
            invalid_matches = set(match_cands) - cand_pool
            if invalid_matches:
                errors.append(
                    f"Final match is not in candidate pool for {s1_id}; violates subset invariant"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching", "-m", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", required=True, help="Path to test dataset directory")
    args = parser.parse_args()

    errors = validate_submission(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
    )

    if errors:
        print(f"FAILED: Found {len(errors)} validation errors:")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        return 1

    print("SUCCESS: Submission bundle is 100% valid and verified against all invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
