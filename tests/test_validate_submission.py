"""
Unit tests for utils/validate_submission.py using lightweight synthetic fixtures.
Runs 100% offline without requiring the real multi-gigabyte dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.validate_submission import validate_submission


@pytest.fixture
def valid_submission_bundle(tmp_path: Path):
    """Create a valid synthetic submission bundle in tmp_path."""
    test_s1_ids = {"S1-1001", "S1-1002", "S1-1003", "S1-1004"}
    test_s2_ids = {"S2-2001", "S2-2002", "S2-2003"}
    test_s3_ids = {"S3-3001", "S3-3002"}

    # Matching: S1-1001 has multi-match, S1-1002 has single match, S1-1003 has empty match (singleton), S1-1004 has empty match
    matching_df = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001,S3-3001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": "S2-2002"},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    matching_file = tmp_path / "matching_results.tsv"
    matching_df.to_csv(matching_file, sep="\t", index=False)

    # Candidate: superset of matching
    candidate_df = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "candidate_entity_ids": "S2-2001,S2-2003,S3-3001"},
        {"source1_entity_id": "S1-1002", "candidate_entity_ids": "S2-2002,S3-3002"},
        {"source1_entity_id": "S1-1003", "candidate_entity_ids": "S2-2003"},  # Has candidates but none matched
        {"source1_entity_id": "S1-1004", "candidate_entity_ids": ""},         # Pure singleton with 0 candidates
    ])
    candidate_file = tmp_path / "candidate_pairs.tsv"
    candidate_df.to_csv(candidate_file, sep="\t", index=False)

    return {
        "matching_path": matching_file,
        "candidate_path": candidate_file,
        "test_s1_ids": test_s1_ids,
        "test_s2_ids": test_s2_ids,
        "test_s3_ids": test_s3_ids,
    }


def test_validator_passes_valid_bundle(valid_submission_bundle):
    """PASS: empty final match, singleton, multiple matches, valid candidate superset."""
    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert errors == []


def test_validator_fails_missing_matching_file(tmp_path, valid_submission_bundle):
    """FAIL: missing matching file."""
    nonexistent = tmp_path / "nonexistent_matching.tsv"
    errors = validate_submission(
        matching_path=nonexistent,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Matching file not found" in e for e in errors)


def test_validator_fails_missing_candidate_file(tmp_path, valid_submission_bundle):
    """FAIL: missing candidate file."""
    nonexistent = tmp_path / "nonexistent_candidate.tsv"
    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=nonexistent,
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Candidate file not found" in e for e in errors)


def test_validator_fails_duplicate_s1_row(tmp_path, valid_submission_bundle):
    """FAIL: duplicate S1 row in matching."""
    dup_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001"},
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    bad_matching_path = tmp_path / "dup_matching.tsv"
    dup_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("duplicate S1 rows" in e for e in errors)


def test_validator_fails_missing_s1(tmp_path, valid_submission_bundle):
    """FAIL: missing S1 entity in matching."""
    missing_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001"},
        # S1-1002 is omitted
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    bad_matching_path = tmp_path / "missing_matching.tsv"
    missing_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("missing 1 test S1 entities" in e for e in errors)


def test_validator_fails_extra_s1(tmp_path, valid_submission_bundle):
    """FAIL: extra unknown S1 entity in matching."""
    extra_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-9999", "matched_entity_ids": ""},  # Unknown S1 ID
    ])
    bad_matching_path = tmp_path / "extra_matching.tsv"
    extra_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("not in test_source1" in e for e in errors)


def test_validator_fails_invalid_target_prefix(tmp_path, valid_submission_bundle):
    """FAIL: invalid target prefix in match list."""
    bad_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S4-9999"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    bad_matching_path = tmp_path / "bad_prefix_matching.tsv"
    bad_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Invalid candidate prefix" in e for e in errors)


def test_validator_fails_s1_inside_match_list(tmp_path, valid_submission_bundle):
    """FAIL: S1 ID inside match list."""
    bad_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S1-1002"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    bad_matching_path = tmp_path / "s1_in_match.tsv"
    bad_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Forbidden S1 ID" in e for e in errors)


def test_validator_fails_duplicate_match(tmp_path, valid_submission_bundle):
    """FAIL: duplicate candidate within single match list."""
    bad_matching = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001,S2-2001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": ""},
    ])
    bad_matching_path = tmp_path / "dup_match_list.tsv"
    bad_matching.to_csv(bad_matching_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Duplicate candidate IDs found in match list" in e for e in errors)


def test_validator_fails_duplicate_candidate(tmp_path, valid_submission_bundle):
    """FAIL: duplicate candidate within single candidate list."""
    bad_candidate = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "candidate_entity_ids": "S2-2001,S2-2001"},
        {"source1_entity_id": "S1-1002", "candidate_entity_ids": "S2-2002"},
        {"source1_entity_id": "S1-1003", "candidate_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "candidate_entity_ids": ""},
    ])
    bad_candidate_path = tmp_path / "dup_candidate_list.tsv"
    bad_candidate.to_csv(bad_candidate_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=bad_candidate_path,
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Duplicate candidate IDs found in candidate list" in e for e in errors)


def test_validator_fails_nonexistent_target(tmp_path, valid_submission_bundle):
    """FAIL: candidate ID does not exist in test Source 2 or 3."""
    bad_candidate = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "candidate_entity_ids": "S2-99999"},
        {"source1_entity_id": "S1-1002", "candidate_entity_ids": ""},
        {"source1_entity_id": "S1-1003", "candidate_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "candidate_entity_ids": ""},
    ])
    bad_candidate_path = tmp_path / "nonexistent_target.tsv"
    bad_candidate.to_csv(bad_candidate_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=bad_candidate_path,
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("does not exist in test Source 2" in e for e in errors)


def test_validator_fails_subset_invariant_violation(tmp_path, valid_submission_bundle):
    """FAIL: match is not present in the candidate set."""
    # S1-1001 matches S2-2001 and S3-3001, but candidate list only contains S2-2001
    bad_candidate = pd.DataFrame([
        {"source1_entity_id": "S1-1001", "candidate_entity_ids": "S2-2001"},  # Missing S3-3001
        {"source1_entity_id": "S1-1002", "candidate_entity_ids": "S2-2002"},
        {"source1_entity_id": "S1-1003", "candidate_entity_ids": ""},
        {"source1_entity_id": "S1-1004", "candidate_entity_ids": ""},
    ])
    bad_candidate_path = tmp_path / "subset_violation.tsv"
    bad_candidate.to_csv(bad_candidate_path, sep="\t", index=False)

    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=bad_candidate_path,
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("violates subset invariant" in e for e in errors)


def test_validator_fails_row_without_tab(tmp_path, valid_submission_bundle):
    """FAIL: physical row has no tab separator."""
    bad_matching_path = tmp_path / "no_tab_matching.tsv"
    # Row 2 has S1-1001 with no tab
    bad_matching_path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1001\n", encoding="utf-8")

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("no tab separator" in e for e in errors)


def test_validator_fails_empty_tokens_in_comma_list(tmp_path, valid_submission_bundle):
    """FAIL: match list has empty tokens like ,S2-A,,"""
    bad_matching_path = tmp_path / "empty_tokens_matching.tsv"
    bad_matching_path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1001\t,S2-2001,,\nS1-1002\t\nS1-1003\t\nS1-1004\t\n",
        encoding="utf-8",
    )

    errors = validate_submission(
        matching_path=bad_matching_path,
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=valid_submission_bundle["test_s2_ids"],
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("Malformed ID list" in e for e in errors)


def test_validator_fails_unconditional_membership_with_empty_target_set(tmp_path, valid_submission_bundle):
    """FAIL: S2-ghost must be rejected even when test_s2_ids is passed as an empty set."""
    errors = validate_submission(
        matching_path=valid_submission_bundle["matching_path"],
        candidate_path=valid_submission_bundle["candidate_path"],
        test_s1_ids=valid_submission_bundle["test_s1_ids"],
        test_s2_ids=set(),  # Empty set: should NOT bypass membership check!
        test_s3_ids=valid_submission_bundle["test_s3_ids"],
    )
    assert any("does not exist in test Source 2" in e for e in errors)

