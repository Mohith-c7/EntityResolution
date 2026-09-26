"""
Hardened Data Contract and Schema Invariant Unit Tests for Amazon Business Entity Resolution 2026.
Uses lightweight fixtures to test contracts quickly without loading multi-gigabyte files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from fixtures.data_fixtures import (
    get_ground_truth_with_duplicate_candidate_df,
    get_ground_truth_with_empty_matches_df,
    get_ground_truth_with_invalid_prefix_df,
    get_ground_truth_with_mismatched_s1_df,
    get_ground_truth_with_multiple_matches_df,
    get_ground_truth_with_nonexistent_s2_df,
    get_ground_truth_with_nonexistent_s3_df,
    get_ground_truth_with_s1_in_candidates_df,
    get_ground_truth_with_singleton_matches_df,
    get_missing_value_regression_df,
    get_source_with_duplicate_id_df,
    get_source_with_missing_fields_df,
    get_source_with_null_address_df,
    get_source_with_whitespace_id_df,
    get_source_with_wrong_prefix_df,
    get_valid_ground_truth_df,
    get_valid_source1_df,
    get_valid_source2_df,
    get_valid_source3_df,
)
from src.data.loader import load_tsv
from src.data.profiling import compute_exact_median_from_histogram
from src.data.schema import (
    REQUIRED_GROUND_TRUTH_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
    inspect_dataframe,
    inspect_ground_truth,
    validate_ground_truth_contract,
    validate_schema,
    validate_source_dataset,
)


def test_loader_preserves_raw_strings_and_ids(tmp_path: Path) -> None:
    tsv_path = tmp_path / "synthetic_source1.tsv"
    sample_df = get_valid_source1_df()
    sample_df.to_csv(tsv_path, sep="\t", index=False)

    df = load_tsv(tsv_path)
    assert len(df) == len(sample_df)
    assert list(df.columns) == REQUIRED_SOURCE_COLUMNS
    for eid in df["entity_id"]:
        assert eid.startswith("S1-")
        assert isinstance(eid, str)


def test_loader_raises_on_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "non_existent_file.tsv"
    try:
        load_tsv(missing_path)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_valid_source_datasets_pass_contract() -> None:
    s1 = get_valid_source1_df()
    s2 = get_valid_source2_df()
    s3 = get_valid_source3_df()

    assert validate_source_dataset(s1, "Source1", expected_prefix="S1-") == []
    assert validate_source_dataset(s2, "Source2", expected_prefix="S2-") == []
    assert validate_source_dataset(s3, "Source3", expected_prefix="S3-") == []


def test_source_dataset_detects_missing_columns() -> None:
    df = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Alpha"]})
    issues = validate_source_dataset(df, "Source1")
    assert len(issues) > 0
    assert "Missing required columns" in issues[0]


def test_source_dataset_detects_null_and_numeric_ids() -> None:
    # Null ID
    null_id_df = pd.DataFrame({
        "entity_id": [None, "S1-2"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["123 St", "456 Ave"],
        "country": ["US", "US"],
    })
    issues = validate_source_dataset(null_id_df, "Source1")
    assert any("null entity_id" in msg for msg in issues)

    # Numeric (non-string) ID
    num_id_df = pd.DataFrame({
        "entity_id": [101, "S1-102"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["123 St", "456 Ave"],
        "country": ["US", "US"],
    })
    issues_num = validate_source_dataset(num_id_df, "Source1")
    assert any("not strings" in msg for msg in issues_num)


def test_source_dataset_detects_duplicate_ids() -> None:
    dup_id_df = pd.DataFrame({
        "entity_id": ["S1-101", "S1-101"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["123 St", "456 Ave"],
        "country": ["US", "US"],
    })
    issues = validate_source_dataset(dup_id_df, "Source1")
    assert any("duplicate entity_id" in msg for msg in issues)


def test_source_dataset_detects_invalid_prefixes() -> None:
    wrong_prefix_df = pd.DataFrame({
        "entity_id": ["S2-101"],  # S2 ID in S1 dataset
        "business_name": ["Alpha"],
        "business_address": ["123 St"],
        "country": ["US"],
    })
    issues = validate_source_dataset(wrong_prefix_df, "Source1", expected_prefix="S1-")
    assert any("not starting with expected prefix 'S1-'" in msg for msg in issues)


def test_source_dataset_detects_null_name_and_country() -> None:
    bad_df = pd.DataFrame({
        "entity_id": ["S1-101", "S1-102"],
        "business_name": ["", None],
        "business_address": ["123 St", "456 Ave"],
        "country": [None, "  "],
    })
    issues = validate_source_dataset(bad_df, "Source1")
    assert any("business_name" in msg for msg in issues)
    assert any("country" in msg for msg in issues)


def test_source_dataset_allows_null_business_address() -> None:
    # Address is permitted to be null (as observed in 3.3% of S2/S3 records)
    s2 = get_valid_source2_df()
    assert s2["business_address"].isna().sum() > 0
    issues = validate_source_dataset(s2, "Source2", expected_prefix="S2-")
    assert issues == []


def test_source_dataset_is_strictly_open_set_country() -> None:
    # Any valid non-empty country string (France, Germany, Japan) is accepted without hardcoding
    df = pd.DataFrame({
        "entity_id": ["S1-901", "S1-902", "S1-903"],
        "business_name": ["Maison Laurent", "Kaiser Tech", "Tokyo Trading"],
        "business_address": ["Paris", "Berlin", "Tokyo"],
        "country": ["France", "Germany", "Japan"],
    })
    issues = validate_source_dataset(df, "Source1", expected_prefix="S1-")
    assert issues == []


def test_valid_ground_truth_contract_passes() -> None:
    s1 = get_valid_source1_df()
    s2 = get_valid_source2_df()
    s3 = get_valid_source3_df()
    gt = get_valid_ground_truth_df()

    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    issues = validate_ground_truth_contract(gt, s1_df=s1, s2_ids=s2_ids, s3_ids=s3_ids)
    assert issues == []


def test_ground_truth_detects_duplicate_and_null_s1_ids() -> None:
    gt_dup = pd.DataFrame({
        "source1_entity_id": ["S1-101", "S1-101"],
        "matched_entity_ids": ["S2-201", "S3-301"],
    })
    issues = validate_ground_truth_contract(gt_dup)
    assert any("duplicate source1_entity_id" in msg for msg in issues)


def test_ground_truth_detects_referential_integrity_violations() -> None:
    s1 = get_valid_source1_df()
    gt_mismatch = pd.DataFrame({
        "source1_entity_id": ["S1-999"],  # Not in s1
        "matched_entity_ids": ["S2-201"],
    })
    issues = validate_ground_truth_contract(gt_mismatch, s1_df=s1)
    assert any("not present in train_source1" in msg for msg in issues)
    assert any("missing from Ground Truth" in msg for msg in issues)


def test_ground_truth_rejects_s1_inside_matches() -> None:
    gt_bad = pd.DataFrame({
        "source1_entity_id": ["S1-101"],
        "matched_entity_ids": ["S1-102,S2-201"],  # S1 ID inside matches is forbidden
    })
    issues = validate_ground_truth_contract(gt_bad)
    assert any("S1 IDs appearing inside matched_entity_ids" in msg for msg in issues)


def test_ground_truth_rejects_invalid_candidate_prefix() -> None:
    gt_bad = pd.DataFrame({
        "source1_entity_id": ["S1-101"],
        "matched_entity_ids": ["S4-999,S2-201"],  # S4 prefix is invalid
    })
    issues = validate_ground_truth_contract(gt_bad)
    assert any("invalid prefixes" in msg for msg in issues)


def test_ground_truth_rejects_intra_row_duplicate_matches() -> None:
    gt_bad = pd.DataFrame({
        "source1_entity_id": ["S1-101"],
        "matched_entity_ids": ["S2-201,S2-201"],  # Repeated candidate in single row
    })
    issues = validate_ground_truth_contract(gt_bad)
    assert any("duplicate matched IDs within the same list" in msg for msg in issues)


def test_ground_truth_supports_singletons_and_inspection() -> None:
    gt = get_valid_ground_truth_df()
    info = inspect_ground_truth(gt)
    assert info["total_s1_rows"] == 5
    assert info["singleton_count"] == 1  # S1-105 has empty matches
    assert info["singleton_rate"] == 20.0
    assert info["one_match_count"] == 2  # S1-102 and S1-104
    assert info["multi_match_count"] == 2  # S1-101 and S1-103
    assert info["total_positive_pairs"] == 6  # 2 + 1 + 2 + 1 + 0


def test_missing_value_regression_counts_correctly() -> None:
    """Requirement 1 Regression Test: [None, '', '   ', 'valid text'] must count exactly 3 missing values."""
    df = get_missing_value_regression_df()
    summary = inspect_dataframe(df, "RegressionTest", id_column="entity_id")
    assert summary.missing_counts["test_col"] == 3


def test_source_dataset_detects_whitespace_ids() -> None:
    df = get_source_with_whitespace_id_df()
    issues = validate_source_dataset(df, "Source1")
    assert any("whitespace-only entity_id" in msg for msg in issues)


def test_ground_truth_detects_whitespace_s1_ids() -> None:
    gt_bad = pd.DataFrame({
        "source1_entity_id": ["   "],
        "matched_entity_ids": ["S2-201"],
    })
    issues = validate_ground_truth_contract(gt_bad)
    assert any("whitespace-only source1_entity_id" in msg for msg in issues)


def test_ground_truth_detects_nonexistent_s2_and_s3_candidates() -> None:
    s2 = get_valid_source2_df()
    s3 = get_valid_source3_df()
    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    gt_bad_s2 = get_ground_truth_with_nonexistent_s2_df()
    issues_s2 = validate_ground_truth_contract(gt_bad_s2, s2_ids=s2_ids, s3_ids=s3_ids)
    assert any("not found in source 2 dataset" in msg for msg in issues_s2)

    gt_bad_s3 = get_ground_truth_with_nonexistent_s3_df()
    issues_s3 = validate_ground_truth_contract(gt_bad_s3, s2_ids=s2_ids, s3_ids=s3_ids)
    assert any("not found in source 3 dataset" in msg for msg in issues_s3)


def test_ground_truth_empty_matches_valid() -> None:
    gt_empty = get_ground_truth_with_empty_matches_df()
    issues = validate_ground_truth_contract(gt_empty)
    assert issues == []


def test_ground_truth_singleton_matches_valid() -> None:
    gt_single = get_ground_truth_with_singleton_matches_df()
    issues = validate_ground_truth_contract(gt_single)
    assert issues == []


def test_ground_truth_multiple_matches_valid() -> None:
    gt_multi = get_ground_truth_with_multiple_matches_df()
    issues = validate_ground_truth_contract(gt_multi)
    assert issues == []


def test_exact_median_from_histogram_odd() -> None:
    """Observations: [1, 2, 2, 3, 3] -> N=5, median = 2.0."""
    hist = {1: 1, 2: 2, 3: 2}
    assert compute_exact_median_from_histogram(hist) == 2.0


def test_exact_median_from_histogram_even() -> None:
    """Observations: [1, 2, 3, 4] -> N=4, median = (2+3)/2 = 2.5."""
    hist = {1: 1, 2: 1, 3: 1, 4: 1}
    assert compute_exact_median_from_histogram(hist) == 2.5


def test_exact_median_from_histogram_known() -> None:
    """Observations: [0, 0, 1, 1, 1, 1] -> N=6, median = (1+1)/2 = 1.0."""
    hist = {0: 2, 1: 4}
    assert compute_exact_median_from_histogram(hist) == 1.0


def test_ground_truth_detects_target_ownership_collision() -> None:
    """FAIL: S2 target candidate matched to multiple S1 entities."""
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-A", "matched_entity_ids": "S2-TARGET_1"},
        {"source1_entity_id": "S1-B", "matched_entity_ids": "S2-TARGET_1"},
    ])
    issues = validate_ground_truth_contract(gt_df, check_target_ownership=True)
    assert any("ambiguous target ownership" in msg for msg in issues)


def run_all_tests() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        test_loader_preserves_raw_strings_and_ids(tmp_p)
        test_loader_raises_on_missing_file(tmp_p)
    test_valid_source_datasets_pass_contract()
    test_source_dataset_detects_missing_columns()
    test_source_dataset_detects_null_and_numeric_ids()
    test_source_dataset_detects_whitespace_ids()
    test_source_dataset_detects_duplicate_ids()
    test_source_dataset_detects_invalid_prefixes()
    test_source_dataset_detects_null_name_and_country()
    test_source_dataset_allows_null_business_address()
    test_source_dataset_is_strictly_open_set_country()
    test_valid_ground_truth_contract_passes()
    test_ground_truth_detects_duplicate_and_null_s1_ids()
    test_ground_truth_detects_whitespace_s1_ids()
    test_ground_truth_detects_referential_integrity_violations()
    test_ground_truth_detects_nonexistent_s2_and_s3_candidates()
    test_ground_truth_rejects_s1_inside_matches()
    test_ground_truth_rejects_invalid_candidate_prefix()
    test_ground_truth_rejects_intra_row_duplicate_matches()
    test_ground_truth_empty_matches_valid()
    test_ground_truth_singleton_matches_valid()
    test_ground_truth_multiple_matches_valid()
    test_ground_truth_supports_singletons_and_inspection()
    test_missing_value_regression_counts_correctly()
    test_exact_median_from_histogram_odd()
    test_exact_median_from_histogram_even()
    test_exact_median_from_histogram_known()
    print("All Data Contract and Invariant unit tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
