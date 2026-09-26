"""
Data contract and loader validation tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from src.data.loader import (  # noqa: E402
    load_ground_truth,
    load_test_source1,
    load_train_source1,
    load_tsv,
)
from src.data.schema import (  # noqa: E402
    REQUIRED_GROUND_TRUTH_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
    inspect_dataframe,
    inspect_ground_truth,
    validate_schema,
)


def test_loader_preserves_raw_strings_and_ids() -> None:
    df = load_train_source1(nrows=10)
    assert len(df) == 10
    assert list(df.columns) == REQUIRED_SOURCE_COLUMNS
    for eid in df["entity_id"]:
        assert eid.startswith("S1-")
        assert isinstance(eid, str)


def test_ground_truth_loader_contract() -> None:
    gt = load_ground_truth(nrows=20)
    assert len(gt) == 20
    assert list(gt.columns) == REQUIRED_GROUND_TRUTH_COLUMNS
    info = inspect_ground_truth(gt)
    assert info["total_s1_rows"] == 20
    assert "mean_matches_per_s1" in info
    assert "singleton_rate" in info


def test_validate_schema_identifies_defects() -> None:
    # Clean df
    clean_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["123 St", "456 Ave"],
        "country": ["US", "India"],
    })
    assert validate_schema(clean_df, "Clean") == []

    # Missing column
    bad_cols_df = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Alpha"]})
    issues = validate_schema(bad_cols_df, "BadCols")
    assert len(issues) > 0
    assert "Missing required columns" in issues[0]

    # Duplicate IDs
    dup_id_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-1"],
        "business_name": ["Alpha", "Beta"],
        "business_address": ["123 St", "456 Ave"],
        "country": ["US", "India"],
    })
    dup_issues = validate_schema(dup_id_df, "DupIDs")
    assert any("duplicate ID" in msg for msg in dup_issues)


def test_test_dataset_schema_contract() -> None:
    test_s1 = load_test_source1(nrows=10)
    assert len(test_s1) == 10
    assert list(test_s1.columns) == REQUIRED_SOURCE_COLUMNS
    issues = validate_schema(test_s1, "TestS1")
    assert issues == []


if __name__ == "__main__":
    test_loader_preserves_raw_strings_and_ids()
    test_ground_truth_loader_contract()
    test_validate_schema_identifies_defects()
    test_test_dataset_schema_contract()
    print("All data contract tests passed successfully!")
