"""
Schema validation, contract enforcement, and inspection utilities for entity resolution datasets.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

VALID_SOURCE_PREFIXES = ("S1-", "S2-", "S3-")
VALID_MATCH_PREFIXES = ("S2-", "S3-")


@dataclass
class DatasetSummary:
    """Structured inspection summary for a business entity dataset."""

    name: str
    total_rows: int
    total_columns: int
    columns: list[str]
    column_types: dict[str, str]
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    unique_ids: int
    duplicate_ids: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        """Return a formatted text table of the summary."""
        lines = [
            f"Dataset: {self.name}",
            f"Rows:        {self.total_rows:,}",
            f"Columns:     {self.total_columns}",
            "",
            "Columns & Missing Values:",
        ]
        for col in self.columns:
            m_cnt = self.missing_counts.get(col, 0)
            m_pct = self.missing_percentages.get(col, 0.0)
            dtype = self.column_types.get(col, "unknown")
            lines.append(f"  - {col:<18} (dtype: {dtype:<8}): {m_cnt:>10,} missing ({m_pct:>6.2f}%)")

        lines.extend([
            "",
            f"Unique IDs:    {self.unique_ids:,}",
            f"Duplicate IDs: {self.duplicate_ids:,}",
            "-" * 50,
        ])
        return "\n".join(lines)


def inspect_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    id_column: str = "entity_id",
) -> DatasetSummary:
    """
    Inspect a dataframe and generate a comprehensive schema summary.

    Args:
        df: Input pandas DataFrame.
        dataset_name: Name identifier for logging and reporting.
        id_column: Primary identifier column to check uniqueness.

    Returns:
        DatasetSummary dataclass instance.
    """
    total_rows = len(df)
    total_columns = len(df.columns)
    columns = list(df.columns)
    column_types = {col: str(df[col].dtype) for col in columns}

    missing_counts: dict[str, int] = {}
    missing_percentages: dict[str, float] = {}
    for col in columns:
        is_missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        cnt = int(is_missing.sum())
        missing_counts[col] = cnt
        missing_percentages[col] = (cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    if id_column in df.columns:
        unique_ids = int(df[id_column].nunique(dropna=False))
        duplicate_ids = total_rows - unique_ids
    else:
        unique_ids = 0
        duplicate_ids = 0

    return DatasetSummary(
        name=dataset_name,
        total_rows=total_rows,
        total_columns=total_columns,
        columns=columns,
        column_types=column_types,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        unique_ids=unique_ids,
        duplicate_ids=duplicate_ids,
    )


def inspect_ground_truth(gt_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze ground truth relationships and return distribution statistics.
    """
    total_rows = len(gt_df)
    s1_ids = gt_df["source1_entity_id"]
    unique_s1 = int(s1_ids.nunique())
    duplicate_s1 = total_rows - unique_s1

    match_col = gt_df["matched_entity_ids"].fillna("")
    matches_list = [
        [m.strip() for m in val.split(",") if m.strip()] if val else []
        for val in match_col
    ]

    match_counts = pd.Series([len(m) for m in matches_list])
    zero_matches = int((match_counts == 0).sum())
    one_match = int((match_counts == 1).sum())
    multi_matches = int((match_counts > 1).sum())

    s2_positives = 0
    s3_positives = 0
    for m_list in matches_list:
        for mid in m_list:
            if mid.startswith("S2-"):
                s2_positives += 1
            elif mid.startswith("S3-"):
                s3_positives += 1

    total_positive_pairs = s2_positives + s3_positives

    return {
        "total_s1_rows": total_rows,
        "unique_s1_ids": unique_s1,
        "duplicate_s1_ids": duplicate_s1,
        "singleton_count": zero_matches,
        "singleton_rate": (zero_matches / total_rows * 100.0) if total_rows > 0 else 0.0,
        "one_match_count": one_match,
        "one_match_rate": (one_match / total_rows * 100.0) if total_rows > 0 else 0.0,
        "multi_match_count": multi_matches,
        "multi_match_rate": (multi_matches / total_rows * 100.0) if total_rows > 0 else 0.0,
        "total_positive_pairs": total_positive_pairs,
        "s2_positives": s2_positives,
        "s3_positives": s3_positives,
        "mean_matches_per_s1": float(match_counts.mean()) if len(match_counts) > 0 else 0.0,
        "median_matches_per_s1": float(match_counts.median()) if len(match_counts) > 0 else 0.0,
        "max_matches_per_s1": int(match_counts.max()) if len(match_counts) > 0 else 0,
        "match_count_distribution": match_counts.value_counts().head(10).to_dict(),
    }


def validate_source_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    expected_prefix: str | None = None,
) -> list[str]:
    """
    Validate all Stage 0 source dataset invariants:
    1. Required columns: entity_id, business_name, business_address, country
    2. entity_id is non-null
    3. entity_id is unique
    4. entity_id has correct prefix (e.g. 'S1-', 'S2-', 'S3-') and string type
    5. business_name is non-null and non-empty
    6. country is non-null and non-empty (strictly open-set, no hardcoded values)
    7. business_address may be null (does NOT raise violation)

    Returns:
        List of error strings (empty if dataset contract is completely valid).
    """
    issues: list[str] = []

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"[{dataset_name}] Missing required columns: {missing_cols}")
        return issues  # Cannot perform column-level checks without required columns

    total_rows = len(df)

    # 2 & 4. entity_id non-null, string type, prefix
    id_series = df["entity_id"]
    null_ids = int(id_series.isna().sum())
    if null_ids > 0:
        issues.append(f"[{dataset_name}] Found {null_ids:,} null entity_id values")

    # Check non-string / numeric IDs
    non_str_ids = sum(1 for val in id_series if not isinstance(val, str))
    if non_str_ids > 0:
        issues.append(f"[{dataset_name}] Found {non_str_ids:,} entity_id values that are not strings")

    # Check whitespace-only IDs
    whitespace_ids = sum(1 for val in id_series if isinstance(val, str) and val.strip() == "")
    if whitespace_ids > 0:
        issues.append(f"[{dataset_name}] Found {whitespace_ids:,} whitespace-only entity_id values")

    # Check prefix
    if expected_prefix is not None:
        invalid_prefix_cnt = sum(
            1 for val in id_series
            if isinstance(val, str) and not val.startswith(expected_prefix)
        )
        if invalid_prefix_cnt > 0:
            issues.append(
                f"[{dataset_name}] Found {invalid_prefix_cnt:,} entity_id values not starting with expected prefix '{expected_prefix}'"
            )
    else:
        invalid_prefix_cnt = sum(
            1 for val in id_series
            if isinstance(val, str) and not val.startswith(VALID_SOURCE_PREFIXES)
        )
        if invalid_prefix_cnt > 0:
            issues.append(
                f"[{dataset_name}] Found {invalid_prefix_cnt:,} entity_id values with invalid source prefix (must start with S1-, S2-, or S3-)"
            )

    # 3. entity_id uniqueness
    unique_ids = int(id_series.nunique(dropna=False))
    if unique_ids != total_rows:
        dup_cnt = total_rows - unique_ids
        issues.append(f"[{dataset_name}] Found {dup_cnt:,} duplicate entity_id values")

    # 5. business_name non-null & non-empty
    name_missing = df["business_name"].isna() | (df["business_name"].astype(str).str.strip() == "")
    name_missing_cnt = int(name_missing.sum())
    if name_missing_cnt > 0:
        issues.append(f"[{dataset_name}] Found {name_missing_cnt:,} null or empty business_name values")

    # 6. country non-null & non-empty (strictly open-set: accepts any non-empty string)
    country_missing = df["country"].isna() | (df["country"].astype(str).str.strip() == "")
    country_missing_cnt = int(country_missing.sum())
    if country_missing_cnt > 0:
        issues.append(f"[{dataset_name}] Found {country_missing_cnt:,} null or empty country values")

    # 7. business_address: intentionally permitted to be null (no check for non-null address)

    return issues


def validate_ground_truth_contract(
    gt_df: pd.DataFrame,
    s1_df: pd.DataFrame | set[str] | None = None,
    s2_ids: set[str] | pd.DataFrame | None = None,
    s3_ids: set[str] | pd.DataFrame | None = None,
    check_target_ownership: bool = False,
) -> list[str]:
    """
    Validate all Stage 0 Ground Truth contract invariants:
    1. Required columns: source1_entity_id, matched_entity_ids
    2. source1_entity_id is unique, non-null, and non-whitespace
    3. source1_entity_id starts with 'S1-'
    4. Referential integrity with train S1: 1-to-1 matching (if s1_df provided)
    5. No S1 ID appears inside matched_entity_ids
    6. No invalid source prefix inside matched_entity_ids (must be S2- or S3-)
    7. No duplicate matched ID occurs within a single GT row (intra-row uniqueness)
    8. Empty matched_entity_ids are valid (singletons)
    9. Referential integrity with S2 / S3 IDs (if s2_ids or s3_ids provided)

    Returns:
        List of error strings (empty if ground truth contract is completely valid).
    """
    issues: list[str] = []

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_GROUND_TRUTH_COLUMNS if c not in gt_df.columns]
    if missing_cols:
        issues.append(f"[Ground Truth] Missing required columns: {missing_cols}")
        return issues

    total_rows = len(gt_df)
    s1_series = gt_df["source1_entity_id"]

    # 2. Null check
    null_s1 = int(s1_series.isna().sum())
    if null_s1 > 0:
        issues.append(f"[Ground Truth] Found {null_s1:,} null source1_entity_id values")

    # Whitespace check
    whitespace_s1 = sum(1 for val in s1_series if isinstance(val, str) and val.strip() == "")
    if whitespace_s1 > 0:
        issues.append(f"[Ground Truth] Found {whitespace_s1:,} whitespace-only source1_entity_id values")

    # 2. Uniqueness
    unique_s1 = int(s1_series.nunique(dropna=False))
    if unique_s1 != total_rows:
        dup_s1 = total_rows - unique_s1
        issues.append(f"[Ground Truth] Found {dup_s1:,} duplicate source1_entity_id values")

    # 3. Source prefix check
    invalid_s1_prefixes = sum(
        1 for val in s1_series
        if not isinstance(val, str) or not val.startswith("S1-")
    )
    if invalid_s1_prefixes > 0:
        issues.append(f"[Ground Truth] Found {invalid_s1_prefixes:,} source1_entity_id values not starting with 'S1-'")

    # 4. Referential integrity with train_source1
    if s1_df is not None:
        if isinstance(s1_df, pd.DataFrame):
            s1_set = set(s1_df["entity_id"]) if "entity_id" in s1_df.columns else set()
        else:
            s1_set = set(s1_df)
        gt_s1_set = set(s1_series)
        diff_gt_s1 = gt_s1_set - s1_set
        diff_s1_gt = s1_set - gt_s1_set
        if diff_gt_s1:
            issues.append(f"[Ground Truth] Found {len(diff_gt_s1):,} source1 IDs not present in train_source1")
        if diff_s1_gt:
            issues.append(f"[Ground Truth] Found {len(diff_s1_gt):,} train_source1 IDs missing from Ground Truth")

    # 5, 6, 7, 8, 9. Matched candidate IDs inspection
    matched_col = gt_df["matched_entity_ids"].fillna("")
    intra_row_dups = 0
    s1_inside_matches = 0
    invalid_match_prefixes = 0
    unrecognized_s2 = 0
    unrecognized_s3 = 0

    s2_set = set(s2_ids["entity_id"]) if isinstance(s2_ids, pd.DataFrame) else s2_ids
    s3_set = set(s3_ids["entity_id"]) if isinstance(s3_ids, pd.DataFrame) else s3_ids

    target_to_s1: dict[str, str] = {}
    ambiguous_targets: set[str] = set()

    for s1_val, raw in zip(s1_series, matched_col):
        raw_s = str(raw).strip()
        if not raw_s:
            # Singletons are valid
            continue

        cands = [c.strip() for c in raw_s.split(",") if c.strip()]

        # 7. Intra-row duplicate check
        if len(cands) != len(set(cands)):
            intra_row_dups += 1

        for cid in cands:
            # Target ownership uniqueness check
            if cid in target_to_s1 and target_to_s1[cid] != s1_val:
                ambiguous_targets.add(cid)
            else:
                target_to_s1[cid] = str(s1_val)

            # 5. No S1 ID in matches
            if cid.startswith("S1-"):
                s1_inside_matches += 1
            # 6. Prefix must be S2- or S3-
            elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                invalid_match_prefixes += 1

            # 9. Referential integrity with S2 / S3
            if s2_set is not None and cid.startswith("S2-") and cid not in s2_set:
                unrecognized_s2 += 1
            if s3_set is not None and cid.startswith("S3-") and cid not in s3_set:
                unrecognized_s3 += 1

    if intra_row_dups > 0:
        issues.append(f"[Ground Truth] Found {intra_row_dups:,} rows with duplicate matched IDs within the same list")
    if check_target_ownership and ambiguous_targets:
        issues.append(f"[Ground Truth] Found {len(ambiguous_targets):,} target candidate IDs linked to multiple S1 entities (ambiguous target ownership)")
    if s1_inside_matches > 0:
        issues.append(f"[Ground Truth] Found {s1_inside_matches:,} S1 IDs appearing inside matched_entity_ids (forbidden)")
    if invalid_match_prefixes > 0:
        issues.append(f"[Ground Truth] Found {invalid_match_prefixes:,} matched candidate IDs with invalid prefixes (must be S2- or S3-)")
    if unrecognized_s2 > 0:
        issues.append(f"[Ground Truth] Found {unrecognized_s2:,} S2 candidate IDs not found in source 2 dataset")
    if unrecognized_s3 > 0:
        issues.append(f"[Ground Truth] Found {unrecognized_s3:,} S3 candidate IDs not found in source 3 dataset")

    return issues


def validate_schema(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str] | None = None,
    id_column: str = "entity_id",
) -> list[str]:
    """
    Backwards-compatible schema validation function.
    Delegates to validate_source_dataset for source data, or validates basic columns and ID uniqueness.
    """
    if required_columns is None or required_columns == REQUIRED_SOURCE_COLUMNS:
        return validate_source_dataset(df, dataset_name)

    issues: list[str] = []
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"[{dataset_name}] Missing required columns: {missing_cols}")

    if id_column in df.columns:
        null_ids = int(df[id_column].isna().sum())
        if null_ids > 0:
            issues.append(f"[{dataset_name}] Found {null_ids:,} null ID values")

        dup_ids = len(df) - int(df[id_column].nunique())
        if dup_ids > 0:
            issues.append(f"[{dataset_name}] Found {dup_ids:,} duplicate ID values")

    return issues
