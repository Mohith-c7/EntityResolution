"""
Schema validation, contract enforcement, and inspection utilities for entity resolution datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


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
        cnt = int(df[col].isna().sum())
        # Also count empty strings or whitespace-only strings as missing for object types
        if df[col].dtype == object:
            cnt += int((df[col].astype(str).str.strip() == "").sum()) - int(df[col].isna().sum())
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

    # Parse matched IDs
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
        "mean_matches_per_s1": float(match_counts.mean()),
        "median_matches_per_s1": float(match_counts.median()),
        "max_matches_per_s1": int(match_counts.max()) if len(match_counts) > 0 else 0,
        "match_count_distribution": match_counts.value_counts().head(10).to_dict(),
    }


def validate_schema(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str] | None = None,
    id_column: str = "entity_id",
) -> list[str]:
    """
    Validate dataset contract. Returns list of warning or error messages (empty if completely clean).
    """
    issues: list[str] = []
    if required_columns is None:
        required_columns = REQUIRED_SOURCE_COLUMNS

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
