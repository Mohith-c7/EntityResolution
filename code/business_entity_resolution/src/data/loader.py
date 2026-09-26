"""
TSV Data Loader for Amazon Business Entity Resolution 2026.

Provides robust, reproducible loading of raw tabular entity datasets and ground truth.
Preserves raw entity IDs, string representations, and null values without normalization,
deduplication, or modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TEST_PATH, TRAIN_PATH

logger = logging.getLogger(__name__)

# Standard column dtypes to ensure IDs and text are loaded as raw string object types
SOURCE_DTYPES: dict[str, Any] = {
    "entity_id": str,
    "business_name": str,
    "business_address": str,
    "country": str,
}

GROUND_TRUTH_DTYPES: dict[str, Any] = {
    "source1_entity_id": str,
    "matched_entity_ids": str,
}


def load_tsv(
    file_path: Path | str,
    *,
    dtype: dict[str, Any] | type | None = str,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    keep_default_na: bool = False,
    na_values: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Safely load a tab-separated dataset file with explicit path and encoding checks.

    Args:
        file_path: Path to the TSV file.
        dtype: Data types specification. Defaults to strings; pass None explicitly for type inference.
        nrows: Optional maximum number of rows to read.
        usecols: Optional subset of columns to load.
        keep_default_na: Whether to convert default strings (like 'NA') to NaN.
                         Defaults to False to protect 2-letter country codes like 'NA' (Namibia).
        na_values: Additional explicit null strings (e.g. ['', 'null', 'NULL']).
        **kwargs: Additional parameters passed to pd.read_csv.

    Returns:
        pd.DataFrame containing the raw loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. Verify dataset path and uncompressed files."
        )

    if na_values is None:
        na_values = ["", "null", "NULL", "None"]

    logger.info("Loading TSV from: %s (nrows=%s)", path, nrows)
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=dtype,
            nrows=nrows,
            usecols=usecols,
            keep_default_na=keep_default_na,
            na_values=na_values,
            encoding="utf-8",
            on_bad_lines="error",
            **kwargs,
        )
    except Exception as exc:
        logger.error("Failed to read TSV from %s: %s", path, exc)
        raise

    return df


def load_train_source1(
    train_dir: Path | str = TRAIN_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load train_source1.tsv containing source 1 business entity records."""
    file_path = Path(train_dir) / "train_source1.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_train_source2(
    train_dir: Path | str = TRAIN_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load train_source2.tsv containing source 2 business entity records."""
    file_path = Path(train_dir) / "train_source2.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_train_source3(
    train_dir: Path | str = TRAIN_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load train_source3.tsv containing source 3 business entity records."""
    file_path = Path(train_dir) / "train_source3.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_ground_truth(
    train_dir: Path | str = TRAIN_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load train_ground_truth.tsv mapping source1 IDs to matched entity IDs."""
    file_path = Path(train_dir) / "train_ground_truth.tsv"
    return load_tsv(file_path, dtype=GROUND_TRUTH_DTYPES, nrows=nrows, **kwargs)


def load_test_source1(
    test_dir: Path | str = TEST_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load test_source1.tsv containing test source 1 business entity records."""
    file_path = Path(test_dir) / "test_source1.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_test_source2(
    test_dir: Path | str = TEST_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load test_source2.tsv containing test source 2 business entity records."""
    file_path = Path(test_dir) / "test_source2.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_test_source3(
    test_dir: Path | str = TEST_PATH,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load test_source3.tsv containing test source 3 business entity records."""
    file_path = Path(test_dir) / "test_source3.tsv"
    return load_tsv(file_path, dtype=SOURCE_DTYPES, nrows=nrows, **kwargs)


def load_dataset(file_path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Generic wrapper for backward compatibility and ad-hoc file inspection."""
    return load_tsv(file_path, **kwargs)
