"""Dataframe preprocessing that preserves source fields alongside clean fields."""

from __future__ import annotations

from typing import Any

from .normalize import extract_digits, name_core, normalize_address, normalize_country, normalize_name


def preprocess_dataframe(
    dataframe: Any,
    *,
    name_column: str = "business_name",
    address_column: str = "business_address",
    country_column: str = "country",
) -> Any:
    """Return a copy with normalized columns, leaving all original values unchanged."""
    required_columns = (name_column, address_column, country_column)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")

    result = dataframe.copy()
    result["norm_name"] = result[name_column].map(normalize_name)
    result["name_core"] = result["norm_name"].map(name_core)
    result["norm_address"] = result[address_column].map(normalize_address)
    result["digit_tokens"] = result[address_column].map(extract_digits)
    result["norm_country"] = result[country_column].map(normalize_country)
    return result
