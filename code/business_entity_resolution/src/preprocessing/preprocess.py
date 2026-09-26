"""Dataframe preprocessing that preserves source fields alongside clean fields."""

from __future__ import annotations

from typing import Any

from .normalize import (
    _is_missing,
    accent_fold,
    extract_digits,
    name_core,
    normalize_address,
    normalize_country,
    normalize_name,
)


def _tokens(value: str) -> list[str]:
    """Return whitespace tokens, or an empty list for an empty normalized value."""
    return value.split() if value else []


def _postcode_candidates(digit_tokens: list[str]) -> list[str]:
    """Return generic long numeric candidates without country-specific assumptions."""
    return [token for token in digit_tokens if len(token) >= 4]


def preprocess_dataframe(
    dataframe: Any,
    *,
    name_column: str = "business_name",
    address_column: str = "business_address",
    country_column: str = "country",
) -> Any:
    """Return a copy with stable normalized columns and unchanged raw columns.

    String fields (``norm_*`` and ``accent_folded_*``) use ``""`` for missing
    inputs. Token and numeric-token fields are ``list[str]`` and use ``[]``.
    Missingness fields are ``bool``. ``postcode_candidates`` is a ``list[str]``
    of address numeric tokens with four or more digits, preserving leading zeros.
    """
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
    result["accent_folded_name"] = result["norm_name"].map(accent_fold)
    result["accent_folded_address"] = result["norm_address"].map(accent_fold)
    result["name_tokens"] = result["norm_name"].map(_tokens)
    result["address_tokens"] = result["norm_address"].map(_tokens)
    result["name_missing"] = result[name_column].map(_is_missing)
    result["address_missing"] = result[address_column].map(_is_missing)
    result["country_missing"] = result[country_column].map(_is_missing)
    result["name_numeric_tokens"] = result[name_column].map(extract_digits)
    result["address_numeric_tokens"] = result[address_column].map(extract_digits)
    result["postcode_candidates"] = result["address_numeric_tokens"].map(_postcode_candidates)
    return result
