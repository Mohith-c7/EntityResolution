"""Deterministic normalization helpers for business entity fields."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


NAME_ABBREVIATIONS = {
    "co": "company", "corp": "corporation", "dept": "department",
    "inc": "incorporated", "intl": "international", "ltd": "limited",
    "mfg": "manufacturing", "pvt": "private", "svc": "services",
    "svcs": "services", "tech": "technology",
}
ADDRESS_ABBREVIATIONS = {
    "apt": "apartment", "ave": "avenue", "blvd": "boulevard", "ct": "court",
    "dr": "drive", "fl": "floor", "hwy": "highway", "ln": "lane",
    "no": "number", "pkwy": "parkway", "rd": "road", "st": "street",
    "ste": "suite",
}
LEGAL_SUFFIXES = frozenset({
    "company", "corporation", "incorporated", "limited", "llc", "llp", "private",
})


def _is_missing(value: Any) -> bool:
    """Return whether a scalar is null without making pandas a dependency here."""
    if value is None:
        return True
    try:
        comparison = value != value
        if comparison is True:
            return True
        return str(comparison) == "<NA>"
    except (TypeError, ValueError):
        return False


def _clean_text(value: Any) -> str:
    """Apply shared Unicode, case, punctuation, and whitespace cleanup."""
    if _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _expand_tokens(text: str, replacements: dict[str, str]) -> str:
    return " ".join(replacements.get(token, token) for token in text.split())


def normalize_name(raw_name: Any) -> str:
    """Normalize a business name, expanding common business abbreviations."""
    if _is_missing(raw_name):
        return ""
    value = unicodedata.normalize("NFKC", str(raw_name)).replace("&", " and ")
    return _expand_tokens(_clean_text(value), NAME_ABBREVIATIONS)


def normalize_address(raw_address: Any) -> str:
    """Normalize an address while retaining all textual and numeric tokens."""
    if _is_missing(raw_address):
        return ""
    value = unicodedata.normalize("NFKC", str(raw_address))
    value = re.sub(r"\bh\s*\.?\s*no\.?\b", " house number ", value, flags=re.I)
    return _expand_tokens(_clean_text(value), ADDRESS_ABBREVIATIONS)


def normalize_country(raw_country: Any) -> str:
    """Normalize country-label formatting without mapping country values."""
    return _clean_text(raw_country)


def extract_digits(text: Any) -> list[str]:
    """Return ordered ASCII digit tokens, including punctuation-separated values."""
    return [
        "".join(str(unicodedata.digit(char)) for char in token)
        for token in re.findall(r"\d+", _clean_text(text))
    ]


def name_core(normalized_name: Any) -> str:
    """Remove legal-form tokens from an already normalized business name."""
    return " ".join(
        token for token in _clean_text(normalized_name).split() if token not in LEGAL_SUFFIXES
    )
