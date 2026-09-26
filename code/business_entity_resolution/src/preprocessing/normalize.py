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
LEGAL_SUFFIXES = frozenset({
    "company", "corporation", "incorporated", "limited", "llc", "llp", "private",
})


def _is_missing(value: Any) -> bool:
    """Return whether a scalar is null without making pandas a dependency here."""
    if value is None:
        return True
    try:
        # This catches Python and NumPy NaN scalar values.  pandas.NA raises
        # when coerced to bool, which is handled below.
        return bool(value != value)
    except (TypeError, ValueError):
        return type(value).__name__ == "NAType"


def _clean_text(value: Any) -> str:
    """Apply shared Unicode, case, punctuation, and whitespace cleanup."""
    if _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    # ``\w`` excludes Unicode combining marks.  Retaining the L, N, and M
    # categories preserves scripts such as Devanagari and Odia intact.
    text = "".join(
        char if char.isspace() or unicodedata.category(char)[0] in {"L", "N", "M"} else " "
        for char in text
    )
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
    """Normalize an address without expanding potentially ambiguous abbreviations."""
    return _clean_text(raw_address)


def normalize_country(raw_country: Any) -> str:
    """Normalize country-label formatting without mapping country values."""
    return _clean_text(raw_country)


def extract_digits(text: Any) -> list[str]:
    """Return ordered ASCII digit tokens, including punctuation-separated values."""
    return [
        "".join(str(unicodedata.digit(char)) for char in token)
        for token in re.findall(r"\d+", _clean_text(text))
    ]


def accent_fold(text: Any) -> str:
    """Return a Latin-accent-folded alternate view without damaging Indic marks."""
    primary = _clean_text(text)
    folded: list[str] = []
    previous_base = ""
    for char in unicodedata.normalize("NFKD", primary):
        category = unicodedata.category(char)
        if category.startswith("M"):
            if "LATIN" not in unicodedata.name(previous_base, ""):
                folded.append(char)
        else:
            folded.append(char)
            if category[0] in {"L", "N"}:
                previous_base = char
    return unicodedata.normalize("NFC", "".join(folded))


def name_core(normalized_name: Any) -> str:
    """Remove legal-form tokens from an already normalized business name."""
    return " ".join(
        token for token in _clean_text(normalized_name).split() if token not in LEGAL_SUFFIXES
    )
