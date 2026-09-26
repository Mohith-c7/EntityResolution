"""Exact and composite blocking-key helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .digit_index import tokenize_digits
from .name_index import tokenize_name


def build_exact_index(records: Any, column: str, *, id_column: str = "entity_id") -> dict[str, tuple[str, ...]]:
    """Build a deterministic exact-value index, ignoring empty keys."""
    index: dict[str, list[str]] = defaultdict(list)
    for record_id, value in zip(records[id_column], records[column]):
        key = str(value or "").strip()
        if key:
            index[key].append(str(record_id))
    return {key: tuple(sorted(ids)) for key, ids in sorted(index.items())}


def lookup_exact(index: dict[str, tuple[str, ...]], value: Any) -> dict[str, float]:
    """Return exact matches with a unit path score."""
    key = str(value or "").strip()
    return {record_id: 1.0 for record_id in index.get(key, ())} if key else {}


def core_prefix(value: Any, length: int = 8) -> str:
    """Return the compact name-core prefix used by the prefix block."""
    return "".join(str(value or "").split())[:length]


def build_prefix_index(records: Any, *, id_column: str = "entity_id", length: int = 8) -> dict[str, tuple[str, ...]]:
    """Build an eight-character normalized-core prefix index."""
    index: dict[str, list[str]] = defaultdict(list)
    for record_id, value in zip(records[id_column], records["name_core"]):
        key = core_prefix(value, length)
        if key:
            index[key].append(str(record_id))
    return {key: tuple(sorted(ids)) for key, ids in sorted(index.items())}


def postcode_name_keys(digit_tokens: Any, normalized_name: Any) -> tuple[tuple[str, str], ...]:
    """Return generic long-number/name-token keys without country-specific rules."""
    postcodes = [token for token in tokenize_digits(digit_tokens) if len(token) >= 4]
    names = tokenize_name(normalized_name)
    return tuple((postcode, name) for postcode in postcodes for name in names)


def build_postcode_name_index(records: Any, *, id_column: str = "entity_id") -> dict[tuple[str, str], tuple[str, ...]]:
    """Build a deterministic postcode-plus-name-token index."""
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record_id, digits, name in zip(records[id_column], records["digit_tokens"], records["norm_name"]):
        for key in postcode_name_keys(digits, name):
            index[key].append(str(record_id))
    return {key: tuple(sorted(ids)) for key, ids in sorted(index.items())}


def lookup_postcode_name(index: dict[tuple[str, str], tuple[str, ...]], digit_tokens: Any, normalized_name: Any) -> dict[str, float]:
    """Return candidates sharing a generic postcode-plus-name-token key."""
    scores: dict[str, float] = defaultdict(float)
    for key in postcode_name_keys(digit_tokens, normalized_name):
        for record_id in index.get(key, ()):
            scores[record_id] += 1.0
    return {record_id: scores[record_id] for record_id in sorted(scores)}


from .token_index import TokenIndex


def build_exact_indexes(records, statistics):
    return {
        kind: TokenIndex(records, kind, statistics)
        for kind in ("exact_name", "exact_core", "core_prefix", "postcode_name")
    }
