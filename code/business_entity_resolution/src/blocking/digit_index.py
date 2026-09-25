"""Rare digit-token index used for candidate blocking."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def tokenize_digits(value: Any) -> tuple[str, ...]:
    """Accept preprocessing lists or strings and return unique ordered digit tokens."""
    if isinstance(value, (list, tuple, set)):
        tokens = (str(token) for token in value)
    else:
        tokens = str(value or "").split()
    return tuple(sorted({token for token in tokens if token.isdigit()}))


class RareDigitIndex:
    """Inverted index that retains only low document-frequency digit tokens."""

    def __init__(self, max_token_df: float = 0.10) -> None:
        if not 0 < max_token_df <= 1:
            raise ValueError("max_token_df must be in (0, 1]")
        self.max_token_df = max_token_df
        self.postings: dict[str, tuple[str, ...]] = {}

    def build(self, records: Any, *, id_column: str = "entity_id") -> "RareDigitIndex":
        document_tokens = [tokenize_digits(value) for value in records["digit_tokens"]]
        frequencies = Counter(token for tokens in document_tokens for token in tokens)
        max_count = max(1, int(len(records) * self.max_token_df))
        postings: dict[str, list[str]] = defaultdict(list)
        for record_id, tokens in zip(records[id_column], document_tokens):
            for token in tokens:
                if frequencies[token] <= max_count:
                    postings[token].append(str(record_id))
        self.postings = {token: tuple(sorted(ids)) for token, ids in sorted(postings.items())}
        return self

    def query(self, digit_tokens: Any) -> dict[str, float]:
        scores: Counter[str] = Counter()
        for token in tokenize_digits(digit_tokens):
            scores.update(self.postings.get(token, ()))
        return {record_id: float(scores[record_id]) for record_id in sorted(scores)}
