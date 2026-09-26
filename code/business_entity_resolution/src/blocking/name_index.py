"""Rare normalized-name token index used for candidate blocking."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def tokenize_name(value: Any) -> tuple[str, ...]:
    """Return deterministic, unique normalized-name tokens."""
    return tuple(sorted(set(str(value or "").split())))


class RareNameIndex:
    """Inverted index that retains only low document-frequency name tokens."""

    def __init__(self, max_token_df: float = 0.10) -> None:
        if not 0 < max_token_df <= 1:
            raise ValueError("max_token_df must be in (0, 1]")
        self.max_token_df = max_token_df
        self.postings: dict[str, tuple[str, ...]] = {}

    def build(self, records: Any, *, id_column: str = "entity_id") -> "RareNameIndex":
        document_tokens = [tokenize_name(value) for value in records["norm_name"]]
        frequencies = Counter(token for tokens in document_tokens for token in tokens)
        max_count = max(1, int(len(records) * self.max_token_df))
        postings: dict[str, list[str]] = defaultdict(list)
        for record_id, tokens in zip(records[id_column], document_tokens):
            for token in tokens:
                if frequencies[token] <= max_count:
                    postings[token].append(str(record_id))
        self.postings = {token: tuple(sorted(ids)) for token, ids in sorted(postings.items())}
        return self

    def query(self, normalized_name: Any) -> dict[str, float]:
        """Return candidate IDs and shared-rare-token counts."""
        scores: Counter[str] = Counter()
        for token in tokenize_name(normalized_name):
            scores.update(self.postings.get(token, ()))
        return {record_id: float(scores[record_id]) for record_id in sorted(scores)}
