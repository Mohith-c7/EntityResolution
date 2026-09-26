"""Compact inverted postings with bounded query work and corpus-wide statistics."""

from __future__ import annotations

import math
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from heapq import nsmallest
from typing import Iterable

from rapidfuzz import fuzz

from .contracts import BlockingConfig, BlockingRecord


KINDS = ("name", "address", "digits", "character", "exact_name", "exact_core", "core_prefix", "postcode_name")


@dataclass
class CorpusStatistics:
    config: BlockingConfig
    record_count: int = 0
    document_frequency: dict[str, Counter] = field(default_factory=lambda: {kind: Counter() for kind in KINDS})

    def update(self, records: Iterable[BlockingRecord]) -> None:
        for record in records:
            self.record_count += 1
            for kind in KINDS:
                self.document_frequency[kind].update(record.keys(kind, self.config))

    def idf(self, kind: str, token: str) -> float:
        return math.log((self.record_count + 1) / (self.document_frequency[kind].get(token, 0) + 1)) + 1

    def posting_limit(self, kind: str) -> int:
        if kind == "character":
            return min(self.config.max_character_postings, max(1, int(self.record_count * self.config.max_character_df_ratio)))
        if kind in ("name", "address", "digits"):
            return min(self.config.max_token_postings, max(1, int(self.record_count * self.config.max_token_df_ratio)))
        return self.config.max_exact_block_size


class TokenIndex:
    """Store integer row positions, not repeated business ID strings."""

    def __init__(self, records: list[BlockingRecord], kind: str, statistics: CorpusStatistics):
        self.records = records
        self.kind = kind
        self.statistics = statistics
        limit = statistics.posting_limit(kind)
        self.postings: dict[str, array] = {}
        self.character_norms = array("d")
        for position, record in enumerate(records):
            keys = record.keys(kind, statistics.config)
            if kind == "character":
                self.character_norms.append(math.sqrt(math.fsum(statistics.idf(kind, token) ** 2 for token in sorted(keys))))
            for token in sorted(keys):
                if statistics.document_frequency[kind][token] <= limit:
                    self.postings.setdefault(token, array("I")).append(position)

    def query(self, record: BlockingRecord) -> dict[int, float]:
        config = self.statistics.config
        keys = record.keys(self.kind, config)
        eligible = sorted(
            (token for token in keys if token in self.postings),
            key=lambda token: (self.statistics.document_frequency[self.kind][token], token),
        )
        query_limit = config.character_query_grams if self.kind == "character" else config.max_query_tokens
        # Exact views have one key; postcode+name can have several compound keys.
        eligible = eligible[:query_limit]
        scores: dict[int, float] = defaultdict(float)
        for token in eligible:
            weight = self.statistics.idf(self.kind, token)
            for position in self.postings[token]:
                scores[position] += weight ** 2 if self.kind == "character" else weight
        if self.kind == "character":
            query_norm = math.sqrt(math.fsum(self.statistics.idf(self.kind, token) ** 2 for token in sorted(keys)))
            scores = {
                position: score / (query_norm * self.character_norms[position])
                for position, score in scores.items()
                if query_norm and self.character_norms[position]
            }
        # Large exact-name/prefix blocks need address evidence before truncation;
        # otherwise an ID-order cutoff can throw away the right branch.
        exact_view = self.kind in ("exact_name", "exact_core", "core_prefix")
        selected = nsmallest(
            config.path_top_k, scores,
            key=lambda position: (
                -scores[position],
                -fuzz.token_sort_ratio(record.address, self.records[position].address)
                if exact_view and record.address and self.records[position].address else 0,
                self.records[position].entity_id,
            ),
        )
        return {position: scores[position] for position in selected}


def weighted_jaccard(left: frozenset[str], right: frozenset[str], kind: str, statistics: CorpusStatistics) -> float:
    union = left | right
    if not left or not right or not union:
        return 0.0
    denominator = math.fsum(statistics.idf(kind, token) for token in sorted(union))
    return math.fsum(statistics.idf(kind, token) for token in sorted(left & right)) / denominator
