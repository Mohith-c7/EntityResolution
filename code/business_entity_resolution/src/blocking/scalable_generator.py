"""Multi-view candidate retrieval and a stable DataFrame handoff."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .address_index import AddressIndex
from .character_index import CharacterIndex
from .contracts import CANDIDATE_COLUMNS, BlockingConfig, BlockingRecord, Candidate, rank_candidates, records_from_dataframe
from .digit_index import DigitIndex
from .exact_blocks import build_exact_indexes
from .name_index import NameIndex
from .identity_reranker import fused_score
from .token_index import CorpusStatistics


@dataclass(frozen=True)
class QueryCandidates:
    reference: BlockingRecord
    before_top_k: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]


class SourceIndex:
    def __init__(self, records: list[BlockingRecord], source: str, config: BlockingConfig, statistics: CorpusStatistics | None = None):
        if source not in ("S2", "S3"):
            raise ValueError("Target source must be S2 or S3")
        if any(not record.entity_id.startswith(f"{source}-") for record in records):
            raise ValueError(f"Target records must belong to {source}")
        if len(records) != len({record.entity_id for record in records}):
            raise ValueError(f"Duplicate {source} target IDs")
        self.records, self.source, self.config = records, source, config
        self.statistics = statistics or CorpusStatistics(config)
        if statistics is None:
            self.statistics.update(records)
        elif statistics.config != config:
            raise ValueError("Index configuration must match corpus statistics")
        self.indexes = {
            "rare_name": NameIndex(records, self.statistics),
            "rare_address": AddressIndex(records, self.statistics),
            "rare_digits": DigitIndex(records, self.statistics),
            "name_character": CharacterIndex(records, self.statistics),
            **build_exact_indexes(records, self.statistics),
        }

    def query(self, reference: BlockingRecord) -> QueryCandidates:
        if not reference.entity_id.startswith("S1-"):
            raise ValueError("A reference must belong to S1")
        paths_by_position: dict[int, set[str]] = defaultdict(set)
        for path, index in self.indexes.items():
            for position in index.query(reference):
                paths_by_position[position].add(path)
        scored = []
        for position, paths in paths_by_position.items():
            target = self.records[position]
            paths_tuple = tuple(sorted(paths))
            scored.append(Candidate(
                reference.entity_id, target.entity_id, self.source,
                fused_score(reference, target, paths_tuple, self.statistics), paths_tuple,
            ))
        scored.sort(key=lambda candidate: (-candidate.blocking_score, candidate.candidate_entity_id))
        return QueryCandidates(reference, tuple(scored), tuple(rank_candidates(scored, self.config.top_k)))


class CandidateGenerator:
    """In-memory integration API; use the CLI for disk-backed shard processing."""

    def __init__(self, source2: pd.DataFrame, source3: pd.DataFrame, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self.indexes = (
            SourceIndex(records_from_dataframe(source2, "S2"), "S2", self.config),
            SourceIndex(records_from_dataframe(source3, "S3"), "S3", self.config),
        )

    def iter_results(self, source1: pd.DataFrame) -> Iterable[QueryCandidates]:
        for reference in records_from_dataframe(source1, "S1"):
            results = [index.query(reference) for index in self.indexes]
            before = tuple(candidate for result in results for candidate in result.before_top_k)
            selected = tuple(candidate for result in results for candidate in result.candidates)
            yield QueryCandidates(reference, before, selected)

    def generate(self, source1: pd.DataFrame) -> pd.DataFrame:
        rows = [candidate.to_dict() for result in self.iter_results(source1) for candidate in result.candidates]
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame, config: BlockingConfig | None = None) -> pd.DataFrame:
    return CandidateGenerator(source2, source3, config).generate(source1)
