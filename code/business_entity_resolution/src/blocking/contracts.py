"""Blocking contracts shared by retrieval, diagnostics, and the model handoff."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


PATHS = (
    "rare_name", "rare_address", "rare_digits", "exact_name", "exact_core",
    "core_prefix", "postcode_name", "name_character",
)
CANDIDATE_COLUMNS = (
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "blocking_score", "rank_within_source", "blocking_paths",
)


@dataclass(frozen=True, slots=True)
class BlockingConfig:
    top_k: int = 20
    path_top_k: int = 200
    max_query_tokens: int = 8
    max_token_df_ratio: float = 0.10
    max_token_postings: int = 512
    character_min_n: int = 3
    character_max_n: int = 5
    character_query_grams: int = 12
    max_character_df_ratio: float = 0.50
    max_character_postings: int = 1024
    max_exact_block_size: int = 4096
    core_prefix_length: int = 8
    exact_name_bonus: float = 0.15
    exact_core_bonus: float = 0.10
    path_agreement_bonus: float = 0.02
    country_mismatch_penalty: float = 0.05

    def __post_init__(self) -> None:
        integer_fields = (
            "top_k", "path_top_k", "max_query_tokens", "max_token_postings",
            "character_min_n", "character_max_n", "character_query_grams",
            "max_character_postings", "max_exact_block_size", "core_prefix_length",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.path_top_k < self.top_k:
            raise ValueError("path_top_k must be at least top_k")
        if self.character_min_n > self.character_max_n:
            raise ValueError("character_min_n cannot exceed character_max_n")
        for field in ("max_token_df_ratio", "max_character_df_ratio"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{field} must be in (0, 1]")
        for field in (
            "exact_name_bonus", "exact_core_bonus", "path_agreement_bonus",
            "country_mismatch_penalty",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scalar_text(value: Any) -> str:
    """Read a normalized scalar without turning a null into a blocking key."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def token_values(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset(value.split())
    return frozenset(scalar_text(token) for token in value if scalar_text(token))


def character_grams(text: str, config: BlockingConfig) -> frozenset[str]:
    # Include boundary markers and retain Unicode. This is a retrieval view,
    # not a replacement for the upstream normalized field.
    if not text:
        return frozenset()
    value = f" {text} "
    return frozenset(
        value[start:start + n]
        for n in range(config.character_min_n, config.character_max_n + 1)
        for start in range(max(0, len(value) - n + 1))
    )


@dataclass(frozen=True, slots=True)
class BlockingRecord:
    entity_id: str
    name: str
    core: str
    address: str
    country: str
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    name_digits: frozenset[str]
    address_digits: frozenset[str]
    postcode_candidates: frozenset[str]
    retrieval_name: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], source: str) -> "BlockingRecord":
        raw_id = row.get("entity_id")
        entity_id = scalar_text(raw_id)
        if (not isinstance(raw_id, str) or not entity_id.startswith(f"{source}-")
                or len(entity_id) <= 3 or entity_id != raw_id
                or any(char.isspace() or char == "," for char in entity_id)):
            raise ValueError(f"Expected a nonblank {source}- ID, got {row.get('entity_id')!r}")
        # Existing repository names and the revised handoff names are supported.
        def field(existing: str, revised: str) -> str:
            if existing in row:
                return scalar_text(row[existing])
            if revised in row:
                return scalar_text(row[revised])
            raise ValueError(f"Missing normalized field {existing!r} (or {revised!r})")

        name = field("norm_name", "name_norm")
        address = field("norm_address", "address_norm")
        country = field("norm_country", "country_norm")
        core = scalar_text(row.get("name_core", name))
        name_digits = token_values(row.get("name_digits", ()))
        address_digits = token_values(row.get("address_digits", row.get("digit_tokens", ())))
        # Format-based numeric candidates only. These are not verified postcodes.
        # Explicit upstream candidates take precedence, including an empty set.
        postcodes = token_values(row["postcode_candidates"]) if "postcode_candidates" in row else (
            frozenset(t for t in address_digits if t.isdecimal() and 4 <= len(t) <= 6)
        )
        return cls(
            entity_id, name, core, address, country,
            token_values(row.get("name_tokens", name)),
            token_values(row.get("address_tokens", address)),
            name_digits, address_digits, postcodes,
            scalar_text(row.get("name_folded", row.get("norm_name_folded", core or name))),
        )

    def keys(self, kind: str, config: BlockingConfig) -> frozenset[str]:
        if kind == "name":
            return self.name_tokens
        if kind == "address":
            return self.address_tokens
        if kind == "digits":
            return frozenset(
                [f"name:{t}" for t in self.name_digits]
                + [f"address:{t}" for t in self.address_digits]
            )
        if kind == "character":
            return character_grams(self.retrieval_name, config)
        if kind == "exact_name":
            return frozenset((self.name,)) if self.name else frozenset()
        if kind == "exact_core":
            return frozenset((self.core,)) if self.core else frozenset()
        if kind == "core_prefix":
            return frozenset(("".join(self.core.split())[:config.core_prefix_length],)) if self.core else frozenset()
        if kind == "postcode_name":
            return frozenset(f"{p}\x1f{t}" for p in self.postcode_candidates for t in self.name_tokens)
        raise ValueError(f"Unknown index kind {kind!r}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, frozenset):
                result[key] = sorted(value)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlockingRecord":
        value = dict(value)
        for field in (
            "name_tokens", "address_tokens", "name_digits", "address_digits", "postcode_candidates",
        ):
            value[field] = frozenset(value[field])
        return cls(**value)


def records_from_dataframe(frame: pd.DataFrame, source: str) -> list[BlockingRecord]:
    records = [BlockingRecord.from_mapping(row, source) for row in frame.to_dict("records")]
    ids = [record.entity_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {source} entity IDs")
    return records


@dataclass(frozen=True, slots=True)
class Candidate:
    source1_entity_id: str
    candidate_entity_id: str
    candidate_source: str
    blocking_score: float
    blocking_paths: tuple[str, ...]
    rank_within_source: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_candidates(candidates: Iterable[Candidate], top_k: int) -> list[Candidate]:
    from dataclasses import replace

    result = []
    for source in ("S2", "S3"):
        selected = sorted(
            (candidate for candidate in candidates if candidate.candidate_source == source),
            key=lambda candidate: (-candidate.blocking_score, candidate.candidate_entity_id),
        )[:top_k]
        result.extend(replace(candidate, rank_within_source=i + 1) for i, candidate in enumerate(selected))
    return result
