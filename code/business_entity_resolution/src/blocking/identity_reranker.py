"""Symmetric identity evidence for retrieval reranking, not a trained matcher."""

from __future__ import annotations

import math

from rapidfuzz import fuzz

from .contracts import BlockingRecord, character_grams
from .token_index import CorpusStatistics, weighted_jaccard


def fused_score(left: BlockingRecord, right: BlockingRecord, paths: tuple[str, ...], statistics: CorpusStatistics) -> float:
    config = statistics.config
    name_overlap = weighted_jaccard(left.name_tokens, right.name_tokens, "name", statistics)
    address_overlap = weighted_jaccard(left.address_tokens, right.address_tokens, "address", statistics)
    char_left = character_grams(left.retrieval_name, config)
    char_right = character_grams(right.retrieval_name, config)
    character_overlap = weighted_jaccard(char_left, char_right, "character", statistics)
    name_fuzzy = fuzz.token_sort_ratio(left.name, right.name) / 100 if left.name and right.name else 0.0
    address_fuzzy = fuzz.token_sort_ratio(left.address, right.address) / 100 if left.address and right.address else 0.0
    digit_union = left.address_digits | right.address_digits
    digit_overlap = len(left.address_digits & right.address_digits) / len(digit_union) if digit_union else 0.0
    score = (
        0.25 * name_overlap + 0.25 * character_overlap + 0.20 * name_fuzzy
        + 0.15 * address_overlap + 0.10 * address_fuzzy + 0.05 * digit_overlap
    )
    if left.name and left.name == right.name:
        score += config.exact_name_bonus
    if left.core and left.core == right.core:
        score += config.exact_core_bonus
    score += config.path_agreement_bonus * math.log1p(len(paths))
    if left.country and right.country and left.country != right.country:
        score -= config.country_mismatch_penalty
    return float(score)
