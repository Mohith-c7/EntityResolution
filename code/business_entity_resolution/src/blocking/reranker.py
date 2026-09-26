"""Deterministic fused-score reranking for blocking candidates."""

from __future__ import annotations

from typing import Mapping


EXACT_NAME_PATH = "exact_norm_name"
EXACT_CORE_PATH = "exact_name_core"


def fused_score(path_scores: Mapping[str, float]) -> float:
    """Sum path scores and apply the architecture's exactness bonuses."""
    score = sum(path_scores.values())
    if path_scores.get(EXACT_NAME_PATH, 0.0) > 0:
        score += 2.0
    if path_scores.get(EXACT_CORE_PATH, 0.0) > 0:
        score += 1.5
    return score


def rerank_candidates(candidate_scores: Mapping[str, Mapping[str, float]], top_k: int = 20) -> list[tuple[str, dict[str, float], float]]:
    """Return the highest-scoring candidates with a stable ID tie-breaker."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = [
        (candidate_id, dict(scores), fused_score(scores))
        for candidate_id, scores in candidate_scores.items()
    ]
    ranked.sort(key=lambda row: (-row[2], row[0]))
    return ranked[:top_k]
