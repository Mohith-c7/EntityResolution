"""Baseline and scalable retrieval APIs; neither makes final match decisions."""

from .candidate_generator import generate_candidates
from .contracts import BlockingConfig
from .scalable_generator import CandidateGenerator, SourceIndex
from .scalable_generator import generate_candidates as generate_scalable_candidates

__all__ = [
    "generate_candidates", "generate_scalable_candidates", "BlockingConfig",
    "CandidateGenerator", "SourceIndex",
]
