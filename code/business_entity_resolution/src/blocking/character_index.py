"""Bounded name character n-gram retrieval for typos and collapsed spaces."""

from .token_index import TokenIndex


class CharacterIndex(TokenIndex):
    def __init__(self, records, statistics):
        super().__init__(records, "character", statistics)
